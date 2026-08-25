"""
/runs routes. This file starts with just POST /runs/build (Workflow A
upload) -- other endpoints (configure, execute, status, etc.) are added
incrementally in later steps.
"""

import os
import tempfile
import warnings
import hashlib
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, Form, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db, get_session
from app.db import repository as repo
from app.core.context_store import set_context, get_context, delete_context
from app.core.exceptions import DatasetValidationError, ConfigValidationError
from app.core.pipeline_orchestrator import evaluate_original_model, run_mitigations
from app.modules.ingestion.dataset_loader import load_csv
from app.modules.ingestion.upload_handler import handle_existing_model_upload
from app.modules.training.trainer import train_new_model
from app.modules.fairness.metrics import validate_protected_attribute_config
from app.modules.fairness.dataset_utils import needs_group_restriction, restrict_df_to_groups
from app.modules.reporting.report_builder import build_report

router = APIRouter(prefix="/runs", tags=["runs"])

UPLOAD_DIR = Path(tempfile.gettempdir()) / "rai_platform_uploads"


class ConfigureRequest(BaseModel):
    protected_attribute: str
    privileged_value: str
    unprivileged_value: str


@router.post("/build")
def build_and_analyze(
    dataset: UploadFile,
    target_column: str = Form(...),
    algorithm_name: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Workflow A: upload a raw dataset, pick a target column + algorithm.
    The backend trains a Pipeline internally and returns a run_id.
    Configure (protected attribute) and execute (run the RAI pipeline)
    happen in separate follow-up calls.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = UPLOAD_DIR / f"{dataset.filename}"
    dataset_bytes = dataset.file.read()
    with open(dataset_path, "wb") as f:
        f.write(dataset_bytes)
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

    df = load_csv(dataset_path, name="Dataset")

    context = train_new_model(df, target_column=target_column, algorithm_name=algorithm_name)

    run = repo.create_run(
        db,
        workflow_type="build_and_analyze",
        status="pending",
        target_column=target_column,
        dataset_name=dataset.filename,
        dataset_hash=dataset_hash,
    )
    set_context(run.run_id, context)

    return {
        "run_id": run.run_id,
        "workflow_type": run.workflow_type,
        "status": run.status,
        "algorithm_name": context.algorithm_name,
        "train_rows": len(context.train_df),
        "test_rows": len(context.test_df),
        "validation_warnings": context.validation_warnings,
    }


@router.post("/analyze")
def analyze_existing_model(
    model: UploadFile,
    train_dataset: UploadFile,
    test_dataset: UploadFile,
    target_column: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Workflow B: upload a previously trained model (.pkl) plus its
    training and testing datasets. The backend validates the model
    (fitted-state check, supported-algorithm check, Pipeline vs bare
    estimator detection and wrapping) before accepting it -- see
    ingestion.upload_handler for the full validation chain.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    model_path = UPLOAD_DIR / f"{model.filename}"
    with open(model_path, "wb") as f:
        f.write(model.file.read())

    train_path = UPLOAD_DIR / f"train_{train_dataset.filename}"
    train_bytes = train_dataset.file.read()
    with open(train_path, "wb") as f:
        f.write(train_bytes)
    dataset_hash = hashlib.sha256(train_bytes).hexdigest()

    test_path = UPLOAD_DIR / f"test_{test_dataset.filename}"
    with open(test_path, "wb") as f:
        f.write(test_dataset.file.read())

    context = handle_existing_model_upload(
        str(model_path), str(train_path), str(test_path), target_column
    )

    run = repo.create_run(
        db,
        workflow_type="analyze_existing",
        status="pending",
        target_column=target_column,
        dataset_name=train_dataset.filename,
        dataset_hash=dataset_hash,
    )
    set_context(run.run_id, context)

    return {
        "run_id": run.run_id,
        "workflow_type": run.workflow_type,
        "status": run.status,
        "algorithm_name": context.algorithm_name,
        "preprocessing_status": context.preprocessing_status,
        "train_rows": len(context.train_df),
        "test_rows": len(context.test_df),
        "validation_warnings": context.validation_warnings,
    }


@router.post("/{run_id}/configure")
def configure_run(run_id: str, body: ConfigureRequest, db: Session = Depends(get_db)):
    """
    Sets the protected-attribute configuration for a run. Must be called
    after upload (build or analyze) and before execute.

    If the protected attribute has more than the two selected values
    (e.g. COMPAS 'race' with 6 categories, user picks 'White' vs
    'Black'), the run is scoped internally to just those two groups --
    AIF360's fairness metrics require a strictly binary protected
    attribute. For an internally-trained model this means retraining
    on the scoped subset (so V1 reflects the actual cohort under
    study); for an uploaded model (which can't be retrained) only
    evaluation/mitigation are scoped, and a validation_warning records
    exactly what was filtered -- nothing is silently dropped.

    Validates the configuration against the actual dataset before
    accepting it, using the same check fairness.metrics relies on
    later -- so a bad config is caught here, not partway through
    pipeline execution.
    """
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    context = get_context(run_id)
    if context is None:
        raise DatasetValidationError(
            f"No uploaded model/dataset found in memory for run '{run_id}'. "
            f"Re-upload via /runs/build or /runs/analyze."
        )

    combined_df = pd.concat([context.train_df, context.test_df], ignore_index=True)

    if needs_group_restriction(
        combined_df, body.protected_attribute, body.privileged_value, body.unprivileged_value
    ):
        group_count = int(combined_df[body.protected_attribute].nunique(dropna=True))
        original_rows = len(combined_df)
        scoped_df = restrict_df_to_groups(
            combined_df, body.protected_attribute, body.privileged_value, body.unprivileged_value
        )
        retained_rows = len(scoped_df)
        warning = (
            f"Protected attribute '{body.protected_attribute}' has {group_count} distinct "
            f"values; this run is scoped to '{body.privileged_value}' vs "
            f"'{body.unprivileged_value}' only ({retained_rows}/{original_rows} rows retained)."
        )

        if context.source == "internally_trained":
            context = train_new_model(
                scoped_df,
                target_column=context.target_column,
                algorithm_name=context.algorithm_name,
            )
            context.validation_warnings = context.validation_warnings + [
                warning + " Model retrained on the scoped subset."
            ]
        else:
            context.train_df = restrict_df_to_groups(
                context.train_df, body.protected_attribute, body.privileged_value, body.unprivileged_value
            )
            context.test_df = restrict_df_to_groups(
                context.test_df, body.protected_attribute, body.privileged_value, body.unprivileged_value
            )
            context.validation_warnings = context.validation_warnings + [
                warning + " This model was uploaded (not trained by the platform), so it "
                "was not retrained -- evaluation and mitigation are scoped to the selected groups."
            ]

        set_context(run_id, context)

    validate_protected_attribute_config(
        context, body.protected_attribute, body.privileged_value, body.unprivileged_value
    )

    run.protected_attribute = body.protected_attribute
    run.privileged_value = body.privileged_value
    run.unprivileged_value = body.unprivileged_value
    db.commit()
    db.refresh(run)

    return {
        "run_id": run.run_id,
        "protected_attribute": run.protected_attribute,
        "privileged_value": run.privileged_value,
        "unprivileged_value": run.unprivileged_value,
        "status": run.status,
        "validation_warnings": context.validation_warnings,
    }


def _evaluate_pipeline_task(run_id: str):
    """
    Background task for phase 1: evaluates and persists V1 only (no
    mitigation). Deliberately does NOT delete_context() on completion --
    phase 2 (_execute_pipeline_task, run via POST /execute) still needs
    the live TrainedModelContext to run mitigations against.
    """
    session = get_session()
    try:
        context = get_context(run_id)
        run = repo.get_run(session, run_id)
        if context is None or run is None:
            repo.update_run_status(
                session, run_id, status="failed", current_stage="failed",
                error_message="Context or run not found at evaluate time.",
            )
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluate_original_model(
                session, run_id, context,
                run.protected_attribute, run.privileged_value, run.unprivileged_value,
            )
    finally:
        session.close()


@router.post("/{run_id}/evaluate")
def evaluate_run(run_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Phase 1: evaluates and persists the original model (V1) only --
    performance, SHAP, error analysis, counterfactuals, fairness. No
    mitigation runs yet. Returns immediately; poll GET
    /runs/{run_id}/status until status is "original_ready", then fetch
    GET /runs/{run_id}/versions to show V1's full analysis before asking
    which mitigation methods to run via POST /execute.

    Requires /configure to have been called first.
    """
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    if not run.protected_attribute:
        raise ConfigValidationError(
            f"Run '{run_id}' has not been configured yet. "
            f"Call POST /runs/{run_id}/configure first."
        )

    if get_context(run_id) is None:
        raise DatasetValidationError(
            f"No uploaded model/dataset found in memory for run '{run_id}'. "
            f"This can happen after a server restart -- re-upload via "
            f"/runs/build or /runs/analyze."
        )

    repo.update_run_status(db, run_id, status="running", current_stage="queued")
    background_tasks.add_task(_evaluate_pipeline_task, run_id)

    return {"run_id": run_id, "status": "running", "current_stage": "queued"}


def _execute_pipeline_task(run_id: str, mitigation_methods: list[str] | None = None):
    """
    Background task for phase 2: runs the requested mitigation methods
    against the V1 already produced by phase 1 (POST /evaluate). Must
    open its own DB session (the request-scoped one from get_db() is
    already closed by the time this runs). Per SDD §13, this is the
    entire "job" -- FastAPI's BackgroundTasks stands in for a message
    queue/worker at MVP scale.

    mitigation_methods: list of registry method names selected by the
    caller (e.g. the frontend's multi-select). Passed straight through
    to run_mitigations, which already supports running several methods
    in one run_id (one sibling version each). If None, run_mitigations
    falls back to its single auto-suggested method, matching the
    platform's original single-mitigation behavior.
    """
    session = get_session()
    try:
        context = get_context(run_id)
        run = repo.get_run(session, run_id)
        v1_row = repo.get_original_version_for_run(session, run_id)
        if context is None or run is None or v1_row is None:
            repo.update_run_status(
                session, run_id, status="failed", current_stage="failed",
                error_message="Context, run, or original model version not found at execute time.",
            )
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_mitigations(
                session, run_id, context, v1_row,
                run.protected_attribute, run.privileged_value, run.unprivileged_value,
                mitigation_methods=mitigation_methods,
            )
    finally:
        # The live Pipeline objects are no longer needed once versioning
        # has persisted every version to disk -- drop them to avoid
        # unbounded memory growth across many runs on a long-lived server.
        delete_context(run_id)
        session.close()


@router.post("/{run_id}/execute")
def execute_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    mitigation_methods: list[str] | None = Form(None),
):
    """
    Phase 2: runs the requested mitigation method(s) against the
    original model. Returns immediately; poll GET /runs/{run_id}/status
    for progress. Requires POST /evaluate to have completed first
    (status "original_ready") -- V1 must already exist.

    mitigation_methods: optional list of registry method names to run
    (e.g. ["Reweighing", "Reject Option Classification"]). If omitted,
    falls back to the single auto-suggested method (prior behavior).
    """
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    if not run.protected_attribute:
        raise ConfigValidationError(
            f"Run '{run_id}' has not been configured yet. "
            f"Call POST /runs/{run_id}/configure first."
        )

    if repo.get_original_version_for_run(db, run_id) is None:
        raise ConfigValidationError(
            f"Run '{run_id}' has no evaluated original model yet. "
            f"Call POST /runs/{run_id}/evaluate first and wait for it "
            f"to complete before requesting mitigations."
        )

    if get_context(run_id) is None:
        raise DatasetValidationError(
            f"No uploaded model/dataset found in memory for run '{run_id}'. "
            f"This can happen after a server restart -- re-upload via "
            f"/runs/build or /runs/analyze."
        )

    repo.update_run_status(db, run_id, status="running", current_stage="queued")
    background_tasks.add_task(_execute_pipeline_task, run_id, mitigation_methods)

    return {"run_id": run_id, "status": "running", "current_stage": "queued"}


@router.get("/{run_id}/status")
def get_run_status(run_id: str, db: Session = Depends(get_db)):
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    return {
        "run_id": run.run_id,
        "status": run.status,
        "current_stage": run.current_stage,
        "error_message": run.error_message,
        "failed_methods": run.failed_methods or [],
    }


def _serialize_version(v) -> dict:
    return {
        "version_id": v.version_id,
        "version_number": v.version_number,
        "parent_version_id": v.parent_version_id,
        "source": v.source,
        "mitigation_method": v.mitigation_method,
        "mitigation_category": v.mitigation_category,
        "algorithm_name": v.algorithm_name,
        "preprocessing_status": v.preprocessing_status,
        "runtime_seconds": v.runtime_seconds,
        "mitigation_seconds": v.mitigation_seconds,
        "analysis_seconds": v.analysis_seconds,
        "performance_metrics": v.performance_metrics,
        "fairness_metrics": v.fairness_metrics,
        "fairness_finding": v.fairness_finding,
        "explainability_results": v.explainability_results,
        "error_analysis_results": v.error_analysis_results,
        "counterfactual_results": v.counterfactual_results,
        "narrative_summary": v.narrative_summary,
        "has_downloadable_model": bool(v.artifact_path),
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


@router.get("/{run_id}/versions")
def list_versions(run_id: str, db: Session = Depends(get_db)):
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    versions = sorted(repo.list_versions_for_run(db, run_id), key=lambda v: v.version_number)
    return {"run_id": run_id, "versions": [_serialize_version(v) for v in versions]}


@router.get("/{run_id}/comparison")
def compare_versions(run_id: str, db: Session = Depends(get_db)):
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    versions = sorted(repo.list_versions_for_run(db, run_id), key=lambda v: v.version_number)
    if len(versions) < 2:
        raise DatasetValidationError(
            "Comparison requires both an original and a debiased version "
            "to exist for this run. Has the pipeline finished running?",
            details={"versions_found": len(versions)},
        )
    v1, v2 = versions[0], versions[1]
    p1, p2 = v1.performance_metrics, v2.performance_metrics
    f1, f2 = v1.fairness_metrics, v2.fairness_metrics

    return {
        "run_id": run_id,
        "v1": _serialize_version(v1),
        "v2": _serialize_version(v2),
        "deltas": {
            "accuracy_change": round(p2.get("accuracy", 0) - p1.get("accuracy", 0), 4),
            "statistical_parity_difference_change": round(
                f2.get("statistical_parity_difference", 0) - f1.get("statistical_parity_difference", 0), 4
            ),
            "disparity_reduced": abs(f2.get("statistical_parity_difference", 0))
            < abs(f1.get("statistical_parity_difference", 0)),
        },
    }


@router.get("/{run_id}/report")
def get_report(run_id: str, db: Session = Depends(get_db)):
    """
    Returns the Responsible AI Report as a PDF. Generated once and
    cached on disk (run.report_path) -- repeated calls re-download the
    same file rather than rebuilding it, per SDD §16 ("performs no new
    computation" applies to re-serving too, not just first generation).
    """
    run = repo.get_run(db, run_id)
    if run is None:
        raise DatasetValidationError(f"Run '{run_id}' not found.")

    if run.report_path and os.path.exists(run.report_path):
        report_path = run.report_path
    else:
        report_path = build_report(db, run_id)

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="Responsible_AI_Report.pdf",
    )
