"""
Drives the Responsible AI pipeline (SDD Sec.5, Sec.13), split into two
callable phases so the platform can show the original model's full
analysis before the person picks mitigation methods, rather than
computing everything (original model + all mitigations) in one blocking
call:

  Phase 1 -- evaluate_original_model(): Evaluate V1 -> Explainability V1
  -> Fairness V1. Creates the V1 version and leaves the run at status
  "original_ready".

  Phase 2 -- run_mitigations(): given an existing V1 (from phase 1) and
  a list of mitigation method names, runs each one -> Retrain/wrap ->
  Evaluate -> Explainability -> Fairness for each, producing one sibling
  version per method. Leaves the run at status "completed".

run_pipeline() composes both phases in one call, unchanged in signature
and behavior, for callers that don't need the two-step UX (e.g.
scripts/run_experiment_batch.py, which runs many combinations
unattended and has no "user" to show V1 to in between).

Takes an already-built TrainedModelContext (produced by ingestion or
training before this runs) plus the protected-attribute configuration.
Updates Run.status/current_stage after every step and never lets an
exception escape uncaught -- on failure, Run.status becomes "failed"
with error_message set, matching the SDD's error-handling strategy.
"""

import time
import warnings

from sqlalchemy.orm import Session

from app.core.exceptions import ValidationError
from app.db import repository as repo
from app.db.models import ModelVersion
from app.schemas.context import TrainedModelContext
from app.schemas.fairness import FairnessFinding

from app.modules.evaluation.metrics import evaluate_model, _resolve_positive_class
from app.modules.explainability.shap_explainer import compute_shap_importance
from app.modules.explainability.error_analysis import analyze_errors
from app.modules.explainability.counterfactuals import generate_counterfactuals
from app.modules.fairness.metrics import compute_fairness_metrics
from app.modules.fairness.insight_engine import derive_fairness_finding
from app.modules.mitigation.registry import get_registration, CATEGORY_PREPROCESSING, CATEGORY_POSTPROCESSING
from app.modules.retraining.model_cloner import retrain_with_preprocessing_result
from app.modules.training.trainer import RANDOM_STATE
from app.modules.versioning.version_manager import create_version


def _analyze_version(context: TrainedModelContext, protected_attribute, privileged_value, unprivileged_value):
    """Runs evaluation + explainability + fairness for one version. Shared
    by both the V1 and V2 passes so the two stay identical by construction.

    Also times itself: SHAP + DiCE counterfactual generation are typically
    the slowest parts of this function and that cost is constant -- every
    mitigation method pays it regardless of what the mitigation itself
    does. Returning analysis_seconds lets callers report mitigation-only
    runtime separately from this constant analysis overhead, instead of
    a combined figure that understates the true gap between methods.
    """
    start_time = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        performance = evaluate_model(context)
        shap_result = compute_shap_importance(context)
        error_analysis = analyze_errors(context)
        counterfactuals = generate_counterfactuals(context, protected_attribute)
        fairness_metrics = compute_fairness_metrics(
            context, protected_attribute, privileged_value, unprivileged_value
        )
        finding = derive_fairness_finding(fairness_metrics)
    analysis_seconds = round(time.perf_counter() - start_time, 4)
    return performance, shap_result, error_analysis, counterfactuals, fairness_metrics, finding, analysis_seconds


def _run_one_mitigation(session, run_id, context, v1_row, method_name, protected_attribute,
                         privileged_value, unprivileged_value, positive_class, random_seed):
    """Runs one mitigation method end-to-end, producing one child version
    of v1_row. Raises on failure -- caller decides whether to skip or abort.

    Times the mitigation-specific step (transform+retrain, or wrap)
    separately from the analysis step (_analyze_version), since the
    latter is a roughly-constant cost every method pays regardless of
    what the mitigation itself does. A single combined runtime figure
    would understate the true gap between e.g. a pre-processing method's
    transform+retrain cost and a post-processing method's near-instant
    wrap cost.
    """
    mitigation_start = time.perf_counter()
    registration = get_registration(method_name)

    if registration.category == CATEGORY_PREPROCESSING:
        preprocessing_result = registration.strategy.apply(
            context.train_df, protected_attribute, context.target_column,
            privileged_value, unprivileged_value, positive_class,
        )
        context_v2 = retrain_with_preprocessing_result(context, preprocessing_result)
    elif registration.category == CATEGORY_POSTPROCESSING:
        wrapped_model = registration.strategy.wrap(
            context.pipeline, context.train_df, protected_attribute, context.target_column,
            privileged_value, unprivileged_value, positive_class,
        )
        context_v2 = TrainedModelContext(
            pipeline=wrapped_model, source=context.source,
            preprocessing_status=context.preprocessing_status,
            estimator_step_name=context.estimator_step_name,
            algorithm_name=context.algorithm_name,
            train_df=context.train_df, test_df=context.test_df,
            target_column=context.target_column,
            validation_warnings=context.validation_warnings,
        )
    else:
        raise ValidationError(
            f"Mitigation category '{registration.category}' is not yet "
            f"executable by this platform (method: '{method_name}')."
        )
    mitigation_seconds = round(time.perf_counter() - mitigation_start, 4)

    perf2, shap2, err2, cf2, fair2, finding2, analysis_seconds = _analyze_version(
        context_v2, protected_attribute, privileged_value, unprivileged_value
    )
    return create_version(
        session, run_id, context_v2, perf2, version_number=2,
        fairness_metrics=fair2, fairness_finding=finding2,
        shap_result=shap2, error_analysis_result=err2, counterfactual_result=cf2,
        mitigation_method=registration.strategy.name,
        mitigation_category=registration.category,
        mitigation_hyperparameters=registration.strategy.get_hyperparameters(),
        mitigation_seconds=mitigation_seconds,
        analysis_seconds=analysis_seconds,
        random_seed=random_seed,
        parent_version_id=v1_row.version_id,
    )


def evaluate_original_model(
    session: Session,
    run_id: str,
    context: TrainedModelContext,
    protected_attribute: str,
    privileged_value: str,
    unprivileged_value: str,
    random_seed: int | None = None,
) -> ModelVersion:
    """
    Phase 1: evaluates and persists V1 only (performance, SHAP, error
    analysis, counterfactuals, fairness) -- no mitigation is run. Leaves
    the run at status "original_ready" so the frontend can show the
    original model's full analysis before the person picks mitigation
    methods.

    random_seed: the seed actually used to produce `context` (train/test
    split + estimator), passed through here purely for recording on the
    ModelVersion. If omitted, falls back to trainer.RANDOM_STATE.
    """
    try:
        effective_seed = random_seed if random_seed is not None else RANDOM_STATE
        repo.update_run_status(session, run_id, status="running", current_stage="evaluating_v1")

        privileged_count = int((context.test_df[protected_attribute] == privileged_value).sum())
        unprivileged_count = int((context.test_df[protected_attribute] == unprivileged_value).sum())
        repo.record_run_sample_stats(
            session, run_id,
            train_row_count=len(context.train_df),
            test_row_count=len(context.test_df),
            test_privileged_count=privileged_count,
            test_unprivileged_count=unprivileged_count,
        )

        perf1, shap1, err1, cf1, fair1, finding1, analysis_seconds1 = _analyze_version(
            context, protected_attribute, privileged_value, unprivileged_value
        )

        repo.update_run_status(session, run_id, status="running", current_stage="saving_version_1")
        v1_row = create_version(
            session, run_id, context, perf1, version_number=1,
            fairness_metrics=fair1, fairness_finding=finding1,
            shap_result=shap1, error_analysis_result=err1, counterfactual_result=cf1,
            analysis_seconds=analysis_seconds1,
            random_seed=effective_seed if context.source == "internally_trained" else None,
        )

        repo.update_run_status(session, run_id, status="original_ready", current_stage="original_ready")
        return v1_row

    except ValidationError as exc:
        repo.update_run_status(
            session, run_id, status="failed", current_stage="failed",
            error_message=exc.message,
        )
        raise
    except Exception as exc:  # noqa: BLE001 -- last-resort catch per SDD Sec.13/14
        repo.update_run_status(
            session, run_id, status="failed", current_stage="failed",
            error_message=f"Unexpected error: {exc}",
        )
        raise


def run_mitigations(
    session: Session,
    run_id: str,
    context: TrainedModelContext,
    v1_row: ModelVersion,
    protected_attribute: str,
    privileged_value: str,
    unprivileged_value: str,
    mitigation_methods: list[str] | None = None,
    skip_failed_methods: bool = True,
    random_seed: int | None = None,
) -> list[ModelVersion]:
    """
    Phase 2: runs each requested mitigation method against the already-
    evaluated v1_row, producing one sibling version per method. Leaves
    the run at status "completed".

    mitigation_methods: list of registry method names to run, each
    producing a sibling child version of V1 (e.g.
    ["Reweighing", "Disparate Impact Remover"]). If None, defaults to
    the single method fairness.thresholds would auto-suggest for V1's
    fairness finding -- identical to the platform's original
    single-mitigation behavior.

    skip_failed_methods: if True (default), a method that raises is
    logged and skipped rather than aborting the whole run -- useful when
    running many methods across many datasets for research data
    collection, so one broken combination doesn't lose the rest.
    """
    try:
        effective_seed = random_seed if random_seed is not None else RANDOM_STATE

        finding1 = FairnessFinding.model_validate(v1_row.fairness_finding)
        methods = mitigation_methods if mitigation_methods else [finding1.suggested_mitigation]
        positive_class = _resolve_positive_class(context)

        child_rows = []
        for method_name in methods:
            repo.update_run_status(session, run_id, status="running", current_stage=f"mitigating:{method_name}")
            try:
                child_row = _run_one_mitigation(
                    session, run_id, context, v1_row, method_name,
                    protected_attribute, privileged_value, unprivileged_value, positive_class,
                    effective_seed,
                )
                child_rows.append(child_row)
            except Exception as exc:  # noqa: BLE001
                if not skip_failed_methods:
                    raise
                repo.append_run_failure(session, run_id, method=method_name, error=str(exc))

        repo.update_run_status(session, run_id, status="completed", current_stage="completed")
        return child_rows

    except ValidationError as exc:
        repo.update_run_status(
            session, run_id, status="failed", current_stage="failed",
            error_message=exc.message,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        repo.update_run_status(
            session, run_id, status="failed", current_stage="failed",
            error_message=f"Unexpected error: {exc}",
        )
        raise


def run_pipeline(
    session: Session,
    run_id: str,
    context: TrainedModelContext,
    protected_attribute: str,
    privileged_value: str,
    unprivileged_value: str,
    mitigation_methods: list[str] | None = None,
    skip_failed_methods: bool = True,
    random_seed: int | None = None,
):
    """
    Convenience wrapper composing both phases in one call, for callers
    that don't need the two-step UX (e.g. scripts/run_experiment_batch.py).
    Signature and behavior unchanged from before the phase 1/phase 2 split.
    """
    v1_row = evaluate_original_model(
        session, run_id, context, protected_attribute, privileged_value, unprivileged_value,
        random_seed=random_seed,
    )
    child_rows = run_mitigations(
        session, run_id, context, v1_row, protected_attribute, privileged_value, unprivileged_value,
        mitigation_methods=mitigation_methods, skip_failed_methods=skip_failed_methods,
        random_seed=random_seed,
    )
    return v1_row, child_rows
