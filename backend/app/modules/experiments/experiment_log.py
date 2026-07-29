"""
Experiment log: a read-time flattening of Run + ModelVersion into one
CSV row per ModelVersion. Not a new table -- the database remains the
single source of truth; this is generated fresh on every export, never
stored or cached.
"""

import csv
import io
import json

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import Run, ModelVersion
from app.modules.explainability.counterfactuals import DEFAULT_NUM_INSTANCES

CSV_COLUMNS = [
    "version_id", "run_id", "created_at", "dataset_name", "dataset_hash",
    "protected_attribute", "privileged_value", "unprivileged_value",
    "algorithm_name", "source", "mitigation_method", "mitigation_category",
    "random_seed", "version_number", "parent_version_id",
    "train_row_count", "test_row_count",
    "test_privileged_count", "test_unprivileged_count",
    "runtime_seconds",
    "classifier_hyperparameters", "mitigation_hyperparameters",
    "accuracy", "precision", "recall", "f1_score",
    "statistical_parity_difference", "disparate_impact_ratio",
    "equal_opportunity_difference", "average_odds_difference", "theil_index",
    "fairness_status", "small_group_warning",
    "top_feature_1", "top_feature_1_score",
    "top_feature_2", "top_feature_2_score",
    "top_feature_3", "top_feature_3_score",
    "cf_instances_requested", "cf_valid_count", "cf_validity_rate",
    "cf_avg_features_changed",
    "cf_avg_features_changed_privileged", "cf_avg_features_changed_unprivileged",
]

FAILURE_CSV_COLUMNS = [
    "run_id", "dataset_name", "dataset_hash", "protected_attribute",
    "mitigation_method", "error_message",
]


def _count_changed_features(original: dict, counterfactual: dict) -> int:
    return sum(
        1 for k, v in counterfactual.items()
        if k in original and original[k] != v
    )


def _compute_cf_summary(version: ModelVersion, run: Run) -> dict:
    """
    Summarizes the raw counterfactual_results JSON into recourse-quality
    metrics that would otherwise never leave the DB blob:
      - validity: how many of the requested query instances actually
        produced a genuinely flipping counterfactual
      - cost: average number of feature values changed per counterfactual
        (sparsity/recourse-cost), overall and split by the original
        instance's protected-attribute group -- the group split is what
        makes this usable for recourse-fairness analysis later.
    """
    cf_result = version.counterfactual_results or {}
    examples = cf_result.get("examples", [])

    valid_count = len(examples)
    protected_attribute = run.protected_attribute

    all_deltas: list[int] = []
    privileged_deltas: list[int] = []
    unprivileged_deltas: list[int] = []

    for example in examples:
        original = example.get("original_instance", {})
        group_value = original.get(protected_attribute) if protected_attribute else None

        for cf_instance in example.get("counterfactual_instances", []):
            n_changed = _count_changed_features(original, cf_instance)
            all_deltas.append(n_changed)
            if group_value == run.privileged_value:
                privileged_deltas.append(n_changed)
            elif group_value == run.unprivileged_value:
                unprivileged_deltas.append(n_changed)

    def _avg(values: list[int]):
        return round(sum(values) / len(values), 4) if values else ""

    return {
        "cf_instances_requested": DEFAULT_NUM_INSTANCES,
        "cf_valid_count": valid_count,
        "cf_validity_rate": (
            round(valid_count / DEFAULT_NUM_INSTANCES, 4) if DEFAULT_NUM_INSTANCES else ""
        ),
        "cf_avg_features_changed": _avg(all_deltas),
        "cf_avg_features_changed_privileged": _avg(privileged_deltas),
        "cf_avg_features_changed_unprivileged": _avg(unprivileged_deltas),
    }


def _flatten(version: ModelVersion, run: Run) -> dict:
    perf = version.performance_metrics or {}
    fair = version.fairness_metrics or {}
    finding = version.fairness_finding or {}
    top_features = (version.explainability_results or {}).get("top_features", [])

    row = {
        "version_id": version.version_id,
        "run_id": version.run_id,
        "created_at": version.created_at.isoformat() if version.created_at else "",
        "dataset_name": run.dataset_name or "",
        "dataset_hash": run.dataset_hash or "",
        "protected_attribute": run.protected_attribute or "",
        "privileged_value": run.privileged_value or "",
        "unprivileged_value": run.unprivileged_value or "",
        "algorithm_name": version.algorithm_name,
        "source": version.source,
        "mitigation_method": version.mitigation_method or "",
        "mitigation_category": version.mitigation_category or "",
        "random_seed": version.random_seed if version.random_seed is not None else "",
        "version_number": version.version_number,
        "parent_version_id": version.parent_version_id or "",
        "train_row_count": run.train_row_count if run.train_row_count is not None else "",
        "test_row_count": run.test_row_count if run.test_row_count is not None else "",
        "test_privileged_count": (
            run.test_privileged_count if run.test_privileged_count is not None else ""
        ),
        "test_unprivileged_count": (
            run.test_unprivileged_count if run.test_unprivileged_count is not None else ""
        ),
        "runtime_seconds": version.runtime_seconds if version.runtime_seconds is not None else "",
        "classifier_hyperparameters": json.dumps(version.hyperparameters or {}),
        "mitigation_hyperparameters": json.dumps(version.mitigation_hyperparameters or {}),
        "accuracy": perf.get("accuracy", ""),
        "precision": perf.get("precision", ""),
        "recall": perf.get("recall", ""),
        "f1_score": perf.get("f1_score", ""),
        "statistical_parity_difference": fair.get("statistical_parity_difference", ""),
        "disparate_impact_ratio": fair.get("disparate_impact_ratio", ""),
        "equal_opportunity_difference": fair.get("equal_opportunity_difference", ""),
        "average_odds_difference": fair.get("average_odds_difference", ""),
        "theil_index": fair.get("theil_index", ""),
        "fairness_status": finding.get("status", ""),
        "small_group_warning": fair.get("small_group_warning", ""),
    }
    for i in range(3):
        if i < len(top_features):
            row[f"top_feature_{i+1}"] = top_features[i].get("feature_name", "")
            row[f"top_feature_{i+1}_score"] = top_features[i].get("importance_score", "")
        else:
            row[f"top_feature_{i+1}"] = ""
            row[f"top_feature_{i+1}_score"] = ""
    row.update(_compute_cf_summary(version, run))
    return row


def export_csv(session: Session) -> str:
    rows = session.execute(
        select(ModelVersion, Run).join(Run, ModelVersion.run_id == Run.run_id)
        .order_by(ModelVersion.created_at)
    ).all()

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for version, run in rows:
        writer.writerow(_flatten(version, run))
    return buffer.getvalue()


def export_failures_csv(session: Session) -> str:
    """
    Every mitigation method that raised and was skipped, across every
    run -- kept separate from the main version-level CSV since a
    failure has no ModelVersion row to attach to. Without this,
    dropped methods are invisible in the research dataset and a
    per-algorithm completion/failure rate can't be reported without
    rerunning the whole batch.
    """
    runs = session.execute(select(Run).order_by(Run.created_at)).scalars().all()

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FAILURE_CSV_COLUMNS)
    writer.writeheader()
    for run in runs:
        for failure in (run.failed_methods or []):
            writer.writerow({
                "run_id": run.run_id,
                "dataset_name": run.dataset_name or "",
                "dataset_hash": run.dataset_hash or "",
                "protected_attribute": run.protected_attribute or "",
                "mitigation_method": failure.get("method", ""),
                "error_message": failure.get("error", ""),
            })
    return buffer.getvalue()
