"""
Fairness metric computation via AIF360.

Takes the protected attribute configuration explicitly (protected
attribute column name, which value is "privileged", which is
"unprivileged") -- this is the "Configure" step from the SDD user
journey (§5), not something this module infers on its own.
"""

import math

import pandas as pd
from aif360.metrics import ClassificationMetric

from app.core.exceptions import ConfigValidationError, DatasetValidationError
from app.modules.evaluation.metrics import _resolve_positive_class
from app.schemas.context import TrainedModelContext
from app.schemas.fairness import FairnessMetrics
from app.modules.fairness.thresholds import MIN_GROUP_SIZE_WARNING
from app.modules.fairness.dataset_utils import to_binary_label_dataset, encode_binary_columns


def _safe_metric(value) -> float | None:
    """
    AIF360 can legitimately return NaN or +/-Infinity for some metrics
    in edge cases (e.g. disparate_impact() divides by the privileged
    group's selection rate, which can be 0; equal_opportunity_difference
    relies on a group's true-positive rate, undefined if that group has
    no actual positives). Both are not valid JSON, so they're converted
    to None here rather than left to crash json.dumps() downstream, and
    the schema field is typed Optional to make "undefined" explicit
    rather than silently coercing it into a misleading number like 0.0.
    """
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def validate_protected_attribute_config(
    context: TrainedModelContext,
    protected_attribute: str,
    privileged_value: str,
    unprivileged_value: str,
) -> None:
    if protected_attribute not in context.test_df.columns:
        raise ConfigValidationError(
            f"Protected attribute column '{protected_attribute}' not "
            f"found in the dataset.",
            details={"available_columns": context.test_df.columns.tolist()},
        )
    if protected_attribute == context.target_column:
        raise ConfigValidationError(
            "Protected attribute cannot be the same column as the target."
        )

    observed_values = set(context.test_df[protected_attribute].dropna().unique().tolist())
    expected_values = {privileged_value, unprivileged_value}
    unexpected = observed_values - expected_values
    if unexpected:
        raise ConfigValidationError(
            f"Protected attribute column '{protected_attribute}' contains "
            f"values other than the configured privileged/unprivileged "
            f"groups. This platform's MVP supports a binary protected "
            f"attribute only.",
            details={
                "configured_values": list(expected_values),
                "unexpected_values": list(unexpected),
            },
        )


def compute_fairness_metrics(
    context: TrainedModelContext,
    protected_attribute: str,
    privileged_value: str,
    unprivileged_value: str,
) -> FairnessMetrics:
    validate_protected_attribute_config(
        context, protected_attribute, privileged_value, unprivileged_value
    )

    positive_class = _resolve_positive_class(context)
    favorable_label = 1
    unfavorable_label = 0

    X_test = context.test_df.drop(columns=[context.target_column])
    y_pred = context.pipeline.predict(X_test)

    working_df = encode_binary_columns(
        context.test_df, protected_attribute, context.target_column,
        privileged_value, unprivileged_value, positive_class,
    )

    pred_source_df = context.test_df.copy()
    pred_source_df[context.target_column] = pd.Series(y_pred, index=pred_source_df.index)
    pred_df = encode_binary_columns(
        pred_source_df, protected_attribute, context.target_column,
        privileged_value, unprivileged_value, positive_class,
    )

    true_dataset = to_binary_label_dataset(
        working_df, protected_attribute, context.target_column,
        favorable_label, unfavorable_label,
    )
    pred_dataset = to_binary_label_dataset(
        pred_df, protected_attribute, context.target_column,
        favorable_label, unfavorable_label,
    )

    privileged_groups = [{protected_attribute: 1}]
    unprivileged_groups = [{protected_attribute: 0}]

    cm = ClassificationMetric(
        true_dataset, pred_dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
    )

    privileged_group_size = int((working_df[protected_attribute] == 1).sum())
    unprivileged_group_size = int((working_df[protected_attribute] == 0).sum())

    if privileged_group_size == 0 or unprivileged_group_size == 0:
        raise DatasetValidationError(
            "One of the configured privileged/unprivileged groups has "
            "zero rows in the test dataset; fairness metrics cannot be "
            "computed.",
            details={
                "privileged_group_size": privileged_group_size,
                "unprivileged_group_size": unprivileged_group_size,
            },
        )

    small_group_warning = (
        privileged_group_size < MIN_GROUP_SIZE_WARNING
        or unprivileged_group_size < MIN_GROUP_SIZE_WARNING
    )

    return FairnessMetrics(
        protected_attribute=protected_attribute,
        privileged_group_label=privileged_value,
        unprivileged_group_label=unprivileged_value,
        privileged_group_size=privileged_group_size,
        unprivileged_group_size=unprivileged_group_size,
        privileged_selection_rate=round(float(cm.selection_rate(privileged=True)), 4),
        unprivileged_selection_rate=round(float(cm.selection_rate(privileged=False)), 4),
        statistical_parity_difference=round(float(cm.statistical_parity_difference()), 4),
        disparate_impact_ratio=_safe_metric(cm.disparate_impact()),
        equal_opportunity_difference=_safe_metric(cm.equal_opportunity_difference()),
        average_odds_difference=_safe_metric(cm.average_odds_difference()),
        theil_index=_safe_metric(cm.theil_index()),
        small_group_warning=small_group_warning,
    )
