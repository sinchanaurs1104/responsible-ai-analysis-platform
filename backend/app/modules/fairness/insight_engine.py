"""
Deterministic fairness insight engine.

This is "Layer 1" from the design decision on AI-assisted narrative
summaries: plain Python rules that compute a reproducible verdict from
FairnessMetrics. A future narrative module only rephrases a
FairnessFinding into prose -- it never decides the verdict itself. This
separation is what keeps the platform's fairness verdicts auditable and
non-hallucinatable.
"""

from app.modules.fairness.thresholds import METRIC_THRESHOLDS, MITIGATION_LOOKUP, MITIGATION_CONFIDENCE
from app.schemas.fairness import FairnessMetrics, FairnessFinding

CANDIDATE_DRIVING_FACTORS = [
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "average_odds_difference",
]


def _select_driving_factor(metrics: FairnessMetrics) -> str:
    """
    Picks whichever of SPD/EOD/AOD has the largest absolute deviation
    from 0 (perfectly fair). This is what let the earlier probe example
    correctly identify SPD as the driving metric even when EOD/AOD sat
    at 0 -- the metric with the real signal wins, not a fixed priority
    order.

    EOD/AOD can be None (undefined -- see fairness.metrics._safe_metric)
    when a subgroup has no actual positive-label instances. Treated as
    zero deviation here, so an undefined metric never gets picked over
    a defined one; statistical_parity_difference is never None, so it's
    always available as the fallback.
    """
    deviations = {}
    for factor in CANDIDATE_DRIVING_FACTORS:
        value = getattr(metrics, factor)
        deviations[factor] = abs(value) if value is not None else 0.0
    return max(deviations, key=deviations.get)


def _classify_status(primary_metric_value: float, driving_factor: str) -> str:
    moderate_threshold, high_threshold = METRIC_THRESHOLDS[driving_factor]
    magnitude = abs(primary_metric_value)
    if magnitude >= high_threshold:
        return "high_disparity"
    if magnitude >= moderate_threshold:
        return "moderate_disparity"
    return "fair"


def derive_fairness_finding(metrics: FairnessMetrics) -> FairnessFinding:
    driving_factor = _select_driving_factor(metrics)
    primary_metric_value = getattr(metrics, driving_factor)
    status = _classify_status(primary_metric_value, driving_factor)

    if metrics.unprivileged_selection_rate < metrics.privileged_selection_rate:
        disadvantaged_group = metrics.unprivileged_group_label
    elif metrics.privileged_selection_rate < metrics.unprivileged_selection_rate:
        disadvantaged_group = metrics.privileged_group_label
    else:
        disadvantaged_group = "neither group (selection rates are equal)"

    selection_rate_gap = round(
        abs(metrics.privileged_selection_rate - metrics.unprivileged_selection_rate), 4
    )

    if status == "fair":
        suggested_mitigation = "None required"
        mitigation_confidence = "standard"
    else:
        suggested_mitigation = MITIGATION_LOOKUP[driving_factor]
        mitigation_confidence = MITIGATION_CONFIDENCE[driving_factor]

    return FairnessFinding(
        status=status,
        primary_metric=driving_factor,
        primary_metric_value=round(float(primary_metric_value), 4),
        driving_factor=driving_factor,
        disadvantaged_group=disadvantaged_group,
        selection_rate_gap=selection_rate_gap,
        suggested_mitigation=suggested_mitigation,
        mitigation_confidence=mitigation_confidence,
    )
