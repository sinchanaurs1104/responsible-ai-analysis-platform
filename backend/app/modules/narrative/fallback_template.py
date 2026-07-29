"""
Fallback narrative renderer. Used when the LLM is unavailable (no API
key, network failure, or failed output validation) so the fairness
summary card always renders something -- LLM narration is additive
polish, never a hard dependency (SDD narrative design decision).

Builds the exact same 5-part structure the LLM path targets, straight
from the already-computed FairnessFinding -- no new information is
introduced here, same as the LLM path is required to do.
"""

from app.schemas.fairness import FairnessFinding


STATUS_LABELS = {
    "high_disparity": "HIGH DISPARITY DETECTED",
    "moderate_disparity": "MODERATE DISPARITY DETECTED",
    "fair": "NO SIGNIFICANT DISPARITY DETECTED",
}

METRIC_DISPLAY_NAMES = {
    "statistical_parity_difference": "Statistical Parity Difference",
    "equal_opportunity_difference": "Equal Opportunity Difference",
    "average_odds_difference": "Average Odds Difference",
}


def render_fallback_summary(finding: FairnessFinding) -> str:
    metric_display = METRIC_DISPLAY_NAMES.get(finding.primary_metric, finding.primary_metric)
    status_line = STATUS_LABELS.get(finding.status, finding.status)

    if finding.status == "fair":
        return (
            f"Fairness Status\n{status_line}\n\n"
            f"Primary Metric\n{metric_display} = {finding.primary_metric_value}\n\n"
            f"Recommendation\nNo mitigation is currently required for this "
            f"model with respect to the configured protected attribute."
        )

    return (
        f"Fairness Status\n{status_line}\n\n"
        f"Primary Cause\n{metric_display} = {finding.primary_metric_value}. "
        f"Selection rate differs by {finding.selection_rate_gap * 100:.1f}% "
        f"between groups, disadvantaging: {finding.disadvantaged_group}.\n\n"
        f"Suggested Mitigation\n{finding.suggested_mitigation}\n\n"
        f"Recommendation\nRetrain the model using {finding.suggested_mitigation} "
        f"and compare results before deployment."
    )
