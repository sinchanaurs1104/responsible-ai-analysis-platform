"""
Fairness schemas.

FairnessMetrics is the raw numeric output of AIF360 -- no interpretation.
FairnessFinding is the deterministic, rule-based interpretation of those
numbers (SDD's "Layer 1" from the AI-assisted-summary design: computed
by plain Python rules, never by an LLM, so the verdict itself stays
reproducible and auditable). A future narrative module only rephrases
a FairnessFinding into prose -- it never computes one.
"""

from typing import Literal

from pydantic import BaseModel


class FairnessMetrics(BaseModel):
    protected_attribute: str
    privileged_group_label: str
    unprivileged_group_label: str
    privileged_group_size: int
    unprivileged_group_size: int
    privileged_selection_rate: float
    unprivileged_selection_rate: float
    statistical_parity_difference: float
    disparate_impact_ratio: float | None
    """None if undefined (division by zero -- the privileged group's
    selection rate was 0)."""
    equal_opportunity_difference: float | None
    """None if undefined (one group had no actual positive-label
    instances in the test set, making its true positive rate undefined)."""
    average_odds_difference: float | None
    """None if undefined, for the same reason as equal_opportunity_difference."""
    theil_index: float | None
    """None if undefined (can occur with degenerate prediction distributions)."""
    small_group_warning: bool
    """True if either group falls below MIN_GROUP_SIZE_WARNING -- metrics
    are still computed but should be treated as low-confidence."""


class FairnessFinding(BaseModel):
    status: Literal["fair", "moderate_disparity", "high_disparity"]
    primary_metric: str
    primary_metric_value: float
    driving_factor: Literal[
        "statistical_parity_difference",
        "equal_opportunity_difference",
        "average_odds_difference",
    ]
    disadvantaged_group: str
    selection_rate_gap: float
    """Absolute difference between privileged and unprivileged selection rates."""
    suggested_mitigation: str
    mitigation_confidence: Literal["standard", "experimental"]
