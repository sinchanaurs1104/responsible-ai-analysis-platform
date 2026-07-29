"""
Threshold constants for fairness classification.

Kept in their own file (per the SDD folder structure) so the cutoffs
can be tuned without touching insight_engine's logic. These are
standard, widely-cited defaults, not something the platform invents:
- Statistical Parity Difference: 0 is perfectly fair; |SPD| > 0.1 is a
  commonly used "high disparity" cutoff.
- Disparate Impact Ratio: the four-fifths (80%) rule is a long-standing
  legal/practical guideline (outside [0.8, 1.25] is a violation).
- Equal Opportunity / Average Odds Difference: same 0.1 convention as SPD.
"""

SPD_HIGH_THRESHOLD = 0.10
SPD_MODERATE_THRESHOLD = 0.05

DIR_LOWER_BOUND = 0.80
DIR_UPPER_BOUND = 1.25

EOD_HIGH_THRESHOLD = 0.10
AOD_HIGH_THRESHOLD = 0.10

MIN_GROUP_SIZE_WARNING = 30
"""Below this size, a group's metrics are flagged as low-confidence
rather than suppressed -- the user should still see the number, just
with an explicit caveat (SDD error-handling case: 'tiny subgroup')."""

# Mitigation lookup table -- one entry per driving factor, mapping the
# kind of disparity detected to the mitigation method best suited to it.
# Confidence varies by how directly the literature supports each pairing
# (see MITIGATION_CONFIDENCE below) -- this is not a uniformly
# "validated" table, and callers should not treat it as one:
#
# - statistical_parity_difference -> Reweighing: well-grounded. Kamiran &
#   Calders (2012) designed Reweighing specifically to satisfy
#   statistical/demographic parity.
# - average_odds_difference -> Calibrated Equalized Odds Postprocessing:
#   well-grounded. AIF360's average_odds_difference *is* the equalized-odds
#   metric (mean of TPR and FPR gaps), and Pleiss et al. (2017) -- the
#   paper behind this method -- explicitly targets equalized odds.
# - equal_opportunity_difference -> Reject Option Classification: weaker.
#   Kamiran et al. (2012) frame Reject Option Classification as supporting
#   several fairness notions (parity, equalized odds, equal opportunity)
#   depending on configuration, but it is not specifically an
#   equal-opportunity method -- this pairing is a reasonable inference,
#   not something backed by a study validating this exact pairing.
#
# Structured as a table (SDD Sec.18) so a new mitigation method is a new/
# changed entry, not a rewrite of insight_engine's logic.
MITIGATION_LOOKUP = {
    "statistical_parity_difference": "Reweighing",
    "average_odds_difference": "Calibrated Equalized Odds Postprocessing",
    "equal_opportunity_difference": "Reject Option Classification",
}

# Paired with MITIGATION_LOOKUP: "standard" for pairings with direct
# literature support, "experimental" for pairings that are a reasonable
# but not independently validated inference. Consumed by insight_engine
# so FairnessFinding.mitigation_confidence reflects this honestly instead
# of a blanket "standard".
MITIGATION_CONFIDENCE = {
    "statistical_parity_difference": "standard",
    "average_odds_difference": "standard",
    "equal_opportunity_difference": "experimental",
}

# (moderate_threshold, high_threshold) per candidate driving-factor metric,
# reusing the constants above so there is exactly one place each number
# is defined.
METRIC_THRESHOLDS = {
    "statistical_parity_difference": (SPD_MODERATE_THRESHOLD, SPD_HIGH_THRESHOLD),
    "equal_opportunity_difference": (SPD_MODERATE_THRESHOLD, EOD_HIGH_THRESHOLD),
    "average_odds_difference": (SPD_MODERATE_THRESHOLD, AOD_HIGH_THRESHOLD),
}
