// Range and ideal-value reference for each AIF360 fairness metric this
// platform surfaces. Kept in one place so every UI location that shows
// a fairness metric (version analysis, compare page, mitigate page info
// box) describes it identically.
export const FAIRNESS_METRIC_INFO = {
  statistical_parity_difference: {
    label: "Statistical Parity Difference",
    range: "-1 to 1",
    ideal: "0",
    description:
      "Difference in selection rate between the privileged and unprivileged group. 0 means both groups are selected at the same rate.",
  },
  disparate_impact_ratio: {
    label: "Disparate Impact Ratio",
    range: "0 to ∞ (typically 0-2)",
    ideal: "1",
    description:
      "Ratio of the unprivileged group's selection rate to the privileged group's. Commonly flagged as concerning below 0.8 or above 1.25 (the \"four-fifths rule\").",
  },
  equal_opportunity_difference: {
    label: "Equal Opportunity Difference",
    range: "-1 to 1",
    ideal: "0",
    description:
      "Difference in true positive rate (recall) between groups -- whether qualified members of each group are equally likely to be correctly selected.",
  },
  average_odds_difference: {
    label: "Average Odds Difference",
    range: "-1 to 1",
    ideal: "0",
    description:
      "Average of the true positive rate and false positive rate differences between groups -- a broader error-rate parity check than equal opportunity alone.",
  },
  theil_index: {
    label: "Theil Index",
    range: "0 to ∞ (unbounded)",
    ideal: "0",
    description:
      "General entropy-based inequality measure across individuals, not just group averages. 0 means perfect equality; higher values mean more inequality.",
  },
};

export function fairnessRangeHint(metricKey) {
  const info = FAIRNESS_METRIC_INFO[metricKey];
  if (!info) return undefined;
  return `Range: ${info.range} · Ideal: ${info.ideal}`;
}
