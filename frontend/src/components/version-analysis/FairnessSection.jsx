import { Metric, Badge } from "../ui";
import ScaleGlyph from "../ScaleGlyph";
import { fmtPct } from "./PerformanceSection";

const STATUS_TONE = {
  fair: "positive",
  moderate_disparity: "neutral",
  high_disparity: "negative",
};

const STATUS_LABEL = {
  fair: "Fair",
  moderate_disparity: "Moderate disparity",
  high_disparity: "High disparity",
};

export default function FairnessSection({ fairness, finding, isMitigated }) {
  if (!fairness) return null;

  return (
    <section className="analysis-section">
      <h3>Fairness</h3>

      {finding && (
        <div className="fairness-finding">
          <Badge tone={STATUS_TONE[finding.status] || "neutral"}>
            {STATUS_LABEL[finding.status] || finding.status}
          </Badge>
          <p>
            Driven by <strong>{finding.driving_factor.replace(/_/g, " ")}</strong> ({finding.primary_metric_value}),
            disadvantaging <strong>{finding.disadvantaged_group}</strong>. Selection-rate gap:{" "}
            {fmtPct(finding.selection_rate_gap)}.
          </p>
          {!isMitigated && finding.suggested_mitigation && (
            <p className="fairness-finding__suggestion">
              Suggested mitigation: {finding.suggested_mitigation}
              {finding.mitigation_confidence === "experimental" && (
                <span className="fairness-finding__experimental"> (experimental)</span>
              )}
            </p>
          )}
        </div>
      )}

      <div className="fairness-scale">
        <ScaleGlyph value={fairness.statistical_parity_difference} size="lg" labelled />
        <div className="fairness-scale__caption">
          Statistical parity difference:{" "}
          <span className="numeric">{fairness.statistical_parity_difference?.toFixed(4)}</span>
        </div>
      </div>

      {fairness.small_group_warning && (
        <p className="fairness-finding__warning">
          One or both groups are small in this test set -- treat these metrics as low-confidence.
        </p>
      )}

      <div className="metric-grid">
        <Metric label={`Selection rate (${fairness.privileged_group_label})`} value={fmtPct(fairness.privileged_selection_rate)} />
        <Metric label={`Selection rate (${fairness.unprivileged_group_label})`} value={fmtPct(fairness.unprivileged_selection_rate)} />
        <Metric label="Disparate impact ratio" value={fmtRatio(fairness.disparate_impact_ratio)} />
        <Metric label="Equal opportunity diff." value={fmtNum(fairness.equal_opportunity_difference)} />
        <Metric label="Average odds diff." value={fmtNum(fairness.average_odds_difference)} />
        <Metric label="Theil index" value={fmtNum(fairness.theil_index)} />
      </div>
    </section>
  );
}

function fmtRatio(v) {
  return v === null || v === undefined ? "undefined" : v.toFixed(3);
}
function fmtNum(v) {
  return v === null || v === undefined ? "undefined" : v.toFixed(4);
}
