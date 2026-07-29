import ScaleGlyph from "../ScaleGlyph";

export default function FairnessComparison({ versions }) {
  return (
    <div className="fairness-compare-list">
      {versions.map((v) => {
        const spd = v.fairness_metrics?.statistical_parity_difference;
        return (
          <div key={v.version_id} className="fairness-compare-row">
            <div className="fairness-compare-row__label">{v.mitigation_method || "Original"}</div>
            <ScaleGlyph value={spd ?? 0} size="md" />
            <div className="fairness-compare-row__value numeric">
              {spd != null ? spd.toFixed(4) : "—"}
            </div>
          </div>
        );
      })}
      <p className="analysis-section__meta">
        Statistical parity difference -- 0 is perfect parity; further from 0 (either direction) means more disparity.
      </p>

      <table className="data-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Disparate impact ratio</th>
            <th>Equal opportunity diff.</th>
            <th>Average odds diff.</th>
            <th>Theil index</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => {
            const f = v.fairness_metrics || {};
            return (
              <tr key={v.version_id}>
                <td>{v.mitigation_method || "Original"}</td>
                <td className="numeric">{fmt(f.disparate_impact_ratio)}</td>
                <td className="numeric">{fmt(f.equal_opportunity_difference)}</td>
                <td className="numeric">{fmt(f.average_odds_difference)}</td>
                <td className="numeric">{fmt(f.theil_index)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v) {
  return v === null || v === undefined ? "undefined" : v.toFixed(4);
}
