import { fmtPct } from "./PerformanceSection";

export default function ErrorAnalysisSection({ errorAnalysis }) {
  if (!errorAnalysis || !errorAnalysis.worst_subgroups?.length) return null;
  const { overall_accuracy, worst_subgroups } = errorAnalysis;

  return (
    <section className="analysis-section">
      <h3>Error analysis &amp; cohorts</h3>
      <p className="analysis-section__meta">Overall accuracy: {fmtPct(overall_accuracy)}</p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Subgroup</th>
            <th>Size</th>
            <th>Subgroup accuracy</th>
            <th>Gap vs. overall</th>
          </tr>
        </thead>
        <tbody>
          {worst_subgroups.map((s, i) => (
            <tr key={i}>
              <td>{s.column}</td>
              <td>{s.subgroup}</td>
              <td className="numeric">{s.subgroup_size}</td>
              <td className="numeric">{fmtPct(s.subgroup_accuracy)}</td>
              <td className={`numeric ${s.accuracy_gap > 0 ? "data-table__cell--warn" : ""}`}>
                {s.accuracy_gap > 0 ? "+" : ""}
                {fmtPct(s.accuracy_gap)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
