// DEFAULT_NUM_INSTANCES in the backend's counterfactuals module -- the
// number of test rows DiCE attempts to explain per version. Not returned
// by the API, so mirrored here to compute a coverage rate. Keep in sync
// with app/modules/explainability/counterfactuals.py if that changes.
const ATTEMPTED_INSTANCES = 3;

export default function CounterfactualComparison({ versions }) {
  const rows = versions.map((v) => {
    const examples = v.counterfactual_results?.examples || [];
    const sparsities = examples.flatMap((ex) =>
      ex.counterfactual_instances.map((ci) => countChangedFeatures(ex.original_instance, ci))
    );
    const avgSparsity = sparsities.length
      ? sparsities.reduce((a, b) => a + b, 0) / sparsities.length
      : null;

    return {
      id: v.version_id,
      name: v.mitigation_method || "Original",
      coverage: examples.length,
      avgSparsity,
      method: v.counterfactual_results?.method || "—",
    };
  });

  return (
    <>
      <table className="data-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Coverage</th>
            <th>Avg. features changed</th>
            <th>Search method</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.name}</td>
              <td className="numeric">
                {r.coverage}/{ATTEMPTED_INSTANCES}
              </td>
              <td className="numeric">{r.avgSparsity != null ? r.avgSparsity.toFixed(1) : "—"}</td>
              <td>DiCE ({r.method})</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="analysis-section__meta">
        Coverage: how many of the {ATTEMPTED_INSTANCES} sampled test instances DiCE found a valid,
        genuinely flipping counterfactual for (the rest timed out or had none within budget). Avg.
        features changed: lower means more minimal, actionable counterfactuals.
      </p>
    </>
  );
}

function countChangedFeatures(original, counterfactual) {
  let changed = 0;
  for (const key of Object.keys(original)) {
    if (String(original[key]) !== String(counterfactual[key])) changed += 1;
  }
  return changed;
}
