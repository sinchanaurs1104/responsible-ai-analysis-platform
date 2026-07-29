export default function ShapComparison({ versions }) {
  const withShap = versions.filter((v) => v.explainability_results?.top_features?.length);
  if (withShap.length === 0) return <p className="analysis-section__meta">No SHAP results available.</p>;

  // union of top feature names across versions, ranked by how often they appear
  const featureRank = new Map();
  withShap.forEach((v) => {
    v.explainability_results.top_features.forEach((f, idx) => {
      featureRank.set(f.feature_name, (featureRank.get(f.feature_name) || 0) + (10 - idx));
    });
  });
  const orderedFeatures = Array.from(featureRank.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name]) => name);

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Feature</th>
          {withShap.map((v) => (
            <th key={v.version_id}>{v.mitigation_method || "Original"}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {orderedFeatures.map((featureName) => (
          <tr key={featureName}>
            <td>{featureName}</td>
            {withShap.map((v) => {
              const match = v.explainability_results.top_features.find((f) => f.feature_name === featureName);
              return (
                <td key={v.version_id} className="numeric">
                  {match ? match.importance_score.toFixed(4) : "—"}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
