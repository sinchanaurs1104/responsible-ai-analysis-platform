export default function ExplainabilitySection({ explainability }) {
  if (!explainability || !explainability.top_features?.length) return null;
  const { top_features, explained_sample_size, explainer_type } = explainability;
  const maxScore = Math.max(...top_features.map((f) => Math.abs(f.importance_score)), 1e-9);

  return (
    <section className="analysis-section">
      <h3>Feature importance (SHAP)</h3>
      <p className="analysis-section__meta">
        {explainer_type} explainer · {explained_sample_size} samples explained
      </p>
      <div className="shap-bars">
        {top_features.map((f) => (
          <div key={f.feature_name} className="shap-bar-row">
            <div className="shap-bar-row__label">{f.feature_name}</div>
            <div className="shap-bar-row__track">
              <div
                className="shap-bar-row__fill"
                style={{ width: `${(Math.abs(f.importance_score) / maxScore) * 100}%` }}
              />
            </div>
            <div className="shap-bar-row__value numeric">{f.importance_score.toFixed(4)}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
