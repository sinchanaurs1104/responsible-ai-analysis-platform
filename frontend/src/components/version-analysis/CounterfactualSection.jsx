import { useState } from "react";

export default function CounterfactualSection({ counterfactual }) {
  const [openIdx, setOpenIdx] = useState(null);
  if (!counterfactual || !counterfactual.examples?.length) return null;
  const { examples, method } = counterfactual;

  return (
    <section className="analysis-section">
      <h3>Counterfactual explanations</h3>
      <p className="analysis-section__meta">{examples.length} example(s) · method: {method}</p>
      <div className="cf-list">
        {examples.map((ex, i) => (
          <div key={i} className="cf-item">
            <button
              type="button"
              className="cf-item__toggle"
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
            >
              Instance {i + 1}: predicted <strong>{String(ex.original_prediction)}</strong> →{" "}
              <strong>{String(ex.counterfactual_prediction)}</strong>
              <span className="cf-item__chevron">{openIdx === i ? "−" : "+"}</span>
            </button>
            {openIdx === i && (
              <div className="cf-item__body">
                <CfTable title="Original" instance={ex.original_instance} />
                {ex.counterfactual_instances.map((ci, j) => (
                  <CfTable key={j} title={`Counterfactual ${j + 1}`} instance={ci} diffAgainst={ex.original_instance} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function CfTable({ title, instance, diffAgainst }) {
  return (
    <div className="cf-table">
      <div className="cf-table__title">{title}</div>
      <table className="data-table">
        <tbody>
          {Object.entries(instance).map(([k, v]) => {
            const changed = diffAgainst && String(diffAgainst[k]) !== String(v);
            return (
              <tr key={k}>
                <td>{k}</td>
                <td className={changed ? "data-table__cell--warn" : ""}>{String(v)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
