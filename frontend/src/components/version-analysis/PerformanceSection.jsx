import { Metric } from "../ui";

export default function PerformanceSection({ performance }) {
  if (!performance) return null;
  const { accuracy, precision, recall, f1_score, confusion_matrix, test_set_size } = performance;

  return (
    <section className="analysis-section">
      <h3>Performance</h3>
      <div className="metric-grid">
        <Metric label="Accuracy" value={fmtPct(accuracy)} />
        <Metric label="Precision" value={fmtPct(precision)} />
        <Metric label="Recall" value={fmtPct(recall)} />
        <Metric label="F1" value={fmtPct(f1_score)} />
        <Metric label="Test set size" value={test_set_size} />
      </div>

      {confusion_matrix && (
        <div className="confusion-matrix">
          <div className="confusion-matrix__cell confusion-matrix__cell--head" />
          <div className="confusion-matrix__cell confusion-matrix__cell--head">Predicted −</div>
          <div className="confusion-matrix__cell confusion-matrix__cell--head">Predicted +</div>

          <div className="confusion-matrix__cell confusion-matrix__cell--head">Actual −</div>
          <div className="confusion-matrix__cell numeric">{confusion_matrix.true_negative}</div>
          <div className="confusion-matrix__cell numeric confusion-matrix__cell--warn">
            {confusion_matrix.false_positive}
          </div>

          <div className="confusion-matrix__cell confusion-matrix__cell--head">Actual +</div>
          <div className="confusion-matrix__cell numeric confusion-matrix__cell--warn">
            {confusion_matrix.false_negative}
          </div>
          <div className="confusion-matrix__cell numeric">{confusion_matrix.true_positive}</div>
        </div>
      )}
    </section>
  );
}

export function fmtPct(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
