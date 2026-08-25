import { Card } from "../ui";
import { FAIRNESS_METRIC_INFO } from "../../api/fairnessMetricInfo";
import "./FairnessMetricsInfoBox.css";

export default function FairnessMetricsInfoBox() {
  return (
    <Card className="fairness-info-box">
      <h3 className="fairness-info-box__title">About these fairness metrics</h3>
      <p className="fairness-info-box__intro">
        Each metric below compares the privileged and unprivileged group on a different aspect of
        the model's predictions. You'll see these same metrics on every version's analysis and on
        the final comparison page.
      </p>
      <dl className="fairness-info-box__list">
        {Object.values(FAIRNESS_METRIC_INFO).map((m) => (
          <div key={m.label} className="fairness-info-box__item">
            <dt>
              {m.label}
              <span className="fairness-info-box__range">
                range {m.range} · ideal {m.ideal}
              </span>
            </dt>
            <dd>{m.description}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}
