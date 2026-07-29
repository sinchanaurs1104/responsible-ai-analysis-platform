import { Card, Badge, Button } from "../ui";
import { getDownloadUrl } from "../../api/client";
import PerformanceSection from "./PerformanceSection";
import FairnessSection from "./FairnessSection";
import ExplainabilitySection from "./ExplainabilitySection";
import ErrorAnalysisSection from "./ErrorAnalysisSection";
import CounterfactualSection from "./CounterfactualSection";
import "./version-analysis.css";

const CATEGORY_LABEL = { pre: "Pre-processing", in: "In-processing", post: "Post-processing" };

/**
 * Renders the full analysis for exactly one version -- reused for the
 * original model and for every mitigated version, whatever the backend
 * returns. No mitigation method names are hardcoded here.
 */
export default function VersionAnalysis({ version }) {
  if (!version) return null;
  const {
    version_number,
    source,
    mitigation_method,
    mitigation_category,
    algorithm_name,
    runtime_seconds,
    performance_metrics,
    fairness_metrics,
    fairness_finding,
    explainability_results,
    error_analysis_results,
    counterfactual_results,
    narrative_summary,
    has_downloadable_model,
    version_id,
  } = version;

  const title = mitigation_method || "Original model";
  const isMitigated = Boolean(mitigation_method);
  // Post-processing methods (Calibrated Equalized Odds, Reject Option
  // Classification) don't train a new model -- they wrap the original
  // estimator with a runtime threshold correction. An artifact technically
  // gets pickled for reproducibility, but offering it as "download the
  // debiased model" would misrepresent what post-processing actually does.
  const canDownload = has_downloadable_model && mitigation_category !== "post";

  return (
    <Card className="version-analysis">
      <div className="version-analysis__header">
        <div>
          <h2>{title}</h2>
          <div className="version-analysis__meta">
            <span>V{version_number}</span>
            <span>·</span>
            <span>{algorithm_name}</span>
            {mitigation_category && (
              <>
                <span>·</span>
                <Badge tone="accent">{CATEGORY_LABEL[mitigation_category] || mitigation_category}</Badge>
              </>
            )}
            {source === "uploaded" && <Badge tone="neutral">uploaded model</Badge>}
            {runtime_seconds != null && (
              <span className="numeric">{runtime_seconds.toFixed(2)}s runtime</span>
            )}
          </div>
        </div>
        {canDownload && (
          <a href={getDownloadUrl(version_id)} target="_blank" rel="noreferrer">
            <Button variant="secondary">Download model</Button>
          </a>
        )}
      </div>

      {narrative_summary && <p className="version-analysis__summary">{narrative_summary}</p>}

      <PerformanceSection performance={performance_metrics} />
      <FairnessSection fairness={fairness_metrics} finding={fairness_finding} isMitigated={isMitigated} />
      <ExplainabilitySection explainability={explainability_results} />
      <ErrorAnalysisSection errorAnalysis={error_analysis_results} />
      <CounterfactualSection counterfactual={counterfactual_results} />
    </Card>
  );
}
