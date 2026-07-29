import { useState, useEffect, useCallback } from "react";
import { listVersions, getReportUrl, getExperimentExportUrl, ApiError } from "../api/client";
import { useRun } from "../api/RunContext";
import { Card, PageHeader, Button, ErrorBanner } from "../components/ui";
import SummaryTable from "../components/compare/SummaryTable";
import PerformanceComparison from "../components/compare/PerformanceComparison";
import FairnessComparison from "../components/compare/FairnessComparison";
import ShapComparison from "../components/compare/ShapComparison";
import CounterfactualComparison from "../components/compare/CounterfactualComparison";

export default function ComparePage() {
  const { runId } = useRun();
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await listVersions(runId);
      setVersions(res.versions || []);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load versions.");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  if (!runId) {
    return (
      <div>
        <PageHeader eyebrow="Step 5" title="Comparison dashboard" />
        <Card>Run the pipeline first.</Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="Step 5"
        title="Comparison dashboard"
        description="Every version side-by-side -- original and every mitigation method that ran."
      />

      <ErrorBanner message={error} />
      {loading && <Card>Loading…</Card>}

      {!loading && versions.length > 0 && (
        <>
          <Card className="compare-section">
            <h2>Summary</h2>
            <SummaryTable versions={versions} />
          </Card>

          <Card className="compare-section">
            <h2>Performance</h2>
            <PerformanceComparison versions={versions} />
          </Card>

          <Card className="compare-section">
            <h2>Fairness</h2>
            <FairnessComparison versions={versions} />
          </Card>

          <Card className="compare-section">
            <h2>SHAP feature comparison</h2>
            <ShapComparison versions={versions} />
          </Card>

          <Card className="compare-section">
            <h2>Counterfactual summary</h2>
            <CounterfactualComparison versions={versions} />
          </Card>

          <Card className="compare-section">
            <h2>Reports &amp; exports</h2>
            <div className="compare-exports">
              <a href={getReportUrl(runId)} target="_blank" rel="noreferrer">
                <Button variant="secondary">Download Responsible AI report (PDF)</Button>
              </a>
              <a href={getExperimentExportUrl()} target="_blank" rel="noreferrer">
                <Button variant="secondary">Export experiment log (CSV)</Button>
              </a>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
