import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { listVersions, ApiError } from "../api/client";
import { useRun } from "../api/RunContext";
import { Card, PageHeader, Button, ErrorBanner, WarningBanner } from "../components/ui";
import VersionAnalysis from "../components/version-analysis/VersionAnalysis";

export default function VersionsPage() {
  const { runId, failedMethods, patch, markStageReached } = useRun();
  const [versions, setVersions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await listVersions(runId);
      setVersions(res.versions || []);
      patch({ versions: res.versions || [] });
      if (res.versions?.length) {
        setActiveId((prev) => prev || res.versions[0].version_id);
      }
      markStageReached("compare");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load versions.");
    } finally {
      setLoading(false);
    }
  }, [runId, patch, markStageReached]);

  useEffect(() => {
    load();
  }, [load]);

  if (!runId) {
    return (
      <div>
        <PageHeader eyebrow="Step 4" title="Versions" />
        <Card>Run the pipeline first.</Card>
      </div>
    );
  }

  const active = versions.find((v) => v.version_id === activeId);

  return (
    <div>
      <PageHeader
        eyebrow="Step 4"
        title="Model versions"
        description="One tab per version -- the original model plus every mitigation method that ran. Each tab shows the full analysis: performance, fairness, SHAP, error cohorts, and counterfactuals."
      />

      <ErrorBanner message={error} />
      <WarningBanner
        warnings={(failedMethods || []).map((f) => `${f.method} did not complete: ${f.error}`)}
      />

      {loading && <Card>Loading versions…</Card>}

      {!loading && versions.length > 0 && (
        <>
          <div className="version-tabs">
            {versions.map((v) => (
              <button
                key={v.version_id}
                className={`version-tab ${v.version_id === activeId ? "version-tab--active" : ""}`}
                onClick={() => setActiveId(v.version_id)}
              >
                {v.mitigation_method || "Original"}
              </button>
            ))}
          </div>

          <VersionAnalysis version={active} />

          {versions.length > 1 && (
            <div style={{ marginTop: "24px" }}>
              <Link to="/compare">
                <Button>Go to comparison dashboard</Button>
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}
