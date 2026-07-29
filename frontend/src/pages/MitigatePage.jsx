import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import { executeRun, getRunStatus, listVersions, ApiError } from "../api/client";
import { MITIGATION_METHODS } from "../api/constants";
import { useRun } from "../api/RunContext";
import { Card, PageHeader, Button, ErrorBanner, Badge } from "../components/ui";
import VersionAnalysis from "../components/version-analysis/VersionAnalysis";

const POLL_INTERVAL_MS = 2000;

export default function MitigatePage() {
  const navigate = useNavigate();
  const { runId, protectedAttribute, privilegedValue, unprivilegedValue, status, currentStage, patch, markStageReached } =
    useRun();

  const [original, setOriginal] = useState(null);
  const [loadingOriginal, setLoadingOriginal] = useState(true);
  const [selected, setSelected] = useState(MITIGATION_METHODS.map((m) => m.name));
  const [submitting, setSubmitting] = useState(false);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState(null);
  const pollTimer = useRef(null);

  // The original model (V1) was already evaluated on the Configure page
  // (POST /evaluate) before landing here -- fetch and show its full
  // analysis so mitigation method choices can be made with it in view,
  // rather than picking blind.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    (async () => {
      setLoadingOriginal(true);
      try {
        const res = await listVersions(runId);
        if (cancelled) return;
        const v1 = (res.versions || []).find((v) => !v.mitigation_method) || null;
        setOriginal(v1);
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load the original model.");
      } finally {
        if (!cancelled) setLoadingOriginal(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const toggle = (name) => {
    setSelected((prev) => (prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]));
  };

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
    setPolling(false);
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const startPolling = useCallback(
    (id) => {
      setPolling(true);
      pollTimer.current = setInterval(async () => {
        try {
          const res = await getRunStatus(id);
          patch({
            status: res.status,
            currentStage: res.current_stage,
            errorMessage: res.error_message,
            failedMethods: res.failed_methods || [],
          });
          if (res.status === "completed") {
            stopPolling();
            markStageReached("versions");
            navigate("/versions");
          } else if (res.status === "failed") {
            stopPolling();
            setError(res.error_message || "The pipeline failed. See run status for details.");
          }
        } catch (err) {
          stopPolling();
          setError(err instanceof ApiError ? err.message : "Lost connection while checking run status.");
        }
      }, POLL_INTERVAL_MS);
    },
    [patch, markStageReached, navigate, stopPolling]
  );

  const handleRun = async () => {
    if (selected.length === 0 || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await executeRun(runId, selected);
      patch({ status: res.status, currentStage: res.current_stage, mitigationMethods: selected });
      startPolling(runId);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the pipeline.");
    } finally {
      setSubmitting(false);
    }
  };

  // Self-healing: if we land on this page believing a run is "running"
  // (e.g. the person navigated to Versions/Compare to peek at partial
  // results, then came back here) but no interval is actually polling --
  // the previous one was torn down on unmount -- re-check the real
  // status once instead of displaying a stale "Running…" forever.
  useEffect(() => {
    if (!runId || status !== "running" || pollTimer.current) return;

    let cancelled = false;
    (async () => {
      try {
        const res = await getRunStatus(runId);
        if (cancelled) return;
        patch({
          status: res.status,
          currentStage: res.current_stage,
          errorMessage: res.error_message,
          failedMethods: res.failed_methods || [],
        });
        if (res.status === "completed") {
          markStageReached("versions");
          navigate("/versions");
        } else if (res.status === "failed") {
          setError(res.error_message || "The pipeline failed. See run status for details.");
        } else if (res.status === "running") {
          startPolling(runId);
        }
      } catch {
        if (!cancelled) setError("Could not confirm the current run status. Try refreshing.");
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  if (!runId || !protectedAttribute) {
    return (
      <div>
        <PageHeader eyebrow="Step 3" title="Run bias mitigation" />
        <Card>Configure the protected attribute first.</Card>
      </div>
    );
  }

  const running = polling || status === "running";

  return (
    <div>
      <PageHeader
        eyebrow="Step 3"
        title="Original model & bias mitigation"
        description={`Auditing "${protectedAttribute}" -- "${privilegedValue}" (privileged) vs "${unprivilegedValue}" (unprivileged). Review the original model below, then select one or more mitigation methods; each produces its own sibling version alongside it.`}
      />

      <ErrorBanner message={error} />

      {loadingOriginal && <Card style={{ marginBottom: "24px" }}>Loading original model…</Card>}
      {original && (
        <div style={{ marginBottom: "24px" }}>
          <VersionAnalysis version={original} />
        </div>
      )}

      <Card>
        <h2 style={{ marginBottom: "16px" }}>Select mitigation methods</h2>
        <div className="mitigate-grid">
          {MITIGATION_METHODS.map((m) => (
            <label key={m.name} className="mitigate-option">
              <input
                type="checkbox"
                checked={selected.includes(m.name)}
                onChange={() => toggle(m.name)}
                disabled={running}
              />
              <div>
                <div className="mitigate-option__label">{m.label}</div>
                <Badge tone={m.category === "pre" ? "accent" : "neutral"}>
                  {m.category === "pre" ? "pre-processing · new model" : "post-processing · no new model"}
                </Badge>
              </div>
            </label>
          ))}
        </div>

        <div style={{ marginTop: "24px", display: "flex", gap: "12px", alignItems: "center" }}>
          <Button onClick={handleRun} disabled={selected.length === 0 || submitting || running}>
            {running ? `Running${currentStage ? ` -- ${currentStage}` : ""}…` : submitting ? "Starting…" : "Run pipeline"}
          </Button>
          {running && (
            <Link to="/versions">
              <Button variant="ghost">View versions completed so far</Button>
            </Link>
          )}
        </div>
      </Card>
    </div>
  );
}
