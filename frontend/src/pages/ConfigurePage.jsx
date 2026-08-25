import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import Papa from "papaparse";
import { configureRun, evaluateOriginal, getRunStatus, ApiError } from "../api/client";
import { useRun } from "../api/RunContext";
import { Card, PageHeader, Button, Field, WarningBanner, ErrorBanner } from "../components/ui";

const POLL_INTERVAL_MS = 2000;

export default function ConfigurePage() {
  const navigate = useNavigate();
  const {
    runId,
    targetColumn,
    _csvColumns,
    _csvFile,
    protectedAttribute: savedProtectedAttribute,
    privilegedValue: savedPrivilegedValue,
    unprivilegedValue: savedUnprivilegedValue,
    validationWarnings,
    status,
    errorMessage,
    patch,
    markStageReached,
    reset,
  } = useRun();

  const attributeOptions = useMemo(
    () => (_csvColumns || []).filter((c) => c !== targetColumn),
    [_csvColumns, targetColumn]
  );

  const [protectedAttribute, setProtectedAttribute] = useState("");
  const [uniqueValues, setUniqueValues] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [privilegedValue, setPrivilegedValue] = useState("");
  const [unprivilegedValue, setUnprivilegedValue] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [evaluateStage, setEvaluateStage] = useState(null);
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const pollTimer = useRef(null);

  useEffect(() => () => {
    if (pollTimer.current) clearInterval(pollTimer.current);
  }, []);

  useEffect(() => {
    if (status === "failed" && errorMessage && !error) {
      setError(errorMessage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleAttributeChange = useCallback(
    (attr) => {
      setProtectedAttribute(attr);
      setUniqueValues([]);
      setPrivilegedValue("");
      setUnprivilegedValue("");
      if (!attr || !_csvFile) return;

      setScanning(true);
      const seen = new Set();
      Papa.parse(_csvFile, {
        header: true,
        skipEmptyLines: true,
        step: (row) => {
          const v = row.data[attr];
          if (v !== undefined && v !== null && v !== "") seen.add(String(v));
        },
        complete: () => {
          setUniqueValues(Array.from(seen).sort());
          setScanning(false);
        },
        error: () => setScanning(false),
      });
    },
    [_csvFile]
  );

  const canSubmit =
    protectedAttribute &&
    privilegedValue &&
    unprivilegedValue &&
    privilegedValue !== unprivilegedValue &&
    !submitting &&
    !evaluating;

  const pollForOriginalReady = useCallback(
    (id) => {
      setEvaluating(true);
      pollTimer.current = setInterval(async () => {
        try {
          const res = await getRunStatus(id);
          setEvaluateStage(res.current_stage);
          patch({ status: res.status, currentStage: res.current_stage, errorMessage: res.error_message });

          if (res.status === "original_ready") {
            clearInterval(pollTimer.current);
            setEvaluating(false);
            markStageReached("mitigate");
            navigate("/mitigate");
          } else if (res.status === "failed") {
            clearInterval(pollTimer.current);
            setEvaluating(false);
            setError(res.error_message || "Evaluating the original model failed.");
          }
        } catch (err) {
          clearInterval(pollTimer.current);
          setEvaluating(false);
          setError(err instanceof ApiError ? err.message : "Lost connection while evaluating the original model.");
        }
      }, POLL_INTERVAL_MS);
    },
    [patch, markStageReached, navigate]
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const configRes = await configureRun(runId, { protectedAttribute, privilegedValue, unprivilegedValue });
      setWarnings(configRes.validation_warnings || []);
      patch({
        protectedAttribute,
        privilegedValue,
        unprivilegedValue,
        validationWarnings: configRes.validation_warnings || [],
      });

      const evalRes = await evaluateOriginal(runId);
      patch({ status: evalRes.status, currentStage: evalRes.current_stage });
      setSubmitting(false);
      pollForOriginalReady(runId);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Configuration failed.");
      setSubmitting(false);
    }
  };

  const handleStartNew = () => {
    reset();
    navigate("/upload");
  };

  if (!runId) {
    return (
      <div>
        <PageHeader eyebrow="Step 2" title="Configure the protected attribute" />
        <Card>Start by uploading a dataset first.</Card>
      </div>
    );
  }

  // Already configured & evaluated for this run -- show what was chosen
  // instead of blank dropdowns. Resubmitting here would re-run /evaluate
  // and create a duplicate "original model" version server-side, so
  // changing this decision requires starting a new run rather than
  // editing in place.
  if (savedProtectedAttribute && status !== "failed" && !submitting && !evaluating) {
    return (
      <div>
        <PageHeader eyebrow="Step 2" title="Configure the protected attribute" />
        <WarningBanner warnings={validationWarnings} />
        <Card>
          <div className="run-summary">
            <div className="run-summary__row">
              <span className="run-summary__label">Protected attribute</span>
              <span className="run-summary__value">{savedProtectedAttribute}</span>
            </div>
            <div className="run-summary__row">
              <span className="run-summary__label">Privileged group</span>
              <span className="run-summary__value">{savedPrivilegedValue}</span>
            </div>
            <div className="run-summary__row">
              <span className="run-summary__label">Unprivileged group</span>
              <span className="run-summary__value">{savedUnprivilegedValue}</span>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", marginTop: "24px" }}>
            <Link to="/mitigate">
              <Button>Continue to Mitigate</Button>
            </Link>
            <Button variant="secondary" onClick={handleStartNew}>
              Start a new run
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  const busy = submitting || evaluating;

  return (
    <div>
      <PageHeader
        eyebrow="Step 2"
        title="Configure the protected attribute"
        description="Pick the column to audit for bias, then choose which value is the privileged group and which is unprivileged. The original model is evaluated as soon as you save -- you'll see its full analysis before picking mitigation methods."
      />

      <ErrorBanner message={error} />
      <WarningBanner warnings={warnings} />

      <Card as="form" onSubmit={handleSubmit}>
        <Field label="Protected attribute">
          <select
            value={protectedAttribute}
            onChange={(e) => handleAttributeChange(e.target.value)}
            disabled={busy}
          >
            <option value="" disabled>
              Select a column…
            </option>
            {attributeOptions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Field>

        {scanning && <p>Scanning values…</p>}

        {uniqueValues.length > 0 && (
          <>
            {uniqueValues.length > 2 && (
              <p className="field__hint" style={{ marginTop: "-8px", marginBottom: "16px" }}>
                {uniqueValues.length} distinct values found -- rows outside your two chosen
                values will be scoped out of this run.
              </p>
            )}

            <Field label="Privileged group">
              <select
                value={privilegedValue}
                onChange={(e) => setPrivilegedValue(e.target.value)}
                disabled={busy}
              >
                <option value="" disabled>
                  Select a value…
                </option>
                {uniqueValues.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Unprivileged group">
              <select
                value={unprivilegedValue}
                onChange={(e) => setUnprivilegedValue(e.target.value)}
                disabled={busy}
              >
                <option value="" disabled>
                  Select a value…
                </option>
                {uniqueValues
                  .filter((v) => v !== privilegedValue)
                  .map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
              </select>
            </Field>
          </>
        )}

        <Button type="submit" disabled={!canSubmit}>
          {submitting
            ? "Saving…"
            : evaluating
            ? `Evaluating original model${evaluateStage ? ` -- ${evaluateStage}` : ""}…`
            : "Save & evaluate original model"}
        </Button>
      </Card>
    </div>
  );
}
