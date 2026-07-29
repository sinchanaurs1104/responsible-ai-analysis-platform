import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import { buildRun, ApiError } from "../api/client";
import { ALGORITHMS } from "../api/constants";
import { useRun } from "../api/RunContext";
import { Card, PageHeader, Button, Field, WarningBanner, ErrorBanner } from "../components/ui";

export default function UploadPage() {
  const navigate = useNavigate();
  const { patch, markStageReached } = useRun();

  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [rowCount, setRowCount] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [algorithmName, setAlgorithmName] = useState(ALGORITHMS[0].name);
  const [parsing, setParsing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);

  const handleFile = useCallback((selected) => {
    setFile(selected);
    setColumns([]);
    setTargetColumn("");
    setError(null);
    if (!selected) return;

    setParsing(true);
    Papa.parse(selected, {
      header: true,
      preview: 500,
      skipEmptyLines: true,
      complete: (results) => {
        setColumns(results.meta.fields || []);
        setRowCount(results.data.length);
        setParsing(false);
      },
      error: (err) => {
        setError(`Could not read this CSV: ${err.message}`);
        setParsing(false);
      },
    });
  }, []);

  const canSubmit = file && targetColumn && algorithmName && !submitting;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await buildRun({ file, targetColumn, algorithmName });
      setWarnings(res.validation_warnings || []);
      patch({
        runId: res.run_id,
        workflowType: res.workflow_type,
        datasetName: file.name,
        algorithmName: res.algorithm_name,
        targetColumn,
        validationWarnings: res.validation_warnings || [],
        status: res.status,
        // Column list is kept for the Configure page's protected-attribute picker.
        _csvColumns: columns,
        _csvFile: file,
      });
      markStageReached("configure");
      navigate("/configure");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed. Is the backend running?");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Step 1"
        title="Upload a dataset & train the original model"
        description="Upload a CSV, pick the column you're predicting, and choose a classifier. The platform trains the original model immediately -- fairness configuration comes next."
      />

      <ErrorBanner message={error} />
      <WarningBanner warnings={warnings} />

      <Card as="form" onSubmit={handleSubmit}>
        <Field label="Dataset (CSV)" hint="Parsed locally first so you can pick the target column below.">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => handleFile(e.target.files?.[0] || null)}
          />
        </Field>

        {parsing && <p>Reading columns…</p>}

        {columns.length > 0 && (
          <>
            <Field label="Target column" hint={rowCount ? `${columns.length} columns detected` : undefined}>
              <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                <option value="" disabled>
                  Select the column to predict…
                </option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Classifier">
              <select value={algorithmName} onChange={(e) => setAlgorithmName(e.target.value)}>
                {ALGORITHMS.map((a) => (
                  <option key={a.name} value={a.name}>
                    {a.label}
                  </option>
                ))}
              </select>
            </Field>
          </>
        )}

        <Button type="submit" disabled={!canSubmit}>
          {submitting ? "Uploading & training…" : "Upload & train"}
        </Button>
      </Card>
    </div>
  );
}
