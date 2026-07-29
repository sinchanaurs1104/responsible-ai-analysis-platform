const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(status, body) {
    super(body?.message || `Request failed (${status})`);
    this.status = status;
    this.errorType = body?.error_type;
    this.details = body?.details;
  }
}

async function handle(res) {
  if (!res.ok) {
    let body = null;
    try {
      body = await res.json();
    } catch {
      // non-JSON error body, leave body null
    }
    throw new ApiError(res.status, body);
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return res.json();
  return res;
}

// ---- Runs ----

export function buildRun({ file, targetColumn, algorithmName }) {
  const form = new FormData();
  form.append("dataset", file);
  form.append("target_column", targetColumn);
  form.append("algorithm_name", algorithmName);
  return fetch(`${BASE_URL}/runs/build`, { method: "POST", body: form }).then(handle);
}

export function analyzeExistingRun({ modelFile, trainFile, testFile, targetColumn }) {
  const form = new FormData();
  form.append("model", modelFile);
  form.append("train_dataset", trainFile);
  form.append("test_dataset", testFile);
  form.append("target_column", targetColumn);
  return fetch(`${BASE_URL}/runs/analyze`, { method: "POST", body: form }).then(handle);
}

export function configureRun(runId, { protectedAttribute, privilegedValue, unprivilegedValue }) {
  return fetch(`${BASE_URL}/runs/${runId}/configure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      protected_attribute: protectedAttribute,
      privileged_value: privilegedValue,
      unprivileged_value: unprivilegedValue,
    }),
  }).then(handle);
}

export function evaluateOriginal(runId) {
  return fetch(`${BASE_URL}/runs/${runId}/evaluate`, { method: "POST" }).then(handle);
}

export function executeRun(runId, mitigationMethods) {
  const form = new FormData();
  (mitigationMethods || []).forEach((m) => form.append("mitigation_methods", m));
  return fetch(`${BASE_URL}/runs/${runId}/execute`, { method: "POST", body: form }).then(handle);
}

export function getRunStatus(runId) {
  return fetch(`${BASE_URL}/runs/${runId}/status`).then(handle);
}

export function listVersions(runId) {
  return fetch(`${BASE_URL}/runs/${runId}/versions`).then(handle);
}

export function compareVersions(runId) {
  return fetch(`${BASE_URL}/runs/${runId}/comparison`).then(handle);
}

export function getReportUrl(runId) {
  return `${BASE_URL}/runs/${runId}/report`;
}

// ---- Versions ----

export function getDownloadUrl(versionId) {
  return `${BASE_URL}/versions/${versionId}/download`;
}

export function getModelCard(versionId) {
  return fetch(`${BASE_URL}/versions/${versionId}/model-card`).then(handle);
}

// ---- Experiments ----

export function getExperimentExportUrl() {
  return `${BASE_URL}/experiments/export`;
}

export { ApiError, BASE_URL };
