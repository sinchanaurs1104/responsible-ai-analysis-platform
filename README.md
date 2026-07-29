# Responsible AI Platform

A full-stack platform for training a model, auditing it for bias against a chosen protected attribute, applying one or more [IBM AIF360](https://github.com/Trusted-AI/AIF360) bias-mitigation methods, and comparing every resulting version side-by-side — performance, fairness, SHAP explainability, error analysis, and counterfactual explanations — inspired by Microsoft's Responsible AI Dashboard.

> **Status:** functional end-to-end (upload → train → configure → evaluate → mitigate → compare → export), built and verified against a live backend. Some areas are MVP-scoped and called out explicitly in [Future Enhancements](#future-enhancements) rather than glossed over.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)

---

## Overview

The platform lets you:

1. Upload a CSV dataset and train a classifier (or upload an already-trained model + train/test split).
2. Configure a **protected attribute** (e.g. `sex`, `race`) with a privileged and unprivileged group — including attributes with more than two categories, where the platform automatically scopes the run to just the two groups you select and reports exactly what was filtered.
3. Evaluate the **original model** — performance, fairness metrics, SHAP feature importance, error-analysis cohorts, and DiCE counterfactual explanations — before deciding on mitigation.
4. Select one or more **bias mitigation methods** (pre-processing and post-processing) to run against the original model, each producing its own sibling model version.
5. Review every version's full analysis individually, then compare all versions together in a single dashboard.
6. Download trained model artifacts (where a genuinely new model exists), a generated PDF Responsible AI Report, and a combined experiment-log CSV export.

The backend is the source of truth for all computation — evaluation, fairness metrics, explainability, and mitigation all run there via [AIF360](https://github.com/Trusted-AI/AIF360), [SHAP](https://github.com/shap/shap), and [DiCE](https://github.com/interpretml/DiCE). The frontend is a thin, stage-based UI over that API and performs no bias computation of its own.

---

## Architecture

```
┌──────────────────────┐        REST / multipart-form         ┌───────────────────────────┐
│   React + Vite SPA   │ ────────────────────────────────────▶ │   FastAPI backend          │
│  (workflow-stage UI) │ ◀──────────────────────────────────── │  (all computation happens  │
└──────────────────────┘        JSON responses + file          │   here)                    │
                                 downloads (model/report/CSV)   └───────────────────────────┘
                                                                        │
                                                                        ▼
                                                        ┌───────────────────────────────┐
                                                        │ SQLite (run + version records) │
                                                        │ Local filesystem (model .pkl   │
                                                        │ artifacts, generated PDFs)      │
                                                        └───────────────────────────────┘
```

### Backend pipeline (two-phase execution)

The core pipeline (`app/core/pipeline_orchestrator.py`) is split into two independently callable phases so the original model's results can be reviewed *before* committing to mitigation methods:

- **Phase 1 — `evaluate_original_model()`**: trains/loads the model, then runs performance evaluation, SHAP explainability, error analysis, counterfactual generation, and fairness metrics for the original model only. Persists it as version 1 and sets run status to `original_ready`.
- **Phase 2 — `run_mitigations()`**: given an already-evaluated original model and a list of selected mitigation method names, runs each method (retraining for pre-processing methods, wrapping the existing estimator for post-processing methods), then repeats the full analysis for each, producing one sibling version per method.

A combined `run_pipeline()` wrapper (unchanged signature) composes both phases in one call for non-interactive use.

### Frontend workflow stages

The UI is organized as five sequential stages (see `frontend/src/api/constants.js`), each backed by real API calls with polling for long-running background work:

`Upload & Train → Configure → Mitigate → Versions → Compare`

A shared `RunContext` (React context) carries run state across stages, and a single reusable `VersionAnalysis` component renders the full analysis (performance, fairness, SHAP, error analysis, counterfactuals) for *any* version — the original model or any mitigated version — so adding a new mitigation method on the backend requires no frontend code changes.

---

## Features

### Implemented

- **Two upload paths**: build-and-train from a raw CSV, or upload an existing trained model with its train/test split (`/runs/build` and `/runs/analyze` — the frontend currently wires up the build-and-train path).
- **Arbitrary protected attributes**: any column, any number of categories. If more than two values are present, the platform scopes the run to the two selected groups (filtering + retraining for internally-trained models, filtering-only for uploaded models) and reports the row counts filtered, rather than requiring a strictly binary column upfront.
- **Four supported classifiers**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Four supported mitigation methods**:
  | Method | Category |
  |---|---|
  | Reweighing | pre-processing (trains a new model) |
  | Disparate Impact Remover | pre-processing (trains a new model) |
  | Calibrated Equalized Odds Postprocessing | post-processing (no new model — wraps the original) |
  | Reject Option Classification | post-processing (no new model — wraps the original) |
- **Multi-method runs**: select any subset of the four methods; each produces its own version in the same run.
- **Full per-version analysis**: accuracy/precision/recall/F1 + confusion matrix, AIF360 fairness metrics (statistical parity difference, disparate impact ratio, equal opportunity difference, average odds difference, Theil index), a rule-based fairness verdict with a suggested mitigation (labeled `standard` or `experimental` confidence depending on how directly the literature supports that pairing), SHAP top-feature importances, worst-performing error-analysis subgroups, and DiCE counterfactual examples.
- **Comparison dashboard**: summary table, performance chart, fairness comparison (including a custom visual "scale" indicator), SHAP feature comparison across versions, and a counterfactual-generation coverage/sparsity summary — not a misleading "flip rate" metric, since every stored counterfactual example is by definition a successful flip.
- **Partial-failure visibility**: if one mitigation method fails (e.g. a missing optional dependency) while others succeed, the run still completes and the failure is surfaced in the UI rather than silently vanishing.
- **Downloadable outputs**: trained model artifacts (gated to genuinely new models — hidden for post-processing methods, which only wrap the original estimator), a generated PDF Responsible AI Report covering every version in the run, and a combined experiment-log CSV export.

### Not yet implemented

- Authentication / multi-user access control.
- The "upload an existing trained model" workflow (`/runs/analyze`) is implemented on the backend but not yet wired into the frontend UI.
- In-processing mitigation methods (e.g. Adversarial Debiasing, Exponentiated Gradient Reduction) — the registry structure supports adding them, but they are not currently registered or implemented.
- Cohort/data-analysis views beyond the error-analysis worst-subgroups table.

---

## Installation

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18+
- `pip`, `npm`

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is now available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`. A SQLite database file and an `artifacts/` folder are created automatically on first run.

> **Note:** Disparate Impact Remover depends on AIF360's optional `BlackBoxAuditing` package, which is not part of `requirements.txt` above and can be finicky to install on some platforms. Without it, Disparate Impact Remover will fail at runtime for a given run while every other method still completes normally (see [Known Limitations](#known-limitations)).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app is now available at `http://localhost:5173`. If your backend isn't running on the default `http://localhost:8000`, create `frontend/.env`:

```
VITE_API_BASE_URL=http://your-backend-host:8000
```

---

## Usage Workflow

1. **Upload & Train** — upload a CSV, pick the target column and a classifier, and submit. The dataset's columns are parsed client-side to populate the dropdowns; training happens on the backend via `POST /runs/build`.
2. **Configure** — pick the protected attribute and its privileged/unprivileged values. Submitting calls `POST /runs/{id}/configure` (which performs any needed group-scoping) followed by `POST /runs/{id}/evaluate`, then polls `GET /runs/{id}/status` until the original model's evaluation completes.
3. **Mitigate** — review the original model's full analysis (rendered inline), then select one or more mitigation methods and submit. This calls `POST /runs/{id}/execute` and polls status until all selected methods finish (or fail individually without aborting the run).
4. **Versions** — one tab per version (original + each mitigated method), each rendering the full analysis via the shared `VersionAnalysis` component.
5. **Compare** — a single dashboard summarizing every version: summary table, performance chart, fairness comparison, SHAP comparison, counterfactual summary, and links to download the PDF report / CSV export / per-version model artifacts.

---

## Project Structure

```
responsible_ai/
├── backend/
│   ├── main.py                       # FastAPI app entry point, route registration, CORS
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/routes/                # runs.py, versions.py, experiments.py
│   │   ├── core/                      # pipeline_orchestrator.py, exceptions, context_store
│   │   ├── db/                        # models.py, repository.py, session.py
│   │   ├── modules/
│   │   │   ├── ingestion/             # dataset/model loading & validation
│   │   │   ├── training/              # classifier training
│   │   │   ├── evaluation/            # performance metrics
│   │   │   ├── fairness/              # AIF360 metrics, thresholds, insight engine, dataset_utils
│   │   │   ├── explainability/        # SHAP, error analysis, DiCE counterfactuals
│   │   │   ├── mitigation/            # preprocessing/ + postprocessing/ strategies + registry
│   │   │   ├── retraining/            # retrain-with-preprocessing-result
│   │   │   ├── versioning/            # version_manager, model_card
│   │   │   ├── reporting/             # PDF report_builder
│   │   │   ├── narrative/             # narrative_generator (+ LLM/fallback template)
│   │   │   ├── storage/               # model_storage, report_storage
│   │   │   └── experiments/           # experiment_log CSV export
│   │   └── schemas/                    # Pydantic schemas (context, metrics, fairness)
│   └── tests/                          # script-style test modules (run directly, not pytest-discovered)
│
└── frontend/
    ├── src/
    │   ├── api/                        # client.js, constants.js, RunContext.jsx
    │   ├── components/
    │   │   ├── version-analysis/       # VersionAnalysis + Performance/Fairness/Explainability/
    │   │   │                           # ErrorAnalysis/Counterfactual sub-sections
    │   │   ├── compare/                # SummaryTable, Performance/Fairness/Shap/Counterfactual
    │   │   │                           # comparison components
    │   │   ├── Sidebar.jsx, AppShell.jsx, ScaleGlyph.jsx, ui.jsx
    │   ├── pages/                      # UploadPage, ConfigurePage, MitigatePage, VersionsPage,
    │   │                               # ComparePage
    │   └── styles/                     # tokens.css (design tokens)
    └── package.json
```

---

## Technologies Used

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [SQLAlchemy](https://www.sqlalchemy.org/) + SQLite — run/version persistence
- [scikit-learn](https://scikit-learn.org/) — classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- [AIF360](https://github.com/Trusted-AI/AIF360) — fairness metrics and bias mitigation algorithms
- [SHAP](https://github.com/shap/shap) — feature importance / explainability
- [DiCE](https://github.com/interpretml/DiCE) — counterfactual explanations
- [ReportLab](https://www.reportlab.com/) — PDF report generation
- [Pydantic](https://docs.pydantic.dev/) — schema validation

**Frontend**
- [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- [React Router](https://reactrouter.com/) — stage-based routing
- [Recharts](https://recharts.org/) — performance/fairness comparison charts
- [PapaParse](https://www.papaparse.com/) — client-side CSV column/value parsing (for populating dropdowns; the backend performs the actual filtering/computation)

---

## Known Limitations

- **Disparate Impact Remover** depends on AIF360's optional `BlackBoxAuditing` dependency, which is not always straightforward to install. If unavailable, this one method fails per run while the rest complete normally, and the failure is surfaced in the UI (`failed_methods` on run status).
- **In-memory run context**: the live model/dataset context used between `/configure`, `/evaluate`, and `/execute` is held in server memory, not persisted — a backend restart mid-run requires re-uploading.
- **SQLite** is used for simplicity; concurrent write load at scale is not a design target of the current implementation.
- **No authentication** — the API and UI assume a single trusted user/environment.
- The `mitigation_confidence` field on fairness findings (`standard` vs `experimental`) reflects how directly the underlying research supports a given driving-factor → method pairing; it is not a claim that the "experimental" pairing is invalid, only that it's a reasonable inference rather than a directly validated one.

---

## Future Enhancements

- Wire the existing "analyze an uploaded model" backend workflow into the frontend.
- Add in-processing mitigation methods (e.g. Adversarial Debiasing, Exponentiated Gradient Reduction) via the existing mitigation registry structure.
- Persist the in-memory run context (e.g. to disk or a job queue) so long-running or interrupted runs survive a backend restart.
- Add authentication and per-user run isolation for multi-user deployments.
- Expand cohort/data-analysis views beyond the current worst-subgroups table (e.g. full interactive cohort explorer, data distribution views).
- Convert the backend's script-style tests into a proper pytest suite with fixtures and CI integration.
- Code-split the frontend bundle (currently a single ~650 KB chunk) for faster initial load.
