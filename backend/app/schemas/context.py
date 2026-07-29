"""
TrainedModelContext is the convergence boundary described in the SDD
(§7, §12): both the "Build & Analyze Model" and "Analyze Existing Model"
workflows must produce one of these before any analysis module runs.

Everything downstream (evaluation, explainability, fairness, mitigation,
retraining, versioning) depends only on this object's shape — never on
which workflow produced it.
"""

from typing import Literal, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict


class TrainedModelContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: Any
    """Always a fitted sklearn.Pipeline by the time this object exists,
    even if the user uploaded a bare estimator (see model_loader.wrap_if_needed)."""

    source: Literal["uploaded", "internally_trained"]

    preprocessing_status: Literal["pipeline_managed", "user_responsibility"]
    """'pipeline_managed' -> the uploaded/trained Pipeline handles its own
    preprocessing end to end (native Pipeline with >1 step, or internally
    trained). 'user_responsibility' -> a bare estimator was wrapped; the
    uploaded datasets are assumed to already be preprocessed identically
    to how the model was originally trained. This flag is read-only by
    downstream modules for disclosure (report, model card) -- it must
    never be branched on for computation logic."""

    estimator_step_name: str
    """Name of the final ('model') step inside the Pipeline. Needed by
    retraining later to target e.g. f"{estimator_step_name}__sample_weight"
    when refitting with Reweighing-derived weights."""

    algorithm_name: str
    """Class name of the final estimator, e.g. 'RandomForestClassifier'.
    Must be a key in core.supported_models.SUPPORTED_MODELS."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str

    validation_warnings: list[str] = []
    """Non-fatal warnings surfaced during ingestion (e.g. preprocessing
    sanity-check flags for bare estimators). Shown to the user, not acted on."""
