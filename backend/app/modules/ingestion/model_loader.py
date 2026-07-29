"""
Loads an uploaded model file and normalizes it into a fitted
sklearn.Pipeline, regardless of whether the user uploaded a Pipeline
directly or a bare fitted estimator (SDD §7, §"reconsidering Pipeline
requirement" decision).

This is intentionally the ONLY place in the codebase that knows the
difference between "uploaded a Pipeline" and "uploaded a bare estimator" --
everything after model_loader.load() deals exclusively with a Pipeline.
"""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from app.core.exceptions import ModelValidationError

# A conservative allowlist of pickle "magic" isn't enforceable at the byte
# level for joblib files, so defense here is: only ever load from a path
# we control (already-saved-to-disk upload), never eval untrusted bytes
# directly, and catch broadly so a malformed/hostile file cannot crash
# the whole request. This is a pragmatic MVP safeguard, not a sandbox --
# see SDD non-functional requirements re: pickle deserialization risk.


def load_model_file(file_path: str | Path) -> BaseEstimator:
    """Deserialize an uploaded model file. Raises ModelValidationError on
    any failure instead of letting a raw exception escape ingestion."""
    path = Path(file_path)

    if not path.exists():
        raise ModelValidationError(
            "Uploaded model file could not be found on disk.",
            details={"path": str(path)},
        )

    if path.suffix not in {".pkl", ".joblib"}:
        raise ModelValidationError(
            "Unsupported model file extension. Expected .pkl or .joblib.",
            details={"extension": path.suffix},
        )

    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see module docstring
        raise ModelValidationError(
            "Uploaded file could not be deserialized as a model. "
            "It may be corrupted or not a valid scikit-learn object.",
            details={"underlying_error": str(exc)},
        ) from exc

    if not isinstance(model, BaseEstimator):
        raise ModelValidationError(
            "Uploaded file did not deserialize into a scikit-learn "
            "estimator or Pipeline.",
            details={"loaded_type": type(model).__name__},
        )

    return model


def is_pipeline(model: BaseEstimator) -> bool:
    return isinstance(model, Pipeline)


def wrap_if_needed(model: BaseEstimator) -> tuple[Pipeline, str, str]:
    """
    Ensures the returned object is always a Pipeline.

    Returns:
        (pipeline, estimator_step_name, preprocessing_status)
    """
    if is_pipeline(model):
        # Native Pipeline: use directly. The final step is the estimator.
        final_step_name = model.steps[-1][0]
        preprocessing_status = "pipeline_managed"
        return model, final_step_name, preprocessing_status

    # Bare estimator: wrap into a single-step Pipeline so every downstream
    # module (retraining, SHAP, serialization) can treat all models
    # uniformly as Pipelines (SDD decision: "Uploaded fitted estimators
    # are automatically wrapped into an internal Pipeline after validation").
    wrapped = Pipeline([("model", model)])
    return wrapped, "model", "user_responsibility"


def get_final_estimator(pipeline: Pipeline) -> BaseEstimator:
    return pipeline.steps[-1][1]
