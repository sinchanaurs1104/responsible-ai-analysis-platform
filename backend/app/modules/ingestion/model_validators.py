"""
Validation checks applied during ingestion (Analyze Existing Model
workflow). Per SDD §15, all validation lives here at the ingestion
checkpoint -- evaluation/fairness/explainability/mitigation modules can
assume they are always given clean, compatible, fitted inputs.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, NotFittedError

from app.core.exceptions import ModelValidationError, DatasetValidationError
from app.core.supported_models import is_supported, SUPPORTED_MODELS


def validate_is_fitted(model: BaseEstimator) -> None:
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise ModelValidationError(
            "Uploaded model does not appear to be fitted. "
            "Train it before uploading, or use the 'Build & Analyze Model' "
            "workflow to train a model on this platform."
        ) from exc
    except TypeError as exc:
        # check_is_fitted raises TypeError for estimators it doesn't
        # recognize as a fitted-checkable type (e.g. some Pipeline edge
        # cases) -- treat as inconclusive rather than failing hard.
        raise ModelValidationError(
            "Could not determine whether the uploaded model is fitted.",
            details={"underlying_error": str(exc)},
        ) from exc


def validate_algorithm_supported(pipeline: Pipeline) -> str:
    """Returns the algorithm name if supported, else raises."""
    final_estimator = pipeline.steps[-1][1]
    algorithm_name = type(final_estimator).__name__

    if not is_supported(algorithm_name):
        raise ModelValidationError(
            f"Algorithm '{algorithm_name}' is not supported in this "
            f"version of the platform.",
            details={"supported_algorithms": list(SUPPORTED_MODELS.keys())},
        )
    return algorithm_name


def validate_preprocessing_assumptions(
    df: pd.DataFrame,
    pipeline: Pipeline,
    preprocessing_status: str,
) -> list[str]:
    """
    Heuristic sanity checks for the 'user_responsibility' case (bare
    estimator wrapped into a single-step Pipeline). These cannot prove
    correctness -- they catch the common, near-certain failure modes
    and hard-reject on those; anything more subtle (e.g. was scaling
    applied) becomes a disclosed warning instead, per the finalized
    "reconsidering Pipeline requirement" decision.

    Returns a list of non-fatal warning strings. Raises
    DatasetValidationError for the hard-reject cases.
    """
    warnings: list[str] = []

    if preprocessing_status != "user_responsibility":
        # Native Pipeline manages its own preprocessing -- nothing to check.
        return warnings

    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        raise DatasetValidationError(
            "Dataset contains non-numeric (text/categorical) columns, "
            "but the uploaded model is a bare estimator that expects "
            "pre-encoded numeric input.",
            details={"non_numeric_columns": object_cols},
        )

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise DatasetValidationError(
            "Dataset contains missing values, but the uploaded model is "
            "a bare estimator that expects fully imputed input.",
            details={"columns_with_missing_values": null_cols},
        )

    final_estimator = pipeline.steps[-1][1]
    expected_n = getattr(final_estimator, "n_features_in_", None)
    if expected_n is not None and df.shape[1] != expected_n:
        raise DatasetValidationError(
            f"Dataset has {df.shape[1]} columns but the model expects "
            f"{expected_n} input features.",
            details={"dataset_columns": df.shape[1], "expected_features": expected_n},
        )

    # Nothing hard-fails past this point -- but this is exactly the case
    # the user asked us to disclose, not silently trust.
    warnings.append(
        "Uploaded model is a bare estimator (not a Pipeline). "
        "Preprocessing of the uploaded datasets is assumed to exactly "
        "match how the model was originally trained. This cannot be "
        "verified automatically."
    )
    return warnings


def validate_dataset_not_empty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise DatasetValidationError(f"{name} dataset is empty.")


def validate_schema_match(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    if train_cols != test_cols:
        raise DatasetValidationError(
            "Training and testing datasets have different columns.",
            details={
                "only_in_train": sorted(train_cols - test_cols),
                "only_in_test": sorted(test_cols - train_cols),
            },
        )
