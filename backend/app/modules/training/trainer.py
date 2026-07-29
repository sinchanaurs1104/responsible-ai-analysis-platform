"""
Entry point for Workflow A (Build & Analyze Model).

Orchestrates: target/algorithm validation -> high-cardinality column
check -> stratified train/test split -> auto-built preprocessing ->
fit -> TrainedModelContext.

This is the only function the API layer should call for this workflow.
Downstream of this function, a Workflow-A-produced context is
indistinguishable in shape from a Workflow-B-produced one (SDD §7, §12).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.exceptions import ConfigValidationError, DatasetValidationError
from app.core.supported_models import is_supported, build_estimator, SUPPORTED_MODELS
from app.schemas.context import TrainedModelContext
from app.modules.training.preprocessing_builder import (
    build_preprocessor,
    detect_high_cardinality_columns,
)

MIN_ROWS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_new_model(
    dataset_df: pd.DataFrame,
    target_column: str,
    algorithm_name: str,
    excluded_columns: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> TrainedModelContext:
    excluded_columns = excluded_columns or []

    # --- Config validation -------------------------------------------------
    if target_column not in dataset_df.columns:
        raise ConfigValidationError(
            f"Target column '{target_column}' not found in dataset.",
            details={"available_columns": dataset_df.columns.tolist()},
        )

    if not is_supported(algorithm_name):
        raise ConfigValidationError(
            f"Algorithm '{algorithm_name}' is not supported in this "
            f"version of the platform.",
            details={"supported_algorithms": list(SUPPORTED_MODELS.keys())},
        )

    if len(dataset_df) < MIN_ROWS:
        raise DatasetValidationError(
            f"Dataset has only {len(dataset_df)} rows; at least "
            f"{MIN_ROWS} are required to train and split meaningfully.",
            details={"row_count": len(dataset_df), "minimum_required": MIN_ROWS},
        )

    y = dataset_df[target_column]
    n_classes = y.nunique(dropna=True)
    if n_classes < 2:
        raise DatasetValidationError(
            "Target column has only one class present; a classifier "
            "cannot be trained on a single-class target.",
            details={"target_column": target_column, "unique_values": y.unique().tolist()},
        )
    if n_classes > 2:
        raise ConfigValidationError(
            "Target column has more than two classes. This platform's "
            "MVP supports binary classification only, since the fairness "
            "metrics used (Statistical Parity Difference, Disparate "
            "Impact, etc.) are defined for binary outcomes.",
            details={"target_column": target_column, "class_count": int(n_classes)},
        )

    feature_df = dataset_df.drop(columns=[target_column])
    if excluded_columns:
        missing = [c for c in excluded_columns if c not in feature_df.columns]
        if missing:
            raise ConfigValidationError(
                "Some columns marked for exclusion do not exist in the dataset.",
                details={"missing_columns": missing},
            )
        feature_df = feature_df.drop(columns=excluded_columns)

    high_cardinality_cols = detect_high_cardinality_columns(feature_df)
    if high_cardinality_cols:
        raise ConfigValidationError(
            "Some categorical columns have too many unique values to "
            "encode meaningfully (likely identifier columns). Exclude "
            "them and retry.",
            details={"high_cardinality_columns": high_cardinality_cols},
        )

    # --- Split ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )

    # --- Build + fit Pipeline -------------------------------------------
    preprocessor = build_preprocessor(feature_df)
    estimator = build_estimator(algorithm_name, random_state=random_state)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)

    train_df = X_train.copy()
    train_df[target_column] = y_train
    test_df = X_test.copy()
    test_df[target_column] = y_test

    return TrainedModelContext(
        pipeline=pipeline,
        source="internally_trained",
        preprocessing_status="pipeline_managed",
        estimator_step_name="model",
        algorithm_name=algorithm_name,
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        target_column=target_column,
        validation_warnings=[],
    )
