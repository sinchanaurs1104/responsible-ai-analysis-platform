"""End-to-end sanity checks for the training module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.exceptions import ConfigValidationError, DatasetValidationError
from app.modules.training.trainer import train_new_model


def make_mixed_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "gender": rng.choice(["male", "female"], size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
    })
    df["approved"] = ((df["income"] > 50000) & (df["age"] > 25)).astype(int)
    return df


def case_success_random_forest():
    print("\n[Case 1] Build & Analyze -- RandomForest on mixed numeric/categorical data")
    df = make_mixed_dataset()
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    assert ctx.source == "internally_trained"
    assert ctx.preprocessing_status == "pipeline_managed"
    assert ctx.algorithm_name == "RandomForestClassifier"
    assert ctx.estimator_step_name == "model"
    assert len(ctx.train_df) + len(ctx.test_df) == len(df)
    # Prove the fitted pipeline actually predicts on raw (unencoded) test rows.
    preds = ctx.pipeline.predict(ctx.test_df.drop(columns=["approved"]))
    assert len(preds) == len(ctx.test_df)
    print(f"  PASS: train={len(ctx.train_df)} test={len(ctx.test_df)} | predictions shape ok")


def case_success_logistic_regression():
    print("\n[Case 2] Build & Analyze -- LogisticRegression, same dataset")
    df = make_mixed_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="LogisticRegression")
    assert ctx.algorithm_name == "LogisticRegression"
    preds = ctx.pipeline.predict(ctx.test_df.drop(columns=["approved"]))
    assert len(preds) == len(ctx.test_df)
    print("  PASS: LogisticRegression pipeline trained and predicts correctly")


def case_missing_target_column():
    print("\n[Case 3] Missing target column -- should raise ConfigValidationError")
    df = make_mixed_dataset(seed=2)
    try:
        train_new_model(df, target_column="does_not_exist", algorithm_name="RandomForestClassifier")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message)


def case_unsupported_algorithm():
    print("\n[Case 4] Unsupported algorithm -- should raise ConfigValidationError")
    df = make_mixed_dataset(seed=3)
    try:
        train_new_model(df, target_column="approved", algorithm_name="SVC")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message)


def case_single_class_target():
    print("\n[Case 5] Single-class target -- should raise DatasetValidationError")
    df = make_mixed_dataset(seed=4)
    df["approved"] = 1
    try:
        train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        raise AssertionError("Expected DatasetValidationError")
    except DatasetValidationError as e:
        print("  PASS:", e.message)


def case_multiclass_target():
    print("\n[Case 6] Multi-class target -- should raise ConfigValidationError (binary-only MVP)")
    df = make_mixed_dataset(seed=5)
    df["approved"] = np.random.default_rng(5).integers(0, 3, size=len(df))
    try:
        train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message)


def case_too_few_rows():
    print("\n[Case 7] Too few rows -- should raise DatasetValidationError")
    df = make_mixed_dataset(n=20, seed=6)
    try:
        train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        raise AssertionError("Expected DatasetValidationError")
    except DatasetValidationError as e:
        print("  PASS:", e.message)


def case_high_cardinality_column():
    print("\n[Case 8] High-cardinality ID-like column -- should raise ConfigValidationError")
    df = make_mixed_dataset(seed=7)
    df["customer_id"] = [f"CUST-{i}" for i in range(len(df))]  # unique per row
    try:
        train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message, "|", e.details)


def case_high_cardinality_column_excluded():
    print("\n[Case 9] High-cardinality column explicitly excluded -- should succeed")
    df = make_mixed_dataset(seed=8)
    df["customer_id"] = [f"CUST-{i}" for i in range(len(df))]
    ctx = train_new_model(
        df,
        target_column="approved",
        algorithm_name="RandomForestClassifier",
        excluded_columns=["customer_id"],
    )
    assert "customer_id" not in ctx.train_df.columns
    print("  PASS: trained successfully after excluding customer_id")


if __name__ == "__main__":
    case_success_random_forest()
    case_success_logistic_regression()
    case_missing_target_column()
    case_unsupported_algorithm()
    case_single_class_target()
    case_multiclass_target()
    case_too_few_rows()
    case_high_cardinality_column()
    case_high_cardinality_column_excluded()
    print("\nAll training tests passed.")
