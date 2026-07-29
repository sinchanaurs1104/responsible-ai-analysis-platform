"""
End-to-end sanity checks for the ingestion module.

Not a full pytest suite yet (that comes with the rest of the modules) --
this deliberately exercises the real code paths against real files on
disk, since ingestion's whole job is file I/O + validation, and mocking
that away would test nothing meaningful.
"""

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.exceptions import ModelValidationError, DatasetValidationError
from app.modules.ingestion.upload_handler import handle_existing_model_upload


def make_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    return df


def write_csv(df, tmpdir, name):
    path = Path(tmpdir) / name
    df.to_csv(path, index=False)
    return str(path)


def case_native_pipeline(tmpdir):
    print("\n[Case 1] Native sklearn.Pipeline (RandomForest) -- should succeed cleanly")
    df = make_dataset()
    X, y = df.drop(columns=["target"]), df["target"]

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=20, random_state=42)),
    ])
    pipeline.fit(X, y)

    model_path = str(Path(tmpdir) / "pipeline_model.pkl")
    joblib.dump(pipeline, model_path)

    train_path = write_csv(df.iloc[:150], tmpdir, "train.csv")
    test_path = write_csv(df.iloc[150:], tmpdir, "test.csv")

    ctx = handle_existing_model_upload(model_path, train_path, test_path, "target")
    assert ctx.source == "uploaded"
    assert ctx.preprocessing_status == "pipeline_managed"
    assert ctx.algorithm_name == "RandomForestClassifier"
    assert ctx.estimator_step_name == "model"
    assert ctx.validation_warnings == []
    print("  PASS:", ctx.algorithm_name, "|", ctx.preprocessing_status, "| warnings:", ctx.validation_warnings)


def case_bare_estimator(tmpdir):
    print("\n[Case 2] Bare fitted estimator (LogisticRegression, pre-encoded data) -- should succeed with a warning")
    df = make_dataset(seed=1)
    X, y = df.drop(columns=["target"]), df["target"]

    model = LogisticRegression()
    model.fit(X, y)

    model_path = str(Path(tmpdir) / "bare_model.pkl")
    joblib.dump(model, model_path)

    train_path = write_csv(df.iloc[:150], tmpdir, "train2.csv")
    test_path = write_csv(df.iloc[150:], tmpdir, "test2.csv")

    ctx = handle_existing_model_upload(model_path, train_path, test_path, "target")
    assert ctx.preprocessing_status == "user_responsibility"
    assert isinstance(ctx.pipeline, Pipeline), "bare estimator must be wrapped into a Pipeline"
    assert ctx.estimator_step_name == "model"
    assert len(ctx.validation_warnings) == 1
    print("  PASS: wrapped into Pipeline |", ctx.preprocessing_status, "| warning:", ctx.validation_warnings[0])


def case_unfitted_model(tmpdir):
    print("\n[Case 3] Unfitted model -- should raise ModelValidationError")
    model = RandomForestClassifier()  # never .fit()
    model_path = str(Path(tmpdir) / "unfitted_model.pkl")
    joblib.dump(model, model_path)

    df = make_dataset(seed=2)
    train_path = write_csv(df.iloc[:150], tmpdir, "train3.csv")
    test_path = write_csv(df.iloc[150:], tmpdir, "test3.csv")

    try:
        handle_existing_model_upload(model_path, train_path, test_path, "target")
        raise AssertionError("Expected ModelValidationError, but none was raised")
    except ModelValidationError as e:
        print("  PASS: correctly rejected ->", e.message)


def case_unsupported_algorithm(tmpdir):
    print("\n[Case 4] Unsupported algorithm (SVC) -- should raise ModelValidationError")
    df = make_dataset(seed=3)
    X, y = df.drop(columns=["target"]), df["target"]
    model = SVC()
    model.fit(X, y)
    model_path = str(Path(tmpdir) / "svc_model.pkl")
    joblib.dump(model, model_path)

    train_path = write_csv(df.iloc[:150], tmpdir, "train4.csv")
    test_path = write_csv(df.iloc[150:], tmpdir, "test4.csv")

    try:
        handle_existing_model_upload(model_path, train_path, test_path, "target")
        raise AssertionError("Expected ModelValidationError, but none was raised")
    except ModelValidationError as e:
        print("  PASS: correctly rejected ->", e.message)


def case_bare_estimator_with_categorical_column(tmpdir):
    print("\n[Case 5] Bare estimator + unencoded categorical column -- should raise DatasetValidationError")
    df = make_dataset(seed=4)
    X, y = df.drop(columns=["target"]), df["target"]
    model = LogisticRegression()
    model.fit(X, y)
    model_path = str(Path(tmpdir) / "bare_model2.pkl")
    joblib.dump(model, model_path)

    # Corrupt one column into a non-numeric/categorical column post-hoc,
    # simulating a user uploading raw (unencoded) data against a bare estimator.
    df_bad = df.copy()
    df_bad["f1"] = df_bad["f1"].apply(lambda v: "high" if v > 0 else "low")

    train_path = write_csv(df_bad.iloc[:150], tmpdir, "train5.csv")
    test_path = write_csv(df_bad.iloc[150:], tmpdir, "test5.csv")

    try:
        handle_existing_model_upload(model_path, train_path, test_path, "target")
        raise AssertionError("Expected DatasetValidationError, but none was raised")
    except DatasetValidationError as e:
        print("  PASS: correctly rejected ->", e.message, "|", e.details)


def case_schema_mismatch(tmpdir):
    print("\n[Case 6] Train/test column mismatch -- should raise DatasetValidationError")
    df = make_dataset(seed=5)
    X, y = df.drop(columns=["target"]), df["target"]
    pipeline = Pipeline([("model", RandomForestClassifier(n_estimators=10, random_state=0))])
    pipeline.fit(X, y)
    model_path = str(Path(tmpdir) / "pipeline_model2.pkl")
    joblib.dump(pipeline, model_path)

    train_path = write_csv(df.iloc[:150], tmpdir, "train6.csv")
    test_bad = df.iloc[150:].drop(columns=["f1"])
    test_path = write_csv(test_bad, tmpdir, "test6.csv")

    try:
        handle_existing_model_upload(model_path, train_path, test_path, "target")
        raise AssertionError("Expected DatasetValidationError, but none was raised")
    except DatasetValidationError as e:
        print("  PASS: correctly rejected ->", e.message, "|", e.details)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        case_native_pipeline(tmpdir)
        case_bare_estimator(tmpdir)
        case_unfitted_model(tmpdir)
        case_unsupported_algorithm(tmpdir)
        case_bare_estimator_with_categorical_column(tmpdir)
        case_schema_mismatch(tmpdir)
    print("\nAll ingestion tests passed.")
