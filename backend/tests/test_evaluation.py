"""
End-to-end sanity checks for evaluation.metrics.

Deliberately runs evaluation against TrainedModelContext objects produced
by BOTH training.trainer (Workflow A) and ingestion.upload_handler
(Workflow B, both native-Pipeline and bare-estimator cases) -- this is
the actual proof that the convergence point works: evaluation.metrics
never branches on where the context came from.
"""

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.training.trainer import train_new_model
from app.modules.ingestion.upload_handler import handle_existing_model_upload
from app.modules.evaluation.metrics import evaluate_model


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


def make_numeric_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    return df


def assert_valid_metrics(pm, expected_test_size):
    for value in [pm.accuracy, pm.precision, pm.recall, pm.f1_score]:
        assert 0.0 <= value <= 1.0, f"metric out of [0,1] range: {value}"
    cm = pm.confusion_matrix
    total = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive
    assert total == expected_test_size, f"confusion matrix total {total} != test size {expected_test_size}"
    assert pm.test_set_size == expected_test_size


def case_from_training_workflow():
    print("\n[Case 1] Evaluate a context produced by training.trainer (Workflow A)")
    df = make_mixed_dataset(seed=10)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    pm = evaluate_model(ctx)
    assert_valid_metrics(pm, expected_test_size=len(ctx.test_df))
    print(f"  PASS: accuracy={pm.accuracy} precision={pm.precision} recall={pm.recall} f1={pm.f1_score}")
    print(f"        confusion_matrix={pm.confusion_matrix}")


def case_from_ingestion_native_pipeline(tmpdir):
    print("\n[Case 2] Evaluate a context produced by ingestion (native Pipeline, Workflow B)")
    df = make_numeric_dataset(seed=11)
    X, y = df.drop(columns=["target"]), df["target"]
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=20, random_state=42)),
    ])
    pipeline.fit(X, y)

    model_path = str(Path(tmpdir) / "pipeline_model.pkl")
    joblib.dump(pipeline, model_path)
    train_path = str(Path(tmpdir) / "train.csv")
    test_path = str(Path(tmpdir) / "test.csv")
    df.iloc[:150].to_csv(train_path, index=False)
    df.iloc[150:].to_csv(test_path, index=False)

    ctx = handle_existing_model_upload(model_path, train_path, test_path, "target")
    pm = evaluate_model(ctx)
    assert_valid_metrics(pm, expected_test_size=len(ctx.test_df))
    print(f"  PASS: accuracy={pm.accuracy} precision={pm.precision} recall={pm.recall} f1={pm.f1_score}")


def case_from_ingestion_bare_estimator(tmpdir):
    print("\n[Case 3] Evaluate a context produced by ingestion (wrapped bare estimator, Workflow B)")
    df = make_numeric_dataset(seed=12)
    X, y = df.drop(columns=["target"]), df["target"]
    model = LogisticRegression()
    model.fit(X, y)

    model_path = str(Path(tmpdir) / "bare_model.pkl")
    joblib.dump(model, model_path)
    train_path = str(Path(tmpdir) / "train2.csv")
    test_path = str(Path(tmpdir) / "test2.csv")
    df.iloc[:150].to_csv(train_path, index=False)
    df.iloc[150:].to_csv(test_path, index=False)

    ctx = handle_existing_model_upload(model_path, train_path, test_path, "target")
    pm = evaluate_model(ctx)
    assert_valid_metrics(pm, expected_test_size=len(ctx.test_df))
    print(f"  PASS (bare estimator, wrapped): accuracy={pm.accuracy} f1={pm.f1_score}")


def case_perfect_separation_sanity():
    print("\n[Case 4] Sanity check: near-perfectly-separable data should score high")
    rng = np.random.default_rng(20)
    n = 200
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)  # trivially separable on f1
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = y

    ctx = train_new_model(df, target_column="target", algorithm_name="LogisticRegression")
    pm = evaluate_model(ctx)
    assert pm.accuracy > 0.9, f"expected high accuracy on separable data, got {pm.accuracy}"
    print(f"  PASS: accuracy={pm.accuracy} (as expected, high on separable data)")


if __name__ == "__main__":
    case_from_training_workflow()
    with tempfile.TemporaryDirectory() as tmpdir:
        case_from_ingestion_native_pipeline(tmpdir)
    with tempfile.TemporaryDirectory() as tmpdir:
        case_from_ingestion_bare_estimator(tmpdir)
    case_perfect_separation_sanity()
    print("\nAll evaluation tests passed.")
