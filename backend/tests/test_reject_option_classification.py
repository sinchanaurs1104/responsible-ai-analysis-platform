import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from app.modules.training.trainer import train_new_model
from app.modules.evaluation.metrics import evaluate_model
from app.modules.fairness.metrics import compute_fairness_metrics
from app.modules.mitigation.registry import get_registration
from app.schemas.context import TrainedModelContext


def make_biased_dataset(n=800, seed=0, gap=13000, noise_rate=0.1):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    threshold = np.where(gender == "female", 45000 + gap, 45000)
    approved = (income > threshold).astype(int)
    flip_mask = rng.random(n) < noise_rate
    approved = np.where(flip_mask, 1 - approved, approved)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "approved": approved})


def case_registry_lookup():
    print("\n[Case 1] ROC registered as post-processing")
    reg = get_registration("Reject Option Classification")
    assert reg.category == "post"
    assert reg.strategy.name == "Reject Option Classification"
    print("  PASS")


def case_wrap_produces_valid_model():
    print("\n[Case 2] wrap() produces a valid, picklable, predict-compatible model")
    df = make_biased_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    reg = get_registration("Reject Option Classification")
    wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)

    X_test = ctx.test_df.drop(columns=["approved"])
    preds = wrapped.predict(X_test)
    proba = wrapped.predict_proba(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(proba.sum(axis=1), 1)

    joblib.dump(wrapped, "/tmp/_roc_test.pkl")
    reloaded = joblib.load("/tmp/_roc_test.pkl")
    assert (preds == reloaded.predict(X_test)).all()
    print(f"  PASS: {len(preds)} predictions, pickle round-trip matches")


def case_predict_proba_agrees_with_predict():
    print("\n[Case 3] predict_proba's argmax always matches predict() -- the bug we fixed")
    df = make_biased_dataset(seed=2)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    reg = get_registration("Reject Option Classification")
    wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)
    X_test = ctx.test_df.drop(columns=["approved"])
    preds = wrapped.predict(X_test)
    proba = wrapped.predict_proba(X_test)
    argmax_class = np.where(proba[:, 1] > proba[:, 0], 1, 0)
    assert (preds == argmax_class).all(), (
        "ROC only sets .labels, never .scores -- predict_proba must derive "
        "from the adjusted labels, not stale pre-adjustment scores"
    )
    print("  PASS")


def case_no_retraining_occurred():
    print("\n[Case 4] base pipeline is untouched")
    df = make_biased_dataset(seed=3)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    perf_before = evaluate_model(ctx)
    reg = get_registration("Reject Option Classification")
    reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)
    perf_after = evaluate_model(ctx)
    assert perf_before.accuracy == perf_after.accuracy
    print("  PASS")


def case_full_cycle_observation():
    print("\n[Case 5] Full cycle -- observed SPD change across 5 seeds (informational, not asserted)")
    seeds = [10, 20, 30, 40, 50]
    for seed in seeds:
        df = make_biased_dataset(seed=seed)
        ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        fair1 = compute_fairness_metrics(ctx, "gender", "male", "female")
        reg = get_registration("Reject Option Classification")
        wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)
        ctx2 = TrainedModelContext(
            pipeline=wrapped, source=ctx.source, preprocessing_status=ctx.preprocessing_status,
            estimator_step_name=ctx.estimator_step_name, algorithm_name=ctx.algorithm_name,
            train_df=ctx.train_df, test_df=ctx.test_df, target_column=ctx.target_column,
            validation_warnings=[],
        )
        fair2 = compute_fairness_metrics(ctx2, "gender", "male", "female")
        print(f"  seed={seed}: SPD {fair1.statistical_parity_difference:.4f} -> "
              f"{fair2.statistical_parity_difference:.4f}")
    print("  PASS (recorded for observation)")


if __name__ == "__main__":
    case_registry_lookup()
    case_wrap_produces_valid_model()
    case_predict_proba_agrees_with_predict()
    case_no_retraining_occurred()
    case_full_cycle_observation()
    print("\nAll Reject Option Classification tests passed.")
