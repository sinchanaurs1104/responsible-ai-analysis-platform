"""
Tests for Calibrated Equalized Odds Postprocessing.

Documented finding (verified against AIF360's actual source, not
speculation): CalibratedEqOddsPostprocessing.predict() works by randomly
replacing a fraction of one group's scores with a constant base-rate
value (a per-instance Bernoulli draw, mixing rate fixed at fit time) to
statistically equalize a chosen cost metric between groups. This is
inherently high-variance -- it does not recalibrate based on features,
it injects randomized noise into a subset of one group's predictions.
On a modest, noisy calibration set this can overshoot out-of-sample,
sometimes worsening SPD/EOD/AOD rather than improving them. This was
confirmed by reading AIF360's own fit()/predict() implementation, and
observed consistently across multiple seeds during development with
cost_constraint='weighted' (AIF360's own default, used here) showing the
smallest degradation of the three available cost constraints.

Because of this, these tests do NOT assert fairness must improve --
only that the mechanism runs correctly, produces a valid wrapped model,
and that the wrapped model is a real, picklable, predict-compatible
artifact. This mirrors how test_disparate_impact_remover.py handles a
similar honest, non-improving empirical result.
"""

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


def make_biased_dataset(n=500, seed=0, gap=13000, noise_rate=0.1):
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
    print("\n[Case 1] CEO registered as post-processing")
    reg = get_registration("Calibrated Equalized Odds Postprocessing")
    assert reg.category == "post"
    assert reg.strategy.name == "Calibrated Equalized Odds Postprocessing"
    print("  PASS")


def case_wrap_produces_valid_model():
    print("\n[Case 2] wrap() produces a valid, picklable, predict-compatible model")
    df = make_biased_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    reg = get_registration("Calibrated Equalized Odds Postprocessing")
    wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)

    X_test = ctx.test_df.drop(columns=["approved"])
    preds = wrapped.predict(X_test)
    proba = wrapped.predict_proba(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(proba.sum(axis=1), 1)
    assert list(wrapped.classes_) == list(ctx.pipeline.steps[-1][1].classes_)

    joblib.dump(wrapped, "/tmp/_ceo_test.pkl")
    reloaded = joblib.load("/tmp/_ceo_test.pkl")
    reloaded_preds = reloaded.predict(X_test)
    assert (preds == reloaded_preds).all(), "predictions must survive a pickle round-trip"
    print(f"  PASS: {len(preds)} predictions, pickle round-trip matches")


def case_no_retraining_occurred():
    print("\n[Case 3] base pipeline is untouched -- no retraining occurs for post-processing")
    df = make_biased_dataset(seed=2)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    perf_before = evaluate_model(ctx)
    reg = get_registration("Calibrated Equalized Odds Postprocessing")
    wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)
    perf_after = evaluate_model(ctx)
    assert perf_before.accuracy == perf_after.accuracy, "wrapping must not mutate the base pipeline"
    assert wrapped.base_pipeline is ctx.pipeline
    print("  PASS")


def case_shap_and_fairness_work_on_wrapped_model():
    print("\n[Case 4] full analysis stack (fairness, SHAP via .steps passthrough) runs on the wrapped model")
    df = make_biased_dataset(seed=3)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    reg = get_registration("Calibrated Equalized Odds Postprocessing")
    wrapped = reg.strategy.wrap(ctx.pipeline, ctx.train_df, "gender", "approved", "male", "female", 1)

    ctx2 = TrainedModelContext(
        pipeline=wrapped, source=ctx.source, preprocessing_status=ctx.preprocessing_status,
        estimator_step_name=ctx.estimator_step_name, algorithm_name=ctx.algorithm_name,
        train_df=ctx.train_df, test_df=ctx.test_df, target_column=ctx.target_column,
        validation_warnings=[],
    )
    fairness = compute_fairness_metrics(ctx2, "gender", "male", "female")
    assert fairness.statistical_parity_difference is not None

    from app.modules.explainability.shap_explainer import compute_shap_importance
    shap_result = compute_shap_importance(ctx2)
    assert len(shap_result.top_features) > 0
    print(f"  PASS: fairness computed (SPD={fairness.statistical_parity_difference}), "
          f"SHAP top feature: {shap_result.top_features[0].feature_name}")


if __name__ == "__main__":
    case_registry_lookup()
    case_wrap_produces_valid_model()
    case_no_retraining_occurred()
    case_shap_and_fairness_work_on_wrapped_model()
    print("\nAll Calibrated Equalized Odds Postprocessing tests passed.")
