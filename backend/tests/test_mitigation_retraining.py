"""
End-to-end sanity checks for mitigation + retraining.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from app.core.exceptions import ModelValidationError
from app.modules.training.trainer import train_new_model
from app.modules.evaluation.metrics import evaluate_model, _resolve_positive_class
from app.modules.fairness.metrics import compute_fairness_metrics
from app.modules.fairness.insight_engine import derive_fairness_finding
from app.modules.mitigation.registry import get_registration, MITIGATION_REGISTRY, CATEGORY_PREPROCESSING
from app.modules.mitigation.base import PreprocessingStrategy, PreprocessingResult
from app.modules.retraining.model_cloner import retrain_with_preprocessing_result


def make_biased_dataset(n=600, seed=0, gap=20000, noise_rate=0.08):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    threshold = np.where(gender == "female", 45000 + gap, 45000)
    approved = (income > threshold).astype(int)
    flip_mask = rng.random(n) < noise_rate
    approved = np.where(flip_mask, 1 - approved, approved)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "approved": approved})


def run_full_mitigation_cycle(algorithm_name: str, seed: int):
    df = make_biased_dataset(seed=seed)
    ctx_v1 = train_new_model(df, target_column="approved", algorithm_name=algorithm_name)

    perf_v1 = evaluate_model(ctx_v1)
    fairness_v1 = compute_fairness_metrics(ctx_v1, "gender", privileged_value="male", unprivileged_value="female")
    finding_v1 = derive_fairness_finding(fairness_v1)

    registration = get_registration(finding_v1.suggested_mitigation)
    assert registration.category == CATEGORY_PREPROCESSING
    positive_class = _resolve_positive_class(ctx_v1)
    preprocessing_result = registration.strategy.apply(
        ctx_v1.train_df, "gender", "approved", "male", "female", positive_class
    )

    ctx_v2 = retrain_with_preprocessing_result(ctx_v1, preprocessing_result)
    perf_v2 = evaluate_model(ctx_v2)
    fairness_v2 = compute_fairness_metrics(ctx_v2, "gender", privileged_value="male", unprivileged_value="female")
    finding_v2 = derive_fairness_finding(fairness_v2)

    return perf_v1, fairness_v1, finding_v1, perf_v2, fairness_v2, finding_v2


def case_registry_lookup():
    print("\n[Case 1] Registry lookup")
    registration = get_registration("Reweighing")
    assert isinstance(registration.strategy, PreprocessingStrategy)
    assert registration.strategy.name == "Reweighing"
    assert registration.category == CATEGORY_PREPROCESSING
    assert "Reweighing" in MITIGATION_REGISTRY
    print("  PASS")


def case_unknown_strategy_rejected():
    print("\n[Case 2] Unknown method rejected")
    from app.core.exceptions import ConfigValidationError
    try:
        get_registration("SomeFutureMethod")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message)


def case_preprocessing_result_shape_and_sign():
    print("\n[Case 3] PreprocessingResult shape/sign")
    df = make_biased_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    positive_class = _resolve_positive_class(ctx)
    registration = get_registration("Reweighing")
    result = registration.strategy.apply(ctx.train_df, "gender", "approved", "male", "female", positive_class)
    assert isinstance(result, PreprocessingResult)
    assert result.sample_weights is not None
    assert len(result.sample_weights) == len(ctx.train_df)
    assert (result.sample_weights > 0).all()
    assert result.transformed_train_df is ctx.train_df or result.transformed_train_df.equals(ctx.train_df)
    print(f"  PASS: {len(result.sample_weights)} weights, range "
          f"[{result.sample_weights.min():.4f}, {result.sample_weights.max():.4f}]")


def case_mismatched_weight_length_rejected():
    print("\n[Case 4] Mismatched weight length rejected")
    df = make_biased_dataset(seed=2)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    bad_result = PreprocessingResult(transformed_train_df=ctx.train_df, sample_weights=np.ones(len(ctx.train_df) - 5))
    try:
        retrain_with_preprocessing_result(ctx, bad_result)
        raise AssertionError("Expected ModelValidationError")
    except ModelValidationError as e:
        print("  PASS:", e.message)


def case_full_cycle_random_forest():
    print("\n[Case 5] Full cycle RandomForest (averaged, 5 seeds)")
    seeds = [10, 20, 30, 40, 50]
    spd_v1_list, spd_v2_list, acc_v1_list, acc_v2_list = [], [], [], []
    for seed in seeds:
        perf_v1, fairness_v1, _, perf_v2, fairness_v2, _ = run_full_mitigation_cycle("RandomForestClassifier", seed=seed)
        spd_v1_list.append(abs(fairness_v1.statistical_parity_difference))
        spd_v2_list.append(abs(fairness_v2.statistical_parity_difference))
        acc_v1_list.append(perf_v1.accuracy)
        acc_v2_list.append(perf_v2.accuracy)
        print(f"  seed={seed}: |SPD| {spd_v1_list[-1]:.4f} -> {spd_v2_list[-1]:.4f}")
    avg_spd_v1, avg_spd_v2 = np.mean(spd_v1_list), np.mean(spd_v2_list)
    avg_acc_drop = np.mean(acc_v1_list) - np.mean(acc_v2_list)
    print(f"  avg |SPD|: {avg_spd_v1:.4f} -> {avg_spd_v2:.4f} | avg accuracy change: {-avg_acc_drop:+.4f}")
    assert avg_spd_v2 < avg_spd_v1
    assert avg_acc_drop < 0.15
    print("  PASS")


def case_full_cycle_logistic_regression():
    print("\n[Case 6] Full cycle LogisticRegression (averaged, 5 seeds)")
    seeds = [11, 21, 31, 41, 51]
    spd_v1_list, spd_v2_list = [], []
    for seed in seeds:
        _, fairness_v1, _, _, fairness_v2, _ = run_full_mitigation_cycle("LogisticRegression", seed=seed)
        spd_v1_list.append(abs(fairness_v1.statistical_parity_difference))
        spd_v2_list.append(abs(fairness_v2.statistical_parity_difference))
        print(f"  seed={seed}: |SPD| {spd_v1_list[-1]:.4f} -> {spd_v2_list[-1]:.4f}")
    avg_spd_v1, avg_spd_v2 = np.mean(spd_v1_list), np.mean(spd_v2_list)
    print(f"  avg |SPD|: {avg_spd_v1:.4f} -> {avg_spd_v2:.4f}")
    assert avg_spd_v2 < avg_spd_v1
    print("  PASS")


def case_v1_and_v2_are_independent_objects():
    print("\n[Case 7] V1/V2 independence")
    df = make_biased_dataset(seed=12)
    ctx_v1 = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    perf_v1_before = evaluate_model(ctx_v1)
    positive_class = _resolve_positive_class(ctx_v1)
    registration = get_registration("Reweighing")
    result = registration.strategy.apply(ctx_v1.train_df, "gender", "approved", "male", "female", positive_class)
    ctx_v2 = retrain_with_preprocessing_result(ctx_v1, result)
    perf_v1_after = evaluate_model(ctx_v1)
    assert perf_v1_before.accuracy == perf_v1_after.accuracy
    assert ctx_v2.pipeline is not ctx_v1.pipeline
    print("  PASS")


if __name__ == "__main__":
    case_registry_lookup()
    case_unknown_strategy_rejected()
    case_preprocessing_result_shape_and_sign()
    case_mismatched_weight_length_rejected()
    case_full_cycle_random_forest()
    case_full_cycle_logistic_regression()
    case_v1_and_v2_are_independent_objects()
    print("\nAll mitigation + retraining tests passed.")
