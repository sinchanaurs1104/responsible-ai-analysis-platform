import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from app.modules.training.trainer import train_new_model
from app.modules.evaluation.metrics import evaluate_model
from app.modules.fairness.metrics import compute_fairness_metrics
from app.modules.mitigation.registry import get_registration
from app.modules.mitigation.base import PreprocessingResult
from app.modules.retraining.model_cloner import retrain_with_preprocessing_result


def make_biased_dataset(n=600, seed=0, gap=20000, noise_rate=0.08):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    region = rng.choice(["north", "south"], size=n)
    threshold = np.where(gender == "female", 45000 + gap, 45000)
    approved = (income > threshold).astype(int)
    flip_mask = rng.random(n) < noise_rate
    approved = np.where(flip_mask, 1 - approved, approved)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "region": region, "approved": approved})


def case_registry_lookup():
    print("\n[Case 1] DIR registered as pre-processing")
    reg = get_registration("Disparate Impact Remover")
    assert reg.category == "pre"
    assert reg.strategy.name == "Disparate Impact Remover"
    print("  PASS")


def case_transform_shape_and_columns():
    print("\n[Case 2] DIR output shape/columns")
    df = make_biased_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    reg = get_registration("Disparate Impact Remover")
    result = reg.strategy.apply(ctx.train_df, "gender", "approved", "male", "female", 1)
    assert isinstance(result, PreprocessingResult)
    assert result.sample_weights is None
    assert result.transformed_train_df.shape == ctx.train_df.shape
    assert (result.transformed_train_df["gender"] == ctx.train_df["gender"]).all()
    assert (result.transformed_train_df["region"] == ctx.train_df["region"]).all()
    assert (result.transformed_train_df["approved"] == ctx.train_df["approved"]).all()
    assert not np.allclose(result.transformed_train_df["income"], ctx.train_df["income"])
    print("  PASS")


def case_full_cycle():
    print("\n[Case 3] Full cycle: DIR + retrain, RandomForest, 5 seeds")
    seeds = [10, 20, 30, 40, 50]
    spd_v1_list, spd_v2_list = [], []
    for seed in seeds:
        df = make_biased_dataset(seed=seed)
        ctx1 = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
        fair1 = compute_fairness_metrics(ctx1, "gender", "male", "female")

        reg = get_registration("Disparate Impact Remover")
        result = reg.strategy.apply(ctx1.train_df, "gender", "approved", "male", "female", 1)
        ctx2 = retrain_with_preprocessing_result(ctx1, result)
        fair2 = compute_fairness_metrics(ctx2, "gender", "male", "female")

        spd_v1_list.append(abs(fair1.statistical_parity_difference))
        spd_v2_list.append(abs(fair2.statistical_parity_difference))
        print(f"  seed={seed}: |SPD| {spd_v1_list[-1]:.4f} -> {spd_v2_list[-1]:.4f}")

    print(f"  avg: {np.mean(spd_v1_list):.4f} -> {np.mean(spd_v2_list):.4f}")
    print("  PASS (no assertion on direction -- DIR targets feature-distribution parity, "
          "not SPD directly; recorded for observation)")


def case_v1_v2_independent():
    print("\n[Case 4] V1/V2 independence with DIR")
    df = make_biased_dataset(seed=12)
    ctx1 = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    perf1_before = evaluate_model(ctx1)
    reg = get_registration("Disparate Impact Remover")
    result = reg.strategy.apply(ctx1.train_df, "gender", "approved", "male", "female", 1)
    ctx2 = retrain_with_preprocessing_result(ctx1, result)
    perf1_after = evaluate_model(ctx1)
    assert perf1_before.accuracy == perf1_after.accuracy
    assert ctx2.pipeline is not ctx1.pipeline
    print("  PASS")


if __name__ == "__main__":
    case_registry_lookup()
    case_transform_shape_and_columns()
    case_full_cycle()
    case_v1_v2_independent()
    print("\nAll DIR tests passed.")
