"""End-to-end sanity checks for the fairness module."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from app.core.exceptions import ConfigValidationError, DatasetValidationError
from app.modules.training.trainer import train_new_model
from app.modules.fairness.metrics import compute_fairness_metrics
from app.modules.fairness.insight_engine import derive_fairness_finding


def make_biased_dataset(n=500, seed=0, gap=20000):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    threshold = np.where(gender == "female", 45000 + gap, 45000)
    approved = (income > threshold).astype(int)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "approved": approved})


def make_fair_dataset(n=500, seed=1):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    approved = (income > 45000).astype(int)  # same threshold regardless of gender
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "approved": approved})


def case_high_disparity_detected():
    print("\n[Case 1] Deliberately biased dataset -- should detect high disparity, SPD as driving factor")
    df = make_biased_dataset(seed=10, gap=20000)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    metrics = compute_fairness_metrics(ctx, "gender", privileged_value="male", unprivileged_value="female")
    finding = derive_fairness_finding(metrics)

    print(f"  SPD={metrics.statistical_parity_difference} DIR={metrics.disparate_impact_ratio} "
          f"EOD={metrics.equal_opportunity_difference} AOD={metrics.average_odds_difference}")
    print(f"  selection_rate male={metrics.privileged_selection_rate} female={metrics.unprivileged_selection_rate}")
    print(f"  finding: status={finding.status} driving_factor={finding.driving_factor} "
          f"disadvantaged_group={finding.disadvantaged_group} mitigation={finding.suggested_mitigation}")

    assert finding.status == "high_disparity"
    assert finding.driving_factor == "statistical_parity_difference"
    assert finding.disadvantaged_group == "female"
    assert finding.suggested_mitigation == "Reweighing"
    print("  PASS")


def case_fair_dataset_no_disparity():
    print("\n[Case 2] Fair dataset (same threshold both groups) -- should classify as fair")
    df = make_fair_dataset(seed=11)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    metrics = compute_fairness_metrics(ctx, "gender", privileged_value="male", unprivileged_value="female")
    finding = derive_fairness_finding(metrics)

    print(f"  SPD={metrics.statistical_parity_difference} status={finding.status} "
          f"mitigation={finding.suggested_mitigation}")
    assert finding.status == "fair"
    assert finding.suggested_mitigation == "None required"
    print("  PASS")


def case_unknown_protected_value_rejected():
    print("\n[Case 3] Protected attribute has a value outside privileged/unprivileged config -- should raise ConfigValidationError")
    df = make_biased_dataset(seed=12)
    df.loc[df.index[:5], "gender"] = "nonbinary"  # third value not configured
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    try:
        compute_fairness_metrics(ctx, "gender", privileged_value="male", unprivileged_value="female")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message, "|", e.details)


def case_missing_protected_attribute():
    print("\n[Case 4] Protected attribute column does not exist -- should raise ConfigValidationError")
    df = make_biased_dataset(seed=13)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    try:
        compute_fairness_metrics(ctx, "does_not_exist", privileged_value="male", unprivileged_value="female")
        raise AssertionError("Expected ConfigValidationError")
    except ConfigValidationError as e:
        print("  PASS:", e.message)


def case_small_group_warning():
    print("\n[Case 5] Tiny subgroup after split -- should set small_group_warning=True")
    rng = np.random.default_rng(14)
    n = 120
    # Only ~5% female -- after an 80/20 split, the test-set female group will be tiny.
    gender = rng.choice(["male", "female"], size=n, p=[0.95, 0.05])
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    approved = (income > 45000).astype(int)
    df = pd.DataFrame({"age": age, "income": income, "gender": gender, "approved": approved})

    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    metrics = compute_fairness_metrics(ctx, "gender", privileged_value="male", unprivileged_value="female")
    print(f"  privileged_group_size={metrics.privileged_group_size} "
          f"unprivileged_group_size={metrics.unprivileged_group_size} "
          f"small_group_warning={metrics.small_group_warning}")
    assert metrics.small_group_warning is True
    print("  PASS")


if __name__ == "__main__":
    case_high_disparity_detected()
    case_fair_dataset_no_disparity()
    case_unknown_protected_value_rejected()
    case_missing_protected_attribute()
    case_small_group_warning()
    print("\nAll fairness tests passed.")
