"""End-to-end sanity checks for the explainability module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.training.trainer import train_new_model
from app.modules.explainability.shap_explainer import compute_shap_importance
from app.modules.explainability.error_analysis import analyze_errors
from app.modules.explainability.counterfactuals import generate_counterfactuals


def make_biased_dataset(n=400, seed=0):
    """Dataset where the 'region' subgroup 'south' is deliberately noisier
    (harder to predict), so error_analysis has something real to flag."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "gender": rng.choice(["male", "female"], size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
    })
    clean_rule = ((df["income"] > 50000) & (df["age"] > 25)).astype(int)
    noise = rng.integers(0, 2, size=n)
    is_south = (df["region"] == "south").values
    df["approved"] = np.where(is_south, noise, clean_rule)
    return df


def case_shap_random_forest():
    print("\n[Case 1] SHAP importance -- RandomForest (tree explainer)")
    df = make_biased_dataset(seed=1)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    result = compute_shap_importance(ctx)
    assert result.explainer_type == "tree"
    assert len(result.top_features) > 0
    assert all(f.importance_score >= 0 for f in result.top_features)
    top_names = [f.feature_name for f in result.top_features]
    print(f"  PASS: top features -> {top_names[:5]}")


def case_shap_logistic_regression():
    print("\n[Case 2] SHAP importance -- LogisticRegression (linear explainer)")
    df = make_biased_dataset(seed=2)
    ctx = train_new_model(df, target_column="approved", algorithm_name="LogisticRegression")
    result = compute_shap_importance(ctx)
    assert result.explainer_type == "linear"
    assert len(result.top_features) > 0
    print(f"  PASS: top features -> {[f.feature_name for f in result.top_features[:5]]}")


def case_error_analysis_flags_known_subgroup():
    print("\n[Case 3] Error analysis -- should flag the deliberately noisy 'south' subgroup")
    df = make_biased_dataset(seed=3)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    result = analyze_errors(ctx)
    assert 0.0 <= result.overall_accuracy <= 1.0
    flagged_columns = {s.column for s in result.worst_subgroups}
    south_flagged = any(
        s.column == "region" and s.subgroup == "south" for s in result.worst_subgroups
    )
    print(f"  overall_accuracy={result.overall_accuracy}")
    for s in result.worst_subgroups:
        print(f"    {s.column}={s.subgroup} | n={s.subgroup_size} acc={s.subgroup_accuracy} gap={s.accuracy_gap}")
    assert south_flagged, "expected the noisy 'south' subgroup to be flagged as underperforming"
    print("  PASS: 'south' subgroup correctly flagged as underperforming")


def case_counterfactuals():
    print("\n[Case 4] DiCE counterfactuals -- RandomForest")
    df = make_biased_dataset(seed=4)
    ctx = train_new_model(df, target_column="approved", algorithm_name="RandomForestClassifier")
    result = generate_counterfactuals(ctx, num_instances=2, total_cfs=2)
    assert result.method == "random"
    print(f"  generated {len(result.examples)} counterfactual example set(s)")
    for ex in result.examples:
        assert ex.original_prediction != ex.counterfactual_prediction, (
            "counterfactual should flip the predicted class"
        )
        assert len(ex.counterfactual_instances) > 0
        print(f"    original_pred={ex.original_prediction} -> counterfactual_pred={ex.counterfactual_prediction} "
              f"({len(ex.counterfactual_instances)} instance(s))")
    assert len(result.examples) > 0, "expected at least one successful counterfactual"
    print("  PASS: counterfactuals generated and flip the predicted class")


if __name__ == "__main__":
    case_shap_random_forest()
    case_shap_logistic_regression()
    case_error_analysis_flags_known_subgroup()
    case_counterfactuals()
    print("\nAll explainability tests passed.")
