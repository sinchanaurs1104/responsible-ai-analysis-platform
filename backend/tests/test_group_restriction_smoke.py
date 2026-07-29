import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from app.modules.training.trainer import train_new_model
from app.modules.fairness.dataset_utils import needs_group_restriction, restrict_df_to_groups
from app.modules.fairness.metrics import validate_protected_attribute_config


def make_multi_race_dataset(n=600, seed=0):
    rng = np.random.default_rng(seed)
    races = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], size=n)
    income = rng.normal(50000, 15000, n)
    label = (income > 50000).astype(int)
    return pd.DataFrame({"race": races, "income": income, "approved": label})


def case_needs_restriction_detected():
    df = make_multi_race_dataset()
    assert needs_group_restriction(df, "race", "White", "Black") is True
    assert needs_group_restriction(
        df[df["race"].isin(["White", "Black"])], "race", "White", "Black"
    ) is False
    print("  PASS: needs_group_restriction correctly flags >2 vs exactly-2 groups")


def case_restrict_and_retrain_internally_trained():
    df = make_multi_race_dataset()
    ctx = train_new_model(df, target_column="approved", algorithm_name="LogisticRegression")

    combined = pd.concat([ctx.train_df, ctx.test_df], ignore_index=True)
    assert needs_group_restriction(combined, "race", "White", "Black")

    scoped_df = restrict_df_to_groups(combined, "race", "White", "Black")
    assert set(scoped_df["race"].unique()) == {"White", "Black"}
    assert len(scoped_df) < len(combined)

    new_ctx = train_new_model(scoped_df, target_column="approved", algorithm_name="LogisticRegression")
    combined_new = pd.concat([new_ctx.train_df, new_ctx.test_df], ignore_index=True)
    assert set(combined_new["race"].unique()) == {"White", "Black"}

    # Now the strict binary validator (used by fairness.metrics) must pass.
    validate_protected_attribute_config(new_ctx, "race", "White", "Black")
    print(f"  PASS: retrained context scoped to 2 groups ({len(scoped_df)}/{len(combined)} rows retained), validator accepts it")


if __name__ == "__main__":
    print("\n[Case 1] needs_group_restriction detects multi-valued attribute")
    case_needs_restriction_detected()

    print("\n[Case 2] restrict + retrain path produces a validator-compatible context")
    case_restrict_and_retrain_internally_trained()

    print("\nAll group-restriction smoke tests passed.")
