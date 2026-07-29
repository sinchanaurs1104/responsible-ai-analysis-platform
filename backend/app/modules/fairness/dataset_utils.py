"""
Shared helper for building an AIF360 BinaryLabelDataset from a plain
pandas DataFrame. Used by both fairness.metrics (to compute metrics)
and mitigation.reweighing (to compute reweighted sample weights) --
factored out here so both stay in sync on encoding conventions instead
of duplicating this logic.
"""

import warnings

import pandas as pd
from aif360.datasets import BinaryLabelDataset


def to_binary_label_dataset(
    df: pd.DataFrame,
    protected_attribute: str,
    target_column: str,
    favorable_label: int = 1,
    unfavorable_label: int = 0,
) -> BinaryLabelDataset:
    """
    df must already have the protected attribute and target column
    encoded as numeric 0/1 (privileged=1/unprivileged=0,
    positive_class=1/negative_class=0) -- callers are responsible for
    that encoding, since what counts as "privileged" or "positive" is
    business configuration, not something this helper should guess.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BinaryLabelDataset(
            df=df,
            label_names=[target_column],
            protected_attribute_names=[protected_attribute],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
        )


def encode_binary_columns(
    df: pd.DataFrame,
    protected_attribute: str,
    target_column: str,
    privileged_value,
    unprivileged_value,
    positive_class,
) -> pd.DataFrame:
    """
    Returns a minimal 2-column DataFrame (protected attribute + target)
    with both encoded numerically, ready for to_binary_label_dataset().
    Rows whose protected attribute value is neither privileged_value nor
    unprivileged_value are NOT filtered here -- callers must validate
    the column only contains the configured values beforehand
    (see fairness.metrics.validate_protected_attribute_config).
    """
    encoded = pd.DataFrame({
        protected_attribute: df[protected_attribute].map(
            {privileged_value: 1, unprivileged_value: 0}
        ),
        target_column: (df[target_column] == positive_class).astype(int),
    })
    return encoded


def needs_group_restriction(
    df: pd.DataFrame, protected_attribute: str, privileged_value, unprivileged_value
) -> bool:
    """
    True if the protected attribute column contains any value other than
    the two configured groups (e.g. COMPAS 'race' has 6 values but the
    user only selected 'White' vs 'Black'). AIF360's binary fairness
    metrics require a strictly binary protected attribute, so any such
    "extra" rows must be scoped out before fairness/mitigation runs --
    see restrict_df_to_groups.
    """
    observed = set(df[protected_attribute].dropna().unique().tolist())
    return not observed.issubset({privileged_value, unprivileged_value})


def restrict_df_to_groups(
    df: pd.DataFrame, protected_attribute: str, privileged_value, unprivileged_value
) -> pd.DataFrame:
    """
    Keeps only rows belonging to the two configured groups. Generic
    across any dataset/attribute -- not specific to any one column or
    dataset (e.g. works the same for COMPAS race, German 'sex', or any
    future upload with a non-binary protected attribute).
    """
    mask = df[protected_attribute].isin([privileged_value, unprivileged_value])
    return df[mask].reset_index(drop=True)
