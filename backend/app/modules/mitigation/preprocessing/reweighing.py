"""
Reweighing bias mitigation via AIF360.

Produces per-row sample weights only, leaving the training data itself
unchanged -- it does NOT retrain anything itself (that's
retraining.model_cloner's job). This keeps the module boundary intact:
mitigation computes what should change, retraining applies it.
"""

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing as AIF360Reweighing

from app.modules.fairness.dataset_utils import to_binary_label_dataset, encode_binary_columns
from app.modules.mitigation.base import PreprocessingStrategy, PreprocessingResult


class ReweighingStrategy(PreprocessingStrategy):
    name = "Reweighing"

    def apply(
        self,
        train_df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ) -> PreprocessingResult:
        encoded = encode_binary_columns(
            train_df, protected_attribute, target_column,
            privileged_value, unprivileged_value, positive_class,
        )

        binary_label_dataset = to_binary_label_dataset(
            encoded, protected_attribute, target_column,
            favorable_label=1, unfavorable_label=0,
        )

        privileged_groups = [{protected_attribute: 1}]
        unprivileged_groups = [{protected_attribute: 0}]

        reweigher = AIF360Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        reweighted_dataset = reweigher.fit_transform(binary_label_dataset)

        # AIF360 preserves row order (verified against source df row-for-row
        # during development), so these weights align directly with
        # train_df's row order -- no re-indexing needed.
        weights = reweighted_dataset.instance_weights.ravel()

        # Reweighing never touches feature values -- the training data
        # passed to retraining is the original, unchanged train_df.
        return PreprocessingResult(
            transformed_train_df=train_df,
            sample_weights=weights,
        )
