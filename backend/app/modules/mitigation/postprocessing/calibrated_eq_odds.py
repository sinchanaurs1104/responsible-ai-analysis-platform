"""
Calibrated Equalized Odds Postprocessing via AIF360.

No retraining: fits a group-conditional score adjustment against the
base pipeline's own predictions on the training data, then wraps the
base pipeline so its predictions get adjusted at inference time. This
is the category usable when a model can't be retrained at all.
"""

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing as AIF360CEO

from app.modules.mitigation.base import PostprocessingStrategy
from app.modules.mitigation.postprocessing.model_wrapper import PostProcessedModelWrapper

COST_CONSTRAINT = "weighted"
SEED = 42


class CalibratedEqualizedOddsStrategy(PostprocessingStrategy):
    name = "Calibrated Equalized Odds Postprocessing"

    def get_hyperparameters(self) -> dict:
        return {"cost_constraint": COST_CONSTRAINT, "seed": SEED}

    def wrap(
        self,
        base_pipeline,
        calibration_df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ):
        final_estimator = base_pipeline.steps[-1][1]
        classes = list(final_estimator.classes_)
        positive_idx = classes.index(positive_class)

        X_calib = calibration_df.drop(columns=[target_column])
        y_true_binary = (calibration_df[target_column] == positive_class).astype(int).values
        scores = base_pipeline.predict_proba(X_calib)[:, positive_idx]
        hard_pred_binary = (base_pipeline.predict(X_calib) == positive_class).astype(int)

        protected_encoded = X_calib[protected_attribute].map(
            {privileged_value: 1, unprivileged_value: 0}
        ).values

        df_true = pd.DataFrame({protected_attribute: protected_encoded, "_label": y_true_binary})
        dataset_true = BinaryLabelDataset(
            df=df_true, label_names=["_label"],
            protected_attribute_names=[protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )

        df_pred = pd.DataFrame({protected_attribute: protected_encoded, "_label": hard_pred_binary})
        dataset_pred = BinaryLabelDataset(
            df=df_pred, label_names=["_label"],
            protected_attribute_names=[protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )
        dataset_pred.scores = scores.reshape(-1, 1)

        adjuster = AIF360CEO(
            privileged_groups=[{protected_attribute: 1}],
            unprivileged_groups=[{protected_attribute: 0}],
            cost_constraint=COST_CONSTRAINT,
            seed=SEED,
        )
        adjuster = adjuster.fit(dataset_true, dataset_pred)

        return PostProcessedModelWrapper(
            base_pipeline=base_pipeline,
            adjuster=adjuster,
            protected_attribute=protected_attribute,
            privileged_value=privileged_value,
            unprivileged_value=unprivileged_value,
            positive_class=positive_class,
        )
