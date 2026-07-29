"""
Reject Option Classification via AIF360.

No retraining: flips predictions within a margin around the decision
boundary, favoring the unprivileged group -- a deterministic rule-based
adjustment (not randomized score-mixing like Calibrated Equalized Odds),
fit to keep Statistical Parity Difference within [metric_lb, metric_ub]
on a calibration set.
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import RejectOptionClassification as AIF360ROC

from app.modules.mitigation.base import PostprocessingStrategy
from app.modules.mitigation.postprocessing.model_wrapper import PostProcessedModelWrapper

LOW_CLASS_THRESH = 0.01
HIGH_CLASS_THRESH = 0.99
NUM_CLASS_THRESH = 100
NUM_ROC_MARGIN = 50
METRIC_NAME = "Statistical parity difference"
METRIC_UB = 0.05
METRIC_LB = -0.05


class RejectOptionClassificationStrategy(PostprocessingStrategy):
    name = "Reject Option Classification"

    def get_hyperparameters(self) -> dict:
        return {
            "low_class_thresh": LOW_CLASS_THRESH,
            "high_class_thresh": HIGH_CLASS_THRESH,
            "num_class_thresh": NUM_CLASS_THRESH,
            "num_ROC_margin": NUM_ROC_MARGIN,
            "metric_name": METRIC_NAME,
            "metric_ub": METRIC_UB,
            "metric_lb": METRIC_LB,
        }

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

        dataset_true = BinaryLabelDataset(
            df=pd.DataFrame({protected_attribute: protected_encoded, "_label": y_true_binary}),
            label_names=["_label"], protected_attribute_names=[protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )
        dataset_pred = BinaryLabelDataset(
            df=pd.DataFrame({protected_attribute: protected_encoded, "_label": hard_pred_binary}),
            label_names=["_label"], protected_attribute_names=[protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )
        dataset_pred.scores = scores.reshape(-1, 1)

        adjuster = AIF360ROC(
            unprivileged_groups=[{protected_attribute: 0}],
            privileged_groups=[{protected_attribute: 1}],
            low_class_thresh=LOW_CLASS_THRESH, high_class_thresh=HIGH_CLASS_THRESH,
            num_class_thresh=NUM_CLASS_THRESH, num_ROC_margin=NUM_ROC_MARGIN,
            metric_name=METRIC_NAME, metric_ub=METRIC_UB, metric_lb=METRIC_LB,
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
