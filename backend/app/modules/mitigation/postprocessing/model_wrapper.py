"""
Wraps an already-fitted Pipeline with a fitted AIF360 post-processing
adjuster (e.g. CalibratedEqOddsPostprocessing). No retraining occurs --
predict()/predict_proba() call the base pipeline first, then apply the
group-conditional adjustment to its output.

Must be picklable on its own (joblib.dump/load), since this becomes a
version's saved artifact exactly like a retrained Pipeline does -- the
"any version downloads as one usable .pkl" guarantee must hold
regardless of which mitigation category produced it.
"""

import numpy as np
import pandas as pd


class PostProcessedModelWrapper:
    def __init__(
        self,
        base_pipeline,
        adjuster,
        protected_attribute: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ):
        self.base_pipeline = base_pipeline
        self.adjuster = adjuster
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.positive_class = positive_class
        # classes_ mirrors the base estimator's, needed by anything
        # downstream that inspects it the same way it would a Pipeline
        # (e.g. evaluation._resolve_positive_class).
        self.classes_ = self._final_estimator().classes_

    def _final_estimator(self):
        return self.base_pipeline.steps[-1][1]

    @property
    def steps(self):
        """
        Passthrough to the base pipeline's steps, so downstream code that
        expects a Pipeline-shaped object (SHAP explainer selection via
        get_final_estimator, pipeline_utils.split_pipeline for feature
        transforms) keeps working without special-casing this wrapper.

        Known limitation: SHAP explains the BASE model's decision
        process, not the post-processing adjustment layered on top of
        it -- there's no meaningful way to attribute feature importance
        to a per-group threshold/mixing-rate adjustment the way SHAP
        attributes it to a tree or linear model. This is disclosed here
        rather than silently produced as if it were a full explanation.
        """
        return self.base_pipeline.steps

    def _build_dataset(self, X: pd.DataFrame, scores: np.ndarray, hard_labels: np.ndarray):
        from aif360.datasets import BinaryLabelDataset

        positive_idx = list(self.classes_).index(self.positive_class)
        df = pd.DataFrame({
            self.protected_attribute: X[self.protected_attribute].map(
                {self.privileged_value: 1, self.unprivileged_value: 0}
            ).values,
            "_label": hard_labels,
        })
        bld = BinaryLabelDataset(
            df=df, label_names=["_label"],
            protected_attribute_names=[self.protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )
        bld.scores = scores[:, positive_idx].reshape(-1, 1)
        return bld, positive_idx

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Deliberately derives probabilities from the ADJUSTED hard labels
        (0.0/1.0), not from adjuster.predict()'s .scores field. Some
        AIF360 postprocessing algorithms (e.g. RejectOptionClassification)
        only ever set .labels and leave .scores as a shallow copy of the
        original, unadjusted input -- reading .scores directly would
        silently return stale probabilities that disagree with what
        predict() actually decided. This guarantees predict_proba's
        argmax always matches predict(), for every postprocessing
        strategy uniformly, at the cost of losing soft-score granularity
        for algorithms (like Calibrated Equalized Odds) that do compute
        genuine adjusted scores.
        """
        adjusted_labels = self.predict(X)
        positive_idx = list(self.classes_).index(self.positive_class)
        out = np.zeros((len(X), len(self.classes_)))
        is_positive = (adjusted_labels == self.positive_class)
        out[is_positive, positive_idx] = 1.0
        out[~is_positive, 1 - positive_idx] = 1.0
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_scores = self.base_pipeline.predict_proba(X)
        base_hard = self.base_pipeline.predict(X)
        hard_binary = (base_hard == self.positive_class).astype(int)

        bld, _ = self._build_dataset(X, base_scores, hard_binary)
        adjusted = self.adjuster.predict(bld)
        adjusted_labels = np.asarray(adjusted.labels).ravel()

        negative_class = [c for c in self.classes_ if c != self.positive_class][0]
        return np.where(adjusted_labels == 1, self.positive_class, negative_class)
