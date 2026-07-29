"""
Abstract interfaces for bias mitigation strategies, one per AIF360
intervention category. Per the v2 design document (Phase 1), these are
deliberately separate ABCs rather than one shared interface, because the
three categories have genuinely different mechanisms:

- Pre-processing: transforms training data (weights, feature values, or
  both) before our existing training step runs.
- In-processing: trains its own fairness-constrained model from scratch
  -- the resulting model is not necessarily the same algorithm as V1.
- Post-processing: wraps an already-fitted model, adjusting predictions
  per group. No retraining occurs.

Only PreprocessingStrategy has a real implementation in Phase 1
(ReweighingStrategy). InprocessingStrategy and PostprocessingStrategy
exist now so the category structure is real and the orchestrator can be
written against it, even though no concrete strategy is registered in
those categories yet -- adding one later (Phase 4/5) means a new
subclass + one registry entry, not a change to this file or the
orchestrator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreprocessingResult:
    """
    Output of a PreprocessingStrategy. Exactly one of the two fields is
    meaningfully used by any given strategy today, but both are always
    present so retraining.model_cloner has one uniform shape to consume
    regardless of which mechanism produced it:

    - Reweighing sets sample_weights, leaves transformed_train_df equal
      to the original (unchanged) training data.
    - Disparate Impact Remover (Phase 3) will set transformed_train_df
      to the repaired feature values, leaving sample_weights as None.
    """
    transformed_train_df: pd.DataFrame
    sample_weights: np.ndarray | None = None


class PreprocessingStrategy(ABC):
    name: str

    def get_hyperparameters(self) -> dict:
        """Returns the fixed hyperparameters this strategy ran with
        (e.g. repair_level). Default empty -- override when a strategy
        has tunable constants worth recording per run."""
        return {}

    @abstractmethod
    def apply(
        self,
        train_df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ) -> PreprocessingResult:
        """
        Returns a PreprocessingResult describing how the training data
        should be adjusted before retraining. Row order of
        transformed_train_df (and sample_weights, if set) must match
        the input train_df's row order.
        """
        raise NotImplementedError


class InprocessingStrategy(ABC):
    """
    Not yet implemented by any concrete strategy (Phase 1 scope).
    Trains and returns its own fitted model directly -- the caller does
    not clone/refit context.pipeline the way it does for pre-processing;
    the resulting model may be a different algorithm than V1 entirely.
    """
    name: str

    def get_hyperparameters(self) -> dict:
        return {}

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ):
        """Returns a fitted, picklable model (e.g. a Pipeline) trained
        directly by this strategy."""
        raise NotImplementedError


class PostprocessingStrategy(ABC):
    """
    Not yet implemented by any concrete strategy (Phase 1 scope).
    Wraps an already-fitted Pipeline, adjusting its predictions per
    group. No retraining occurs -- this is the category usable when a
    model can't be retrained at all (e.g. an uploaded estimator that
    doesn't support sample_weight).
    """
    name: str

    def get_hyperparameters(self) -> dict:
        return {}

    @abstractmethod
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
        """Returns a new, picklable, predict()-compatible object that
        wraps base_pipeline with group-conditional adjustment."""
        raise NotImplementedError
