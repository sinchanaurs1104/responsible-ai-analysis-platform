"""
Shared helper used by both shap_explainer and counterfactuals: splits a
Pipeline into "everything before the final step" (preprocessing) and
the final estimator, and applies that preprocessing to raw feature data.

Works uniformly whether the Pipeline has 1 step (wrapped bare estimator,
SDD "user_responsibility" case) or many steps (native uploaded Pipeline,
or the platform's own auto-built training Pipeline) -- this module never
needs to know which case it's in.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.modules.ingestion.model_loader import get_final_estimator


def split_pipeline(pipeline: Pipeline):
    """Returns (preprocessing_pipeline_or_None, final_estimator)."""
    final_estimator = get_final_estimator(pipeline)
    if len(pipeline.steps) > 1:
        preprocessing = Pipeline(pipeline.steps[:-1])
    else:
        preprocessing = None
    return preprocessing, final_estimator


def transform_features(pipeline: Pipeline, X: pd.DataFrame):
    """
    Returns (X_transformed_as_ndarray, feature_names).

    If the Pipeline has preprocessing steps, applies them and tries to
    recover output feature names via get_feature_names_out(). Falls back
    to generic names if that isn't available (some transformers don't
    implement it) or to the original column names if there is no
    preprocessing at all (single-step Pipeline).
    """
    preprocessing, _ = split_pipeline(pipeline)

    if preprocessing is None:
        X_transformed = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [
            f"feature_{i}" for i in range(X_transformed.shape[1])
        ]
        return X_transformed, feature_names

    X_transformed = preprocessing.transform(X)
    if hasattr(X_transformed, "toarray"):  # sparse output (e.g. OneHotEncoder)
        X_transformed = X_transformed.toarray()

    try:
        feature_names = list(preprocessing.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return X_transformed, feature_names
