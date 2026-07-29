"""
Global feature importance via SHAP.

Explainer type (tree vs linear) is looked up from the same
core.supported_models registry used by ingestion/training to validate
algorithms -- this is the "registry as single source of truth" pattern
from the SDD applied to explainability's own needs (§9).
"""

import numpy as np
import shap

from app.core.exceptions import ModelValidationError
from app.core.supported_models import get_model_info
from app.schemas.context import TrainedModelContext
from app.schemas.metrics import SHAPResult, FeatureImportance
from app.modules.explainability.pipeline_utils import split_pipeline, transform_features

MAX_SAMPLE_SIZE = 100
TOP_N_FEATURES = 10


def _clean_feature_name(transformed_name: str, original_columns: list[str]) -> str:
    """
    Maps a ColumnTransformer-produced name (e.g. 'cat__marital-status_Married-civ-spouse'
    or 'num__capital_gain') back to its original raw column name, so the UI
    never shows preprocessing artifacts. Falls back to a best-effort cleanup
    of the raw name if no original column matches.

    Deliberately matches on the LONGEST original column name that the
    remainder starts with, to correctly handle cases where one column
    name is a prefix of another (e.g. 'education' vs 'education-num').
    """
    remainder = transformed_name.split("__", 1)[-1] if "__" in transformed_name else transformed_name

    best_match = None
    for col in original_columns:
        normalized = col.replace(" ", "_").replace("-", "_")
        candidates = (col, normalized)
        for candidate in candidates:
            if remainder == candidate or remainder.startswith(candidate + "_"):
                if best_match is None or len(candidate) > len(best_match):
                    best_match = col
                break

    base_name = best_match if best_match is not None else remainder
    pretty = base_name.replace("-", " ").replace("_", " ").strip()
    return " ".join(word.capitalize() for word in pretty.split())


def compute_shap_importance(context: TrainedModelContext) -> SHAPResult:
    _, final_estimator = split_pipeline(context.pipeline)

    model_info = get_model_info(context.algorithm_name)
    explainer_type = model_info.get("shap_explainer")
    if explainer_type is None:
        raise ModelValidationError(
            f"No SHAP explainer is registered for algorithm "
            f"'{context.algorithm_name}'."
        )

    X_test = context.test_df.drop(columns=[context.target_column])
    sample = X_test.sample(
        n=min(MAX_SAMPLE_SIZE, len(X_test)), random_state=42
    )
    X_transformed, feature_names = transform_features(context.pipeline, sample)

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(X_transformed)
        # For binary classifiers, some SHAP versions return a list
        # [class_0_values, class_1_values]; take the positive class.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        # Some tree explainers return shape (n_samples, n_features, n_classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

    elif explainer_type == "linear":
        # LinearExplainer needs a background distribution; use the
        # sample itself (masker) -- adequate for an MVP-scale explanation.
        explainer = shap.LinearExplainer(final_estimator, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    else:
        raise ModelValidationError(
            f"Unsupported SHAP explainer type '{explainer_type}' "
            f"registered for algorithm '{context.algorithm_name}'."
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    original_columns = [c for c in context.train_df.columns if c != context.target_column]
    aggregated: dict[str, float] = {}
    for i, raw_name in enumerate(feature_names):
        clean_name = _clean_feature_name(raw_name, original_columns)
        aggregated[clean_name] = aggregated.get(clean_name, 0.0) + float(mean_abs_shap[i])

    top_items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_FEATURES]

    top_features = [
        FeatureImportance(feature_name=name, importance_score=round(score, 5))
        for name, score in top_items
    ]

    return SHAPResult(
        top_features=top_features,
        explained_sample_size=len(sample),
        explainer_type=explainer_type,
    )
