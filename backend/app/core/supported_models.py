"""
Single source of truth for which scikit-learn estimator classes the
platform supports.

Used by:
- ingestion.model_validators  -> reject unsupported algorithms early
- training.trainer            -> restrict algorithm choice in Workflow A
- explainability (later)      -> pick the right SHAP explainer type

Adding a new algorithm later is a one-line addition here — no other
module needs to change (see SDD §18, Future Extensibility).
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS = {
    "RandomForestClassifier": {
        "estimator_class": RandomForestClassifier,
        "shap_explainer": "tree",
        "supports_sample_weight": True,
        "default_params": {"n_estimators": 100, "random_state": 42, "min_samples_leaf": 5},
    },
    "DecisionTreeClassifier": {
        "estimator_class": DecisionTreeClassifier,
        "shap_explainer": "tree",
        "supports_sample_weight": True,
        "default_params": {"random_state": 42, "min_samples_leaf": 5},
    },
    "GradientBoostingClassifier": {
        "estimator_class": GradientBoostingClassifier,
        "shap_explainer": "tree",
        "supports_sample_weight": True,
        "default_params": {"random_state": 42, "min_samples_leaf": 5},
    },
    "LogisticRegression": {
        "estimator_class": LogisticRegression,
        "shap_explainer": "linear",
        "supports_sample_weight": True,
        "default_params": {"random_state": 42, "max_iter": 1000},
    },
}


def is_supported(algorithm_name: str) -> bool:
    return algorithm_name in SUPPORTED_MODELS


def get_model_info(algorithm_name: str) -> dict:
    return SUPPORTED_MODELS.get(algorithm_name, {})


def build_estimator(algorithm_name: str, random_state: int | None = None):
    """Instantiate a fresh (unfitted) estimator for the given algorithm
    name, used by training.trainer for Workflow A. Raises KeyError if
    the algorithm isn't supported -- callers should validate with
    is_supported() first to raise a proper ModelValidationError instead.

    random_state, if given, overrides the algorithm's default seed --
    needed to run the same algorithm across multiple seeds for
    stability experiments without editing SUPPORTED_MODELS itself."""
    info = SUPPORTED_MODELS[algorithm_name]
    estimator_class = info["estimator_class"]
    default_params = dict(info.get("default_params", {}))
    if random_state is not None and "random_state" in default_params:
        default_params["random_state"] = random_state
    return estimator_class(**default_params)
