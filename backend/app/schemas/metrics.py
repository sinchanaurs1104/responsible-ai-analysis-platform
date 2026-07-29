"""
Structured output schemas for metrics-producing modules (evaluation,
fairness). Kept separate from schemas/context.py since these are
*outputs* of analysis, not the *input* convergence object.
"""

from pydantic import BaseModel


class ConfusionMatrix(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrix
    positive_class_label: int | str
    test_set_size: int


class FeatureImportance(BaseModel):
    feature_name: str
    importance_score: float


class SHAPResult(BaseModel):
    top_features: list[FeatureImportance]
    explained_sample_size: int
    explainer_type: str  # "tree" | "linear"


class SubgroupError(BaseModel):
    column: str
    subgroup: str
    subgroup_size: int
    subgroup_accuracy: float
    overall_accuracy: float
    accuracy_gap: float  # overall - subgroup, positive means subgroup underperforms


class ErrorAnalysisResult(BaseModel):
    overall_accuracy: float
    worst_subgroups: list[SubgroupError]


class CounterfactualExample(BaseModel):
    original_instance: dict
    original_prediction: int | str
    counterfactual_instances: list[dict]
    counterfactual_prediction: int | str


class CounterfactualResult(BaseModel):
    examples: list[CounterfactualExample]
    method: str = "random"


