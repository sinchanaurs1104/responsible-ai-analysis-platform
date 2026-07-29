"""
Model evaluation: accuracy, precision, recall, F1, confusion matrix.

Operates purely on TrainedModelContext -- this module has no idea
whether the model was uploaded or trained internally, and no idea
whether it's a native Pipeline or a wrapped bare estimator. That
distinction was fully resolved by ingestion/training before this
module ever runs (SDD §7, §9).
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
)

from app.core.exceptions import DatasetValidationError
from app.modules.ingestion.model_loader import get_final_estimator
from app.schemas.context import TrainedModelContext
from app.schemas.metrics import PerformanceMetrics, ConfusionMatrix


def _resolve_positive_class(context: TrainedModelContext):
    """
    Follows sklearn's own convention for binary classifiers: the second
    entry in classes_ (i.e. classes_[-1] for a 2-class problem) is the
    one treated as "positive" by predict_proba's second column. Using
    the same convention here keeps metrics consistent with anything
    explainability/fairness modules compute later using predict_proba.
    """
    final_estimator = get_final_estimator(context.pipeline)
    classes = getattr(final_estimator, "classes_", None)
    if classes is None or len(classes) != 2:
        raise DatasetValidationError(
            "Model does not expose exactly two classes; binary "
            "classification is required for evaluation.",
            details={"classes_found": None if classes is None else list(classes)},
        )
    return classes[-1]


def evaluate_model(context: TrainedModelContext) -> PerformanceMetrics:
    if context.target_column not in context.test_df.columns:
        raise DatasetValidationError(
            f"Target column '{context.target_column}' not found in test dataset."
        )

    X_test = context.test_df.drop(columns=[context.target_column])
    y_test = context.test_df[context.target_column]

    positive_class = _resolve_positive_class(context)
    y_pred = context.pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0)

    final_estimator = get_final_estimator(context.pipeline)
    labels_order = list(final_estimator.classes_)  # [negative, positive]
    tn, fp, fn, tp = sk_confusion_matrix(y_test, y_pred, labels=labels_order).ravel()

    return PerformanceMetrics(
        accuracy=round(float(accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1_score=round(float(f1), 4),
        confusion_matrix=ConfusionMatrix(
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp),
        ),
        positive_class_label=positive_class if not hasattr(positive_class, "item") else positive_class.item(),
        test_set_size=len(y_test),
    )
