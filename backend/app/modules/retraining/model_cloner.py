"""
Retrains a fresh clone of the original Pipeline using the output of a
PreprocessingStrategy.

Deliberately mitigation-agnostic: this module has no idea whether the
PreprocessingResult came from Reweighing (weights only, data unchanged)
or a future method like Disparate Impact Remover (transformed features,
no weights) -- it only knows how to clone a Pipeline and refit it on
whatever training data and weights (if any) it's given, targeting
sample_weight at the correct step (context.estimator_step_name) only
when weights are actually present.
"""

from sklearn.base import clone

from app.core.exceptions import ModelValidationError
from app.schemas.context import TrainedModelContext
from app.modules.mitigation.base import PreprocessingResult


def retrain_with_preprocessing_result(
    context: TrainedModelContext, result: PreprocessingResult
) -> TrainedModelContext:
    train_df = result.transformed_train_df

    if result.sample_weights is not None and len(result.sample_weights) != len(train_df):
        raise ModelValidationError(
            "Sample weights length does not match training data length.",
            details={
                "weights_length": len(result.sample_weights),
                "train_df_length": len(train_df),
            },
        )

    cloned_pipeline = clone(context.pipeline)

    X_train = train_df.drop(columns=[context.target_column])
    y_train = train_df[context.target_column]

    fit_params = {}
    if result.sample_weights is not None:
        fit_params[f"{context.estimator_step_name}__sample_weight"] = result.sample_weights

    try:
        cloned_pipeline.fit(X_train, y_train, **fit_params)
    except TypeError as exc:
        raise ModelValidationError(
            f"The underlying estimator ('{context.algorithm_name}') does "
            f"not accept sample_weight during fit; it cannot be retrained "
            f"with mitigation weights.",
            details={"underlying_error": str(exc)},
        ) from exc

    return TrainedModelContext(
        pipeline=cloned_pipeline,
        source=context.source,
        preprocessing_status=context.preprocessing_status,
        estimator_step_name=context.estimator_step_name,
        algorithm_name=context.algorithm_name,
        train_df=train_df,
        test_df=context.test_df,
        target_column=context.target_column,
        validation_warnings=context.validation_warnings,
    )
