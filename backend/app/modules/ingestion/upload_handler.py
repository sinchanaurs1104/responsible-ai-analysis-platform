"""
Entry point for Workflow B (Analyze Existing Model).

Orchestrates: safe model loading -> fitted check -> algorithm support
check -> Pipeline/bare-estimator detection & wrapping -> dataset loading
-> schema/preprocessing validation -> TrainedModelContext.

This is the only function the API layer should call for this workflow;
everything else in ingestion/ is an internal building block.
"""

from app.core.exceptions import DatasetValidationError
from app.schemas.context import TrainedModelContext
from app.modules.ingestion import model_loader, model_validators, dataset_loader


def handle_existing_model_upload(
    model_file_path: str,
    train_csv_path: str,
    test_csv_path: str,
    target_column: str,
) -> TrainedModelContext:
    # 1. Load and validate the model file itself.
    raw_model = model_loader.load_model_file(model_file_path)
    model_validators.validate_is_fitted(raw_model)

    # 2. Normalize to a Pipeline (wraps bare estimators automatically).
    pipeline, estimator_step_name, preprocessing_status = model_loader.wrap_if_needed(
        raw_model
    )

    # 3. Confirm the underlying algorithm is one the platform supports.
    algorithm_name = model_validators.validate_algorithm_supported(pipeline)

    # 4. Load datasets.
    train_df = dataset_loader.load_csv(train_csv_path, name="Training dataset")
    test_df = dataset_loader.load_csv(test_csv_path, name="Testing dataset")

    # 5. Cross-dataset and target-column checks.
    model_validators.validate_schema_match(train_df, test_df)
    if target_column not in train_df.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' not found in training dataset.",
            details={"available_columns": train_df.columns.tolist()},
        )

    # 6. Preprocessing sanity checks (only bites for bare-estimator case).
    # Checked against the feature columns only (target excluded), since the
    # target column is not part of the model's input.
    feature_df = train_df.drop(columns=[target_column])
    warnings = model_validators.validate_preprocessing_assumptions(
        feature_df, pipeline, preprocessing_status
    )

    return TrainedModelContext(
        pipeline=pipeline,
        source="uploaded",
        preprocessing_status=preprocessing_status,
        estimator_step_name=estimator_step_name,
        algorithm_name=algorithm_name,
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        validation_warnings=warnings,
    )
