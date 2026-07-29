"""
Auto-builds preprocessing for Workflow A (Build & Analyze Model).

Because the platform trains the model itself in this workflow, it can
guarantee the resulting Pipeline is always fully self-contained
(preprocessing_status = "pipeline_managed") -- there is no user-supplied
preprocessing to trust or distrust, unlike the bare-estimator case in
ingestion.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app.core.exceptions import ConfigValidationError

HIGH_CARDINALITY_THRESHOLD = 50


def detect_high_cardinality_columns(
    feature_df: pd.DataFrame, threshold: int = HIGH_CARDINALITY_THRESHOLD
) -> list[str]:
    """Flags categorical columns with too many unique values to
    one-hot encode sensibly (e.g. an ID column). Does not flag numeric
    columns even if high-cardinality, since those are handled by scaling,
    not encoding."""
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    return [
        col for col in categorical_cols
        if feature_df[col].nunique(dropna=True) > threshold
    ]


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """
    Splits columns by dtype and builds a ColumnTransformer:
      - numeric columns -> median impute + standard scale
      - categorical columns -> most-frequent impute + one-hot encode

    Caller is responsible for excluding/rejecting high-cardinality
    categorical columns beforehand via detect_high_cardinality_columns().
    """
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ConfigValidationError(
            "No usable feature columns were found in the dataset after "
            "excluding the target column."
        )

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers)
