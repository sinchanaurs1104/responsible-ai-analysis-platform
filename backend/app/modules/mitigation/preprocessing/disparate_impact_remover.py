"""
Disparate Impact Remover bias mitigation via AIF360.

Unlike Reweighing, this repairs feature values directly -- it produces
no sample weights. Only NUMERIC feature columns are repaired (DIR's
quantile-based repair is designed for continuous features); categorical
columns, the protected attribute, and the target column pass through
unchanged. transformed_train_df carries the repaired result;
sample_weights stays None.
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover as AIF360DIR

from app.modules.mitigation.base import PreprocessingStrategy, PreprocessingResult

REPAIR_LEVEL = 1.0  # full repair


class DisparateImpactRemoverStrategy(PreprocessingStrategy):
    name = "Disparate Impact Remover"

    def get_hyperparameters(self) -> dict:
        return {"repair_level": REPAIR_LEVEL}

    def apply(
        self,
        train_df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        privileged_value,
        unprivileged_value,
        positive_class,
    ) -> PreprocessingResult:
        feature_cols = [c for c in train_df.columns if c != target_column]
        numeric_feature_cols = [
            c for c in train_df[feature_cols].select_dtypes(include="number").columns
            if c != protected_attribute
        ]

        if not numeric_feature_cols:
            # Nothing DIR can repair (e.g. an all-categorical dataset) --
            # return the data unchanged rather than fail the pipeline.
            return PreprocessingResult(transformed_train_df=train_df, sample_weights=None)

        repair_input = train_df[numeric_feature_cols].copy()
        repair_input[protected_attribute] = train_df[protected_attribute].map(
            {privileged_value: 1, unprivileged_value: 0}
        )
        repair_input[target_column] = (train_df[target_column] == positive_class).astype(int)

        binary_label_dataset = BinaryLabelDataset(
            df=repair_input, label_names=[target_column],
            protected_attribute_names=[protected_attribute],
            favorable_label=1, unfavorable_label=0,
        )

        remover = AIF360DIR(repair_level=REPAIR_LEVEL, sensitive_attribute=protected_attribute)
        repaired_dataset = remover.fit_transform(binary_label_dataset)
        repaired_df, _ = repaired_dataset.convert_to_dataframe()

        transformed = train_df.copy()
        for col in numeric_feature_cols:
            transformed[col] = repaired_df[col].values

        return PreprocessingResult(
            transformed_train_df=transformed,
            sample_weights=None,
        )
