"""
Subgroup error analysis: finds feature-defined subgroups (categorical
columns as-is, numeric columns binned into quartiles) where accuracy is
meaningfully worse than the overall test-set accuracy.

This is the lighter, pandas-only alternative to Microsoft's
`erroranalysis` package flagged as an option in the design discussion --
same analytical goal (find where the model underperforms), no extra
dependency. Operates on RAW (untransformed) test data so subgroup labels
stay human-readable (e.g. "region = south", not a one-hot column name).
"""

import pandas as pd

from app.schemas.context import TrainedModelContext
from app.schemas.metrics import ErrorAnalysisResult, SubgroupError

MIN_SUBGROUP_SIZE = 5
ACCURACY_GAP_THRESHOLD = 0.10  # flag subgroups at least 10 points below overall
TOP_N_WORST = 5
NUMERIC_BINS = 4


def _bin_numeric_column(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series, q=NUMERIC_BINS, duplicates="drop").astype(str)
    except ValueError:
        # Not enough distinct values to bin meaningfully -- treat as-is.
        return series.astype(str)


def analyze_errors(context: TrainedModelContext) -> ErrorAnalysisResult:
    X_test = context.test_df.drop(columns=[context.target_column])
    y_test = context.test_df[context.target_column]
    y_pred = context.pipeline.predict(X_test)

    correct = (y_test.values == y_pred)
    overall_accuracy = round(float(correct.mean()), 4)

    subgroup_results: list[SubgroupError] = []

    for col in X_test.columns:
        series = X_test[col]
        if pd.api.types.is_numeric_dtype(series):
            grouped_labels = _bin_numeric_column(series)
        else:
            grouped_labels = series.astype(str)

        temp = pd.DataFrame({"group": grouped_labels, "correct": correct})
        for group_value, group_df in temp.groupby("group", observed=True):
            size = len(group_df)
            if size < MIN_SUBGROUP_SIZE:
                continue
            subgroup_accuracy = round(float(group_df["correct"].mean()), 4)
            gap = round(overall_accuracy - subgroup_accuracy, 4)
            if gap >= ACCURACY_GAP_THRESHOLD:
                subgroup_results.append(
                    SubgroupError(
                        column=col,
                        subgroup=str(group_value),
                        subgroup_size=size,
                        subgroup_accuracy=subgroup_accuracy,
                        overall_accuracy=overall_accuracy,
                        accuracy_gap=gap,
                    )
                )

    subgroup_results.sort(key=lambda s: s.accuracy_gap, reverse=True)

    return ErrorAnalysisResult(
        overall_accuracy=overall_accuracy,
        worst_subgroups=subgroup_results[:TOP_N_WORST],
    )
