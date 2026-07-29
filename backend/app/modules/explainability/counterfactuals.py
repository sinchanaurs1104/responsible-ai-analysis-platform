"""
Counterfactual explanations via DiCE.

DiCE's sklearn backend calls the model's .predict/.predict_proba directly
on raw (untransformed) feature rows, so passing context.pipeline in
directly works whether it has 1 step or many -- DiCE never needs to know
about the preprocessing split that shap_explainer.py cares about. This
is the same "everything downstream just uses the Pipeline" principle
applying naturally here, without needing pipeline_utils at all.
"""

import threading
import warnings

import dice_ml

from app.core.exceptions import DatasetValidationError
from app.schemas.context import TrainedModelContext
from app.schemas.metrics import CounterfactualResult, CounterfactualExample

DEFAULT_NUM_INSTANCES = 3
DEFAULT_TOTAL_CFS = 3
PER_INSTANCE_TIMEOUT_SECONDS = 15
"""
DiCE's 'random' search loop is technically bounded (a fixed number of
feature-subset iterations), but was observed to stall for a long time
on certain instances against certain fitted pipelines -- e.g. after
Reweighing shifts a model's decision boundary, some query rows appear
to have no nearby flip under random sampling, and the repeated
predict_fn calls through a full ColumnTransformer pipeline compound the
slowdown. Rather than depend on understanding every internal cause, a
hard wall-clock timeout per instance guarantees this stage can never
stall the whole pipeline -- a stuck instance is skipped, same as an
instance DiCE outright fails on.
"""


def _generate_with_timeout(explainer, query_row, total_cfs, features_to_vary=None):
    """
    Runs the DiCE call in a daemon thread and joins with a timeout.

    Deliberately NOT implemented with ThreadPoolExecutor: its context
    manager calls shutdown(wait=True) on exit unconditionally, which
    blocks until the worker thread finishes regardless of any explicit
    shutdown(wait=False) called inside -- this was tested directly and
    confirmed to silently defeat the timeout (a 30s artificial hang
    still took the full 30s to return instead of stopping at 15s).
    A plain daemon Thread + join(timeout=...) has no such pitfall:
    join() genuinely returns after the timeout, and daemon=True
    guarantees a still-running thread can never block process exit.
    """
    result: dict = {}

    def target():
        try:
            kwargs = dict(total_CFs=total_cfs, desired_class="opposite")
            if features_to_vary is not None:
                kwargs["features_to_vary"] = features_to_vary
            result["value"] = explainer.generate_counterfactuals(query_row, **kwargs)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=PER_INSTANCE_TIMEOUT_SECONDS)

    if thread.is_alive():
        # Still running -- we can't force-kill a Python thread, but
        # daemon=True means it will never block the process from
        # continuing or exiting. We simply stop waiting on it here.
        raise TimeoutError(
            f"DiCE counterfactual generation exceeded "
            f"{PER_INSTANCE_TIMEOUT_SECONDS}s for this instance."
        )

    if "error" in result:
        raise result["error"]
    return result["value"]


def generate_counterfactuals(
    context: TrainedModelContext,
    protected_attribute: str | None = None,
    num_instances: int = DEFAULT_NUM_INSTANCES,
    total_cfs: int = DEFAULT_TOTAL_CFS,
) -> CounterfactualResult:
    feature_cols = [c for c in context.train_df.columns if c != context.target_column]
    numeric_cols = context.train_df[feature_cols].select_dtypes(include="number").columns.tolist()

    # Protected attributes must never be suggested as a change in a
    # counterfactual -- DiCE's features_to_vary restricts the search to
    # every feature except the protected one(s), keeping it immutable.
    if protected_attribute is not None and protected_attribute in feature_cols:
        features_to_vary = [c for c in feature_cols if c != protected_attribute]
    else:
        features_to_vary = None

    data_interface = dice_ml.Data(
        dataframe=context.train_df,
        continuous_features=numeric_cols,
        outcome_name=context.target_column,
    )
    model_interface = dice_ml.Model(model=context.pipeline, backend="sklearn")
    explainer = dice_ml.Dice(data_interface, model_interface, method="random")

    X_test = context.test_df.drop(columns=[context.target_column])
    if len(X_test) == 0:
        raise DatasetValidationError("Test dataset has no rows to explain.")

    sample_size = min(num_instances, len(X_test))
    query_instances = X_test.sample(n=sample_size, random_state=42)

    examples: list[CounterfactualExample] = []

    for idx in range(sample_size):
        query_row = query_instances.iloc[[idx]]
        original_prediction = context.pipeline.predict(query_row)[0]
        original_prediction = (
            original_prediction.item() if hasattr(original_prediction, "item") else original_prediction
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf_result = _generate_with_timeout(
                    explainer, query_row, total_cfs, features_to_vary=features_to_vary
                )
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
        except Exception:
            # Covers both DiCE's own failures (e.g. no valid counterfactual
            # found for this instance) and our timeout above -- either way,
            # skip this instance rather than failing the whole request.
            continue

        if cf_df is None or cf_df.empty:
            continue

        # DiCE's "random" search method does not guarantee every row in
        # cf_df actually flips the predicted class within its search
        # budget -- filter down to only the rows that genuinely do,
        # rather than trusting the label DiCE attached.
        actual_preds = context.pipeline.predict(
            cf_df.drop(columns=[context.target_column])
        )
        flipped_mask = actual_preds != original_prediction
        cf_df_flipped = cf_df[flipped_mask]

        if cf_df_flipped.empty:
            # DiCE failed to find a genuinely flipping counterfactual for
            # this instance within budget -- skip rather than show a
            # misleading non-counterfactual.
            continue

        flipped_preds = actual_preds[flipped_mask]
        counterfactual_prediction = flipped_preds[0]
        counterfactual_prediction = (
            counterfactual_prediction.item()
            if hasattr(counterfactual_prediction, "item")
            else counterfactual_prediction
        )

        examples.append(
            CounterfactualExample(
                original_instance=query_row.iloc[0].to_dict(),
                original_prediction=original_prediction,
                counterfactual_instances=cf_df_flipped.drop(
                    columns=[context.target_column]
                ).to_dict(orient="records"),
                counterfactual_prediction=counterfactual_prediction,
            )
        )

    return CounterfactualResult(examples=examples, method="random")