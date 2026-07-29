"""
Creates and persists ModelVersion rows. This is the only module that
writes to the model_versions table -- evaluation/fairness/mitigation
modules only produce structured results, they never touch the DB.
"""

import sklearn
import aif360
import shap

from sqlalchemy.orm import Session

from app.db import repository as repo
from app.db.models import ModelVersion
from app.modules.ingestion.model_loader import get_final_estimator
from app.modules.storage import model_storage
from app.schemas.context import TrainedModelContext
from app.schemas.metrics import PerformanceMetrics, SHAPResult, ErrorAnalysisResult, CounterfactualResult
from app.schemas.fairness import FairnessMetrics, FairnessFinding

LIBRARY_VERSIONS = {
    "scikit-learn": sklearn.__version__,
    "aif360": aif360.__version__,
    "shap": shap.__version__,
}


def create_version(
    session: Session,
    run_id: str,
    context: TrainedModelContext,
    performance_metrics: PerformanceMetrics,
    version_number: int,
    fairness_metrics: FairnessMetrics | None = None,
    fairness_finding: FairnessFinding | None = None,
    shap_result: SHAPResult | None = None,
    error_analysis_result: ErrorAnalysisResult | None = None,
    counterfactual_result: CounterfactualResult | None = None,
    mitigation_method: str | None = None,
    mitigation_category: str | None = None,
    mitigation_hyperparameters: dict | None = None,
    runtime_seconds: float | None = None,
    random_seed: int | None = None,
    parent_version_id: str | None = None,
) -> ModelVersion:
    final_estimator = get_final_estimator(context.pipeline)
    hyperparameters = {
        k: v for k, v in final_estimator.get_params().items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    version = repo.create_model_version(
        session,
        run_id=run_id,
        parent_version_id=parent_version_id,
        version_number=version_number,
        source=context.source,
        mitigation_method=mitigation_method,
        mitigation_category=mitigation_category,
        mitigation_hyperparameters=mitigation_hyperparameters or {},
        runtime_seconds=runtime_seconds,
        random_seed=random_seed,
        algorithm_name=context.algorithm_name,
        preprocessing_status=context.preprocessing_status,
        hyperparameters=hyperparameters,
        library_versions=LIBRARY_VERSIONS,
        performance_metrics=performance_metrics.model_dump(),
        fairness_metrics=fairness_metrics.model_dump() if fairness_metrics else {},
        fairness_finding=fairness_finding.model_dump() if fairness_finding else {},
        explainability_results=shap_result.model_dump() if shap_result else {},
        error_analysis_results=error_analysis_result.model_dump() if error_analysis_result else {},
        counterfactual_results=counterfactual_result.model_dump() if counterfactual_result else {},
    )

    artifact_path = model_storage.save_model(context.pipeline, version.version_id)
    version.artifact_path = artifact_path
    session.commit()
    session.refresh(version)

    return version
