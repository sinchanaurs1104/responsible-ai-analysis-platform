"""
Repository layer -- the only place in the codebase that issues
SQLAlchemy queries. versioning/version_manager.py (next module) and the
future API routes both go through this, never through db.models
directly.
"""

from sqlalchemy.orm import Session

from app.db.models import Run, ModelVersion


def create_run(session: Session, **kwargs) -> Run:
    run = Run(**kwargs)
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)


def update_run_status(
    session: Session, run_id: str, status: str, current_stage: str | None = None,
    error_message: str | None = None,
) -> Run | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    run.status = status
    if current_stage is not None:
        run.current_stage = current_stage
    if error_message is not None:
        run.error_message = error_message
    session.commit()
    session.refresh(run)
    return run


def create_model_version(session: Session, **kwargs) -> ModelVersion:
    version = ModelVersion(**kwargs)
    session.add(version)
    session.commit()
    session.refresh(version)
    return version


def get_model_version(session: Session, version_id: str) -> ModelVersion | None:
    return session.get(ModelVersion, version_id)


def list_versions_for_run(session: Session, run_id: str) -> list[ModelVersion]:
    run = session.get(Run, run_id)
    return list(run.versions) if run else []


def get_original_version_for_run(session: Session, run_id: str) -> ModelVersion | None:
    """The one version with mitigation_method=None for this run (V1) --
    used to split V1 evaluation from mitigation into two separate calls
    without recomputing or duplicating V1."""
    for v in list_versions_for_run(session, run_id):
        if v.mitigation_method is None:
            return v
    return None


def record_run_sample_stats(
    session: Session, run_id: str, train_row_count: int, test_row_count: int,
    test_privileged_count: int, test_unprivileged_count: int,
) -> Run | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    run.train_row_count = train_row_count
    run.test_row_count = test_row_count
    run.test_privileged_count = test_privileged_count
    run.test_unprivileged_count = test_unprivileged_count
    session.commit()
    session.refresh(run)
    return run


def append_run_failure(session: Session, run_id: str, method: str, error: str) -> Run | None:
    """Appends to Run.failed_methods rather than overwriting -- multiple
    methods can fail within the same run and each must survive in the
    record, not just the most recent one."""
    run = session.get(Run, run_id)
    if run is None:
        return None
    failures = list(run.failed_methods or [])
    failures.append({"method": method, "error": error})
    run.failed_methods = failures
    session.commit()
    session.refresh(run)
    return run


def set_report_path(session: Session, run_id: str, report_path: str) -> Run | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    run.report_path = report_path
    session.commit()
    session.refresh(run)
    return run
