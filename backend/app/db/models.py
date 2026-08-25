"""
SQLAlchemy models: Run and ModelVersion (SDD §11, Database/Schema).

Binary artifacts (models, reports) are never stored here -- only
filesystem paths (artifact_path). Structured results (performance
metrics, fairness metrics, fairness finding) are stored as JSON columns,
since they're already Pydantic schemas from evaluation/fairness -- no
separate relational shape needed for them at MVP scale.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    workflow_type: Mapped[str] = mapped_column(String)  # "build_and_analyze" | "analyze_existing"
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|completed|failed
    current_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    protected_attribute: Mapped[str | None] = mapped_column(String, nullable=True)
    privileged_value: Mapped[str | None] = mapped_column(String, nullable=True)
    unprivileged_value: Mapped[str | None] = mapped_column(String, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_name: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    """SHA-256 of the raw uploaded dataset file bytes -- lets a later
    reader confirm exactly which version of a dataset file (e.g. if
    adult.csv is edited/replaced later) produced this run's results."""

    train_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_privileged_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_unprivileged_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """Group counts in the actual test split used for this run --
    split/seed-dependent, so only recoverable later by rerunning with
    the identical seed and environment."""

    failed_methods: Mapped[list] = mapped_column(JSON, default=list)
    """List of {"method": str, "error": str} for every mitigation method
    that raised during this run and was skipped, so failures survive in
    the experiment record instead of only the single most recent one
    overwriting error_message."""

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String, nullable=True)

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"))
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.version_id"), nullable=True
    )
    version_number: Mapped[int] = mapped_column(Integer)

    source: Mapped[str] = mapped_column(String)  # "uploaded" | "internally_trained"
    mitigation_method: Mapped[str | None] = mapped_column(String, nullable=True)
    mitigation_category: Mapped[str | None] = mapped_column(String, nullable=True)  # pre | in | post
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algorithm_name: Mapped[str] = mapped_column(String)
    preprocessing_status: Mapped[str] = mapped_column(String)

    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    mitigation_hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    library_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """Deprecated: kept for backward compatibility with rows created
    before this split. New rows populate mitigation_seconds +
    analysis_seconds instead, since this combined figure conflates the
    mitigation-specific cost with the constant SHAP/DiCE analysis cost,
    making cross-method runtime comparisons misleading."""
    mitigation_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """Time for the mitigation-specific step only (transform+retrain
    for pre-processing, wrap for post-processing). None for V1 (no
    mitigation applied)."""
    analysis_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """Time for the shared analysis pass (SHAP + error analysis +
    counterfactuals + fairness metrics) -- roughly constant regardless
    of mitigation method, so kept separate from mitigation_seconds for
    fair method-to-method runtime comparisons."""

    performance_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    fairness_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    fairness_finding: Mapped[dict] = mapped_column(JSON, default=dict)
    explainability_results: Mapped[dict] = mapped_column(JSON, default=dict)
    error_analysis_results: Mapped[dict] = mapped_column(JSON, default=dict)
    counterfactual_results: Mapped[dict] = mapped_column(JSON, default=dict)
    narrative_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    run: Mapped["Run"] = relationship(back_populates="versions")
