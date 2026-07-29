"""
Central error hierarchy for the Responsible AI Platform.

Per the SDD (Error Handling Strategy), validation happens at fixed
checkpoints (ingestion, dataset compatibility, configuration) and raises
one of these typed errors. FastAPI exception handlers (added in the API
layer later) catch these and return structured JSON instead of raw
tracebacks or generic 500s.
"""


class ValidationError(Exception):
    """Base class for all validation errors raised anywhere in the pipeline."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ModelValidationError(ValidationError):
    """Raised when an uploaded model file is invalid, unfitted, unsupported,
    or otherwise cannot be safely used by the platform."""


class DatasetValidationError(ValidationError):
    """Raised when uploaded dataset(s) are malformed, empty, or incompatible
    with the model (schema mismatch, missing columns, etc.)."""


class ConfigValidationError(ValidationError):
    """Raised when pipeline configuration (protected attribute, target
    column, group definitions) is missing or invalid."""
