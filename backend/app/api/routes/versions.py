"""
/versions routes: download endpoints. Starts with just the model file
download (simplest of the three) -- model-card and report downloads
are added in follow-up steps.
"""

import os

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import repository as repo
from app.core.exceptions import DatasetValidationError
from app.modules.versioning.model_card import build_model_card

router = APIRouter(prefix="/versions", tags=["versions"])


@router.get("/{version_id}/download")
def download_model(version_id: str, db: Session = Depends(get_db)):
    """
    Streams the saved model artifact for this version. Works for both
    the original (V1) and debiased (V2) versions -- whichever
    version_id is requested. The filename download hint uses the
    mitigation method so a user pulling V2 gets a clearly-labeled file
    (e.g. debiased_model_Reweighing.pkl) rather than a bare UUID.
    """
    version = repo.get_model_version(db, version_id)
    if version is None:
        raise DatasetValidationError(f"Version '{version_id}' not found.")

    if not version.artifact_path or not os.path.exists(version.artifact_path):
        raise DatasetValidationError(
            f"Model artifact for version '{version_id}' was not found on disk.",
            details={"artifact_path": version.artifact_path},
        )

    label = version.mitigation_method or "original"
    filename = f"model_v{version.version_number}_{label}.pkl"

    return FileResponse(
        path=version.artifact_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/{version_id}/model-card")
def get_model_card(version_id: str, db: Session = Depends(get_db)):
    """
    Returns the model card as JSON -- the minimal information a user
    needs to understand how this specific model was produced and how
    to call it correctly (SDD §17). Meant to accompany the .pkl download.
    """
    version = repo.get_model_version(db, version_id)
    if version is None:
        raise DatasetValidationError(f"Version '{version_id}' not found.")

    run = repo.get_run(db, version.run_id)
    if run is None:
        raise DatasetValidationError(f"Parent run for version '{version_id}' not found.")

    return build_model_card(version, run)
