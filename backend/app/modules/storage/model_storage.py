"""
Local-disk artifact storage. Per SDD §18, this is the one file that
would need to change to swap in S3-compatible storage later -- nothing
else in the codebase should ever call joblib.dump/open() directly on a
model path.
"""

import os
from pathlib import Path

import joblib

ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", "./artifacts"))


def save_model(pipeline, version_id: str) -> str:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / f"{version_id}.pkl"
    joblib.dump(pipeline, path)
    return str(path)


def load_model(artifact_path: str):
    return joblib.load(artifact_path)


def save_json(data: dict, version_id: str, suffix: str = "model_card") -> str:
    import json

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / f"{version_id}_{suffix}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(path)
