"""
Report artifact path helper. Mirrors model_storage.py's pattern
(local disk, path returned/stored, swappable later per SDD §18) but for
the PDF report rather than model files.
"""

from pathlib import Path

from app.modules.storage.model_storage import ARTIFACT_ROOT


def get_report_path(run_id: str) -> str:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return str(ARTIFACT_ROOT / f"{run_id}_Responsible_AI_Report.pdf")
