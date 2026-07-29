from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.modules.experiments.experiment_log import export_csv

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/export")
def export_experiment_log(db: Session = Depends(get_db)):
    csv_text = export_csv(db)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiment_log.csv"},
    )
