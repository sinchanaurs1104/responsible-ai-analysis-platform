"""
FastAPI application entry point.

This file stays minimal on purpose: route registration happens here,
but route logic lives entirely in app/api/routes/. Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.exceptions import ValidationError
from app.db.session import init_db
from app.api.routes import runs, versions, experiments

app = FastAPI(
    title="Responsible AI Platform",
    version="0.1.0",
)

# MVP-simple CORS: allow the local frontend dev server. Tighten this
# before any real deployment.
app.add_middleware(
    CORSMiddleware,
    # Matches any port on localhost/127.0.0.1 -- Vite silently increments
    # its port (5173, 5174, ...) whenever the default is already taken by
    # another running dev server instance, so a fixed port list keeps
    # breaking in exactly this way during development. Tighten this to
    # an explicit origin before any real deployment.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
def validation_error_handler(request, exc: ValidationError):
    """
    Per SDD §14: validation errors return structured JSON
    ({error_type, message, details}), never a raw traceback or generic
    500. This is the single place that translates our internal
    exception hierarchy into HTTP responses.
    """
    return JSONResponse(status_code=400, content=exc.to_dict())


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(runs.router)
app.include_router(versions.router)
app.include_router(experiments.router)
