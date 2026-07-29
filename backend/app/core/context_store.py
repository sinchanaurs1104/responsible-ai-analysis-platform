"""
In-memory store mapping run_id -> TrainedModelContext.

MVP-simple by design: the context holds a live fitted sklearn Pipeline,
which isn't something we want to serialize in/out of the DB on every
request. This keeps upload -> configure -> execute working within a
single server process.

Known limitation (acceptable for MVP, worth knowing before deploying):
this does NOT survive a server restart, and does NOT work correctly if
uvicorn is run with multiple worker processes (each worker would have
its own empty store). Run with a single worker for now
(`uvicorn main:app --workers 1`, which is also the default).
"""

from app.schemas.context import TrainedModelContext

_store: dict[str, TrainedModelContext] = {}


def set_context(run_id: str, context: TrainedModelContext) -> None:
    _store[run_id] = context


def get_context(run_id: str) -> TrainedModelContext | None:
    return _store.get(run_id)


def delete_context(run_id: str) -> None:
    _store.pop(run_id, None)
