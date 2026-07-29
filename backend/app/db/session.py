"""
Database engine and session setup. SQLite file on local disk for the
MVP (SDD: no enterprise infra) -- swappable later via DATABASE_URL alone.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.db.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./responsible_ai_platform.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


def get_db():
    """FastAPI dependency: yields a session, closes it after the request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
