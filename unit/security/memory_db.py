"""A throwaway in-memory SQLite behind the backend's own session factory.

`memory_database(owui_module, *tables)` points every model call that opens its own session at a
fresh in-memory database holding just those ORM tables, so a test drives the real model code
and never touches the scratch database under DATA_DIR.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


@asynccontextmanager
async def memory_database(owui_module, *tables) -> AsyncIterator[async_sessionmaker]:
    """Yield the session factory every model now uses; `tables` are ORM classes to create."""
    db_module = owui_module("open_webui.internal.db")
    # one shared connection, or each session would open a database of its own
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as connection:
            for table in tables:
                await connection.run_sync(table.__table__.create)
        sessions = async_sessionmaker(
            bind=engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
        )
        with patch.object(db_module, "AsyncSessionLocal", sessions):
            yield sessions
    finally:
        await engine.dispose()
