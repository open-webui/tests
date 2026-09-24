"""Regression: the built-in text searches run beside the event loop, not on it.

open-webui 0.11.4 fix `82f11b14c` (#29621) plus the knowledge_fs half in `d9c8de9c`:
`grep_chat_files` and `grep_knowledge_files` ran `_grep_file_models` directly, and
`kb_exec`'s grep built its matcher and scanned every line inline, so a model-issued search
over a large file held every other request on the worker for as long as it took. All three
now hand the matching to `asyncio.to_thread`.

Stays a unit test: the symptom is event-loop latency inside one worker, which an HTTP client
cannot separate from ordinary request time. Each test runs the public tool against a real
file row in an in-memory database while a ticker coroutine measures the longest stall of the
loop.

Discriminates: passes on dev bbfa876af, fails with the `asyncio.to_thread` calls replaced by
direct calls (the loop stalls for the whole matching pass).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]

USER = {"id": "searcher", "role": "user"}
REQUEST = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
FILE_ID = "f-1"
# A regex (the `|`) so each line costs an RE2 search: a few hundred ms in total.
PATTERN = "needle|pin"
FILLER_LINES = 200_000


@pytest_asyncio.fixture
async def large_file(owui_module):
    """One large file owned by USER, served from an in-memory database."""
    db_module = owui_module("open_webui.internal.db")
    files = owui_module("open_webui.models.files")
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(db_module.Base.metadata.create_all)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def in_memory_session():
        async with sessions() as session:
            yield session

    content = "\n".join([f"filler line {index}" for index in range(FILLER_LINES)] + ["the needle"])
    with patch.object(db_module, "get_async_db", in_memory_session):
        await files.Files.insert_new_file(
            USER["id"],
            files.FileForm(id=FILE_ID, filename="f-1.md", path="", data={"content": content}),
        )
        yield f"{FILE_ID}  f-1.md:{FILLER_LINES + 1}: the needle"
    await engine.dispose()


async def _longest_stall(search) -> tuple[str, float, float]:
    """Await `search` while a ticker measures the longest gap between its ticks."""
    gaps: list[float] = []
    searching = True

    async def tick() -> None:
        last = time.perf_counter()
        while searching:
            await asyncio.sleep(0.002)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticker = asyncio.create_task(tick())
    started = time.perf_counter()
    result = await search
    elapsed = time.perf_counter() - started
    searching = False
    await ticker
    return result, max(gaps), elapsed


def _assert_loop_stayed_free(tool: str, stall: float, elapsed: float) -> None:
    assert stall < elapsed / 3, (
        f"{tool} stalled the event loop for {stall * 1000:.0f} ms of a "
        f"{elapsed * 1000:.0f} ms search, so every other request on the worker waited for "
        "the matching pass (#29621)"
    )


async def test_grep_chat_files_leaves_the_loop_free(owui_module, large_file):
    builtin = owui_module("open_webui.tools.builtin")

    result, stall, elapsed = await _longest_stall(
        builtin.grep_chat_files(
            pattern=PATTERN,
            __request__=REQUEST,
            __user__=USER,
            __files__=[{"type": "file", "id": FILE_ID}],
        )
    )

    assert result == large_file
    _assert_loop_stayed_free("grep_chat_files", stall, elapsed)


async def test_grep_knowledge_files_leaves_the_loop_free(owui_module, large_file):
    builtin = owui_module("open_webui.tools.builtin")

    result, stall, elapsed = await _longest_stall(
        builtin.grep_knowledge_files(
            pattern=PATTERN,
            __request__=REQUEST,
            __user__=USER,
            __model_knowledge__=[{"type": "file", "id": FILE_ID}],
        )
    )

    assert result == large_file
    _assert_loop_stayed_free("grep_knowledge_files", stall, elapsed)


async def test_kb_exec_grep_leaves_the_loop_free(owui_module, large_file):
    knowledge_fs = owui_module("open_webui.tools.knowledge_fs")

    result, stall, elapsed = await _longest_stall(
        knowledge_fs.kb_exec(
            f'grep -E "{PATTERN}"',
            __user__=USER,
            __model_knowledge__=[{"type": "file", "id": FILE_ID}],
        )
    )

    assert "the needle" in result, result
    _assert_loop_stayed_free("kb_exec grep", stall, elapsed)
