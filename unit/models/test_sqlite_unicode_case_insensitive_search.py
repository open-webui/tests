"""Regression: SQLite search and tag filtering only folded case for unaccented
English letters (commit 26f37426b, shipped in v0.11.2).

SQLite's built-in LIKE and LOWER() are ASCII-only, and SQLAlchemy compiles
`ilike` on SQLite to `lower(x) LIKE lower(?)`. A chat titled "CAFÉ" or "ПРИВЕТ"
was therefore invisible to a lowercase search, and a prompt tagged "CAFÉ" was
invisible to the tag filter. The fix registers a Python `like` implementation on
every SQLite connection in `_apply_sqlite_pragmas`, which lowercases both sides
with `str.lower()`, and routes the prompt tag clause through LIKE ... ESCAPE so
it picks up the same folding.

Discriminates: passes on v0.11.2 and v0.11.3, fails on v0.11.1 (no `like`
override, so non-ASCII rows never match a lowercase query, and the prompt tag
clause still compares SQLite's ASCII-only LOWER()).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.regression

OWNER = "alice"

# Each case is (stored text, lowercase query that must find it).
UNICODE_CASES = [
    pytest.param("CAFÉ Süd", "café süd", id="latin-accented"),
    pytest.param("ПРИВЕТ Мир", "привет мир", id="cyrillic"),
    pytest.param("ΑΘΗΝΑ", "αθηνα", id="greek"),
]


@pytest.fixture(scope="module")
def internal_db(owui_module):
    return owui_module("open_webui.internal.db")


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def prompt_models(owui_module):
    return owui_module("open_webui.models.prompts")


@pytest_asyncio.fixture
async def db(internal_db, chat_models, prompt_models, owui_module, tmp_path):
    """A private SQLite database wired up the way the app wires its own.

    The shipped `_apply_sqlite_pragmas` runs on every connection, so the model
    layer executes its real SQL against the real connection setup.
    """
    users = owui_module("open_webui.models.users")
    access_grants = owui_module("open_webui.models.access_grants")
    db_path = tmp_path / "search.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    for table in (
        chat_models.Chat.__table__,
        prompt_models.Prompt.__table__,
        users.User.__table__,
        access_grants.AccessGrant.__table__,
    ):
        table.create(sync_engine)
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    @event.listens_for(async_engine.sync_engine, "connect")
    def apply_pragmas(dbapi_connection, connection_record):
        internal_db._apply_sqlite_pragmas(dbapi_connection)

    session_maker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    with patch.object(internal_db, "AsyncSessionLocal", session_maker):
        yield session_maker
    await async_engine.dispose()


async def add_chat(chat_models, session_maker, chat_id: str, title: str) -> None:
    async with session_maker() as session:
        session.add(
            chat_models.Chat(
                id=chat_id,
                user_id=OWNER,
                title=title,
                chat={"title": title, "history": {"messages": {}}},
                created_at=1,
                updated_at=1,
                meta={},
            )
        )
        await session.commit()


async def add_prompt(
    prompt_models, session_maker, prompt_id: str, name: str, tags: list[str] | None = None
) -> None:
    now = int(time.time())
    async with session_maker() as session:
        session.add(
            prompt_models.Prompt(
                id=prompt_id,
                command=f"/{prompt_id}",
                user_id=OWNER,
                name=name,
                content=name,
                tags=tags or [],
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def search_chat_titles(chat_models, query: str) -> list[str]:
    chats = await chat_models.Chats.get_chat_list_by_user_id(OWNER, filter={"query": query})
    return [chat.id for chat in chats]


async def search_prompt_names(prompt_models, query: str) -> list[str]:
    result = await prompt_models.Prompts.search_prompts(OWNER, filter={"query": query})
    return [prompt.id for prompt in result.items]


async def search_prompt_tag(prompt_models, tag: str) -> list[str]:
    result = await prompt_models.Prompts.search_prompts(OWNER, filter={"tag": tag})
    return [prompt.id for prompt in result.items]


# ---------------------------------------------------------------------------
# Narrow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored", "query"), UNICODE_CASES)
async def test_chat_title_search_folds_non_ascii_case(chat_models, db, stored, query):
    """NARROW: SQLite's own lower() leaves these characters untouched."""
    await add_chat(chat_models, db, "target", stored)

    assert await search_chat_titles(chat_models, query) == ["target"]


@pytest.mark.asyncio
async def test_prompt_tag_filter_folds_non_ascii_case(prompt_models, db):
    """NARROW: the tag clause compared SQLite LOWER() against a Python-lowered tag."""
    await add_prompt(prompt_models, db, "tagged", "Coffee notes", tags=["CAFÉ"])

    assert await search_prompt_tag(prompt_models, "café") == ["tagged"]


# ---------------------------------------------------------------------------
# Broad
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored", "query"), UNICODE_CASES)
async def test_prompt_search_and_tag_filter_agree_on_case(prompt_models, db, stored, query):
    """BROAD: one collation covers every LIKE-backed lookup, search and tags alike."""
    await add_prompt(prompt_models, db, "target", stored, tags=[stored])

    assert await search_prompt_names(prompt_models, query) == ["target"]
    assert await search_prompt_tag(prompt_models, query) == ["target"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored", "query"), UNICODE_CASES)
async def test_search_folds_in_both_directions(chat_models, db, stored, query):
    """BROAD: an uppercase query must find lowercase rows too."""
    await add_chat(chat_models, db, "target", query)

    assert await search_chat_titles(chat_models, stored.upper()) == ["target"]


# ---------------------------------------------------------------------------
# Nearby
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ascii_chat_title_search_still_matches(chat_models, db):
    await add_chat(chat_models, db, "ascii", "Weekly REPORT")

    assert await search_chat_titles(chat_models, "weekly report") == ["ascii"]


@pytest.mark.asyncio
async def test_ascii_prompt_search_and_tag_filter_still_match(prompt_models, db):
    await add_prompt(prompt_models, db, "ascii", "Weekly REPORT", tags=["Weekly"])

    assert await search_prompt_names(prompt_models, "weekly") == ["ascii"]
    assert await search_prompt_tag(prompt_models, "weekly") == ["ascii"]


@pytest.mark.asyncio
async def test_tag_filter_treats_wildcards_literally(prompt_models, db):
    """NEARBY: routing tags through LIKE must not turn tag text into a pattern."""
    await add_prompt(prompt_models, db, "percent", "Discount", tags=["50% OFF"])
    await add_prompt(prompt_models, db, "underscore", "Snake", tags=["A_B"])

    assert await search_prompt_tag(prompt_models, "50% off") == ["percent"]
    assert await search_prompt_tag(prompt_models, "5%") == []
    assert await search_prompt_tag(prompt_models, "a_b") == ["underscore"]
    assert await search_prompt_tag(prompt_models, "axb") == []


@pytest.mark.asyncio
async def test_search_does_not_strip_accents(chat_models, db):
    """NEARBY: case folding only. "cafe" is still a different word from "café"."""
    await add_chat(chat_models, db, "accented", "CAFÉ")

    assert await search_chat_titles(chat_models, "cafe") == []


@pytest.mark.asyncio
async def test_non_matching_query_returns_nothing(chat_models, db):
    await add_chat(chat_models, db, "one", "Weekly report")

    assert await search_chat_titles(chat_models, "invoice") == []


@pytest.mark.asyncio
async def test_empty_filter_returns_every_chat(chat_models, db):
    await add_chat(chat_models, db, "one", "Weekly report")
    await add_chat(chat_models, db, "two", "CAFÉ")

    assert sorted(await search_chat_titles(chat_models, "")) == ["one", "two"]
