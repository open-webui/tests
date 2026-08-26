"""Regression: three 0.11.0 chat-store fixes (folder paging, sidebar ordering,
PostgreSQL chat search).

* `409fb39` (#26786) a shared folder was listed with a single
  `get_all_chats_by_folder_id(skip=0, limit=60)` call and there was no way to
  count the folder, so chats past the sixtieth simply never appeared. 0.11.0
  adds `count_all_chats_by_folder_id`, `sort_by`/`sort_dir`, and pages the
  router through the count.
* `f1ded94` / `a9617ca` background writes during a reply (suggestions, sources,
  status updates, saved variables, the automatic title) went through the chat
  upsert, which always bumped `Chat.updated_at`, so the sidebar re-sorted
  mid-reply. 0.11.0 threads `touch: bool = True` through the upsert and every
  background call-site passes `touch=False`.
* `cc9a445` on PostgreSQL a content search built one EXISTS clause over the
  legacy flat `chat->'messages'` array, so current conversations (whose text
  lives in `chat_message` and under `chat->'history'->'messages'`) were never
  matched. 0.11.0 ORs all three sources.

0.11.1 kept all three sources but split the match into a phrase clause plus one clause per
word, so the content bind is now one per clause rather than a single `content_key`.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (no folder count accessor and
no sort parameters, a background message write bumps `updated_at`, and the
PostgreSQL search SQL names only the legacy message array).

The SQL for the PostgreSQL branch is captured by a session stub that reports the
postgresql dialect and raises out of `execute`, so nothing needs a live server.
"""

from __future__ import annotations

import ast
import inspect
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.regression

OWNER = "alice"
FOLDER_ID = "folder-1"
CHAT_ID = "chat-1"
SEEDED_UPDATED_AT = 1000

CHAT_UPSERT_FUNCTIONS = (
    "upsert_message_to_chat_by_id_and_message_id",
    "add_message_status_to_chat_by_id_and_message_id",
)

# Background writes during a reply, by module and by the message field each one writes.
BACKGROUND_WRITES = {
    "open_webui/utils/middleware.py": ("followUps", "selectedModelId"),
    "open_webui/socket/main.py": ("embeds", "files", "sources"),
    "open_webui/utils/context_compaction.py": ("contextSummary",),
}


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def folders_router(owui_module):
    return owui_module("open_webui.routers.folders")


@pytest_asyncio.fixture
async def chat_db(chat_models, owui_module, tmp_path):
    """A private SQLite database holding the chat tables.

    The shipped `AsyncSessionLocal` is swapped for one bound to it, so the model
    layer runs its real SQL without touching the suite's shared store.
    """
    internal_db = owui_module("open_webui.internal.db")
    chat_messages = owui_module("open_webui.models.chat_messages")
    db_path = tmp_path / "chats.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    chat_models.Chat.__table__.create(sync_engine)
    chat_messages.ChatMessage.__table__.create(sync_engine)
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
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


async def add_chat(
    chat_models,
    session_maker,
    *,
    chat_id: str,
    title: str,
    updated_at: int,
    folder_id: str | None = FOLDER_ID,
    pinned: bool = False,
    archived: bool = False,
    meta: dict | None = None,
    chat: dict | None = None,
) -> None:
    async with session_maker() as session:
        session.add(
            chat_models.Chat(
                id=chat_id,
                user_id=OWNER,
                title=title,
                chat=chat if chat is not None else {"title": title, "history": {"messages": {}}},
                folder_id=folder_id,
                pinned=pinned,
                archived=archived,
                created_at=1,
                updated_at=updated_at,
                meta=meta if meta is not None else {},
            )
        )
        await session.commit()


async def seed_folder(chat_models, session_maker, count: int) -> list[str]:
    """`count` plain chats in FOLDER_ID, oldest first. Returns ids newest first."""
    for index in range(count):
        await add_chat(
            chat_models,
            session_maker,
            chat_id=f"chat-{index:03d}",
            title=f"Chat {index:03d}",
            updated_at=SEEDED_UPDATED_AT + index,
        )
    return [f"chat-{index:03d}" for index in reversed(range(count))]


async def read_chat_row(chat_models, session_maker, chat_id: str = CHAT_ID):
    async with session_maker() as session:
        result = await session.execute(
            select(chat_models.Chat).where(chat_models.Chat.id == chat_id)
        )
        return result.scalar_one()


# ---------------------------------------------------------------------------
# 1. Folder paging (#26786, 409fb39)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_folder_chat_count_sees_every_chat_not_just_the_first_page(chat_models, chat_db):
    """NARROW: a 65-chat folder has no count accessor at all pre-fix."""
    await seed_folder(chat_models, chat_db, 65)

    total = await chat_models.Chats.count_all_chats_by_folder_id(FOLDER_ID)

    assert total == 65


@pytest.mark.asyncio
async def test_second_page_returns_the_chats_past_the_first_sixty(chat_models, chat_db):
    """NEARBY: skip/limit already reached them, the router just never asked."""
    expected = await seed_folder(chat_models, chat_db, 65)

    page_two = await chat_models.Chats.get_all_chats_by_folder_id(FOLDER_ID, skip=60, limit=60)

    assert [chat["id"] for chat in page_two] == expected[60:]


@pytest.mark.asyncio
async def test_count_and_list_exclude_the_same_chats(chat_models, chat_db):
    """BROAD: pinned, archived and internal chats are hidden from both accessors."""
    await add_chat(chat_models, chat_db, chat_id="plain", title="Plain", updated_at=1000)
    await add_chat(
        chat_models, chat_db, chat_id="pinned", title="Pinned", updated_at=1001, pinned=True
    )
    await add_chat(
        chat_models, chat_db, chat_id="archived", title="Archived", updated_at=1002, archived=True
    )
    await add_chat(
        chat_models,
        chat_db,
        chat_id="internal",
        title="Internal",
        updated_at=1003,
        meta={"internal": True},
    )
    await add_chat(
        chat_models,
        chat_db,
        chat_id="elsewhere",
        title="Elsewhere",
        updated_at=1004,
        folder_id="folder-2",
    )

    listed = await chat_models.Chats.get_all_chats_by_folder_id(FOLDER_ID, skip=0, limit=60)
    total = await chat_models.Chats.count_all_chats_by_folder_id(FOLDER_ID)

    assert [chat["id"] for chat in listed] == ["plain"]
    assert total == 1


@pytest.mark.asyncio
async def test_folder_listing_sorts_by_title_ascending(chat_models, chat_db):
    """NARROW: sort_by/sort_dir do not exist pre-fix."""
    await add_chat(chat_models, chat_db, chat_id="c1", title="Zulu", updated_at=1002)
    await add_chat(chat_models, chat_db, chat_id="c2", title="Alpha", updated_at=1001)
    await add_chat(chat_models, chat_db, chat_id="c3", title="Mike", updated_at=1000)

    listed = await chat_models.Chats.get_all_chats_by_folder_id(
        FOLDER_ID, skip=0, limit=60, sort_by="title", sort_dir="asc"
    )

    assert [chat["title"] for chat in listed] == ["Alpha", "Mike", "Zulu"]


@pytest.mark.asyncio
async def test_unknown_sort_key_falls_back_to_updated_at(chat_models, chat_db):
    """NARROW: an unrecognised sort key keeps the newest-first default."""
    await add_chat(chat_models, chat_db, chat_id="c1", title="Zulu", updated_at=1002)
    await add_chat(chat_models, chat_db, chat_id="c2", title="Alpha", updated_at=1001)
    await add_chat(chat_models, chat_db, chat_id="c3", title="Mike", updated_at=1000)

    listed = await chat_models.Chats.get_all_chats_by_folder_id(
        FOLDER_ID, skip=0, limit=60, sort_by="nonsense", sort_dir="desc"
    )

    assert [chat["id"] for chat in listed] == ["c1", "c2", "c3"]


def test_shared_folder_route_pages_through_the_folder_count(folders_router):
    """BROAD: the route that dropped the tail must take a page and use the count."""
    handler = folders_router.get_shared_folder_chats

    assert "page" in inspect.signature(handler).parameters
    assert "count_all_chats_by_folder_id" in inspect.getsource(handler)


# ---------------------------------------------------------------------------
# 2. Sidebar ordering / touch=False (f1ded94, a9617ca)
# ---------------------------------------------------------------------------


def upsert_kwargs(chat_models) -> dict:
    """`touch=False` where the parameter exists, so the pre-fix ref fails on the
    bumped timestamp rather than on a TypeError."""
    upsert = chat_models.ChatTable.upsert_message_to_chat_by_id_and_message_id
    return {"touch": False} if "touch" in inspect.signature(upsert).parameters else {}


@pytest.mark.asyncio
async def test_background_message_write_leaves_updated_at_alone(chat_models, chat_db):
    """NARROW: a background upsert must not re-sort the sidebar mid-reply."""
    await add_chat(
        chat_models, chat_db, chat_id=CHAT_ID, title="Reply", updated_at=SEEDED_UPDATED_AT
    )

    await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
        CHAT_ID,
        "msg-1",
        {"role": "assistant", "content": "streamed chunk"},
        **upsert_kwargs(chat_models),
    )

    row = await read_chat_row(chat_models, chat_db)
    assert row.chat["history"]["messages"]["msg-1"]["content"] == "streamed chunk"
    assert row.updated_at == SEEDED_UPDATED_AT


@pytest.mark.asyncio
async def test_foreground_message_write_still_bumps_updated_at(chat_models, chat_db):
    """NEARBY: the default path keeps moving the chat to the top."""
    await add_chat(
        chat_models, chat_db, chat_id=CHAT_ID, title="Reply", updated_at=SEEDED_UPDATED_AT
    )

    await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
        CHAT_ID, "msg-1", {"role": "user", "content": "hello"}
    )

    row = await read_chat_row(chat_models, chat_db)
    assert row.updated_at > SEEDED_UPDATED_AT


def _chat_upserts_opting_out_of_touch(source: str) -> list[ast.Call]:
    """Chat-upsert calls that pass `touch=False`."""
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in CHAT_UPSERT_FUNCTIONS
        and any(
            keyword.arg == "touch"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
    ]


def _literal_payload_keys(call: ast.Call) -> set[str]:
    return {
        key.value
        for arg in call.args
        if isinstance(arg, ast.Dict)
        for key in arg.keys
        if isinstance(key, ast.Constant)
    }


@pytest.mark.parametrize(("relative_path", "fields"), sorted(BACKGROUND_WRITES.items()))
def test_background_call_sites_opt_out_of_touching_the_chat(
    open_webui_backend, relative_path, fields
):
    """BROAD: every background message write must pass touch=False on its chat upsert."""
    source = (open_webui_backend / relative_path).read_text(encoding="utf-8")

    opted_out = _chat_upserts_opting_out_of_touch(source)
    untouched_fields: set[str] = set()
    for call in opted_out:
        untouched_fields |= _literal_payload_keys(call)

    missing = sorted(set(fields) - untouched_fields)
    assert not missing, f"{relative_path} re-sorts the sidebar when it writes {missing}"
    assert len(opted_out) == len(re.findall(r"touch\s*=\s*False", source)), (
        f"{relative_path} has a touch=False that is not on a chat upsert"
    )


# ---------------------------------------------------------------------------
# 3. PostgreSQL chat search (cc9a445)
# ---------------------------------------------------------------------------


class StatementCaptured(BaseException):
    """BaseException so a production `except Exception` cannot swallow it."""

    def __init__(self, statement):
        super().__init__("statement captured")
        self.statement = statement


class PostgresSessionStub:
    """Stands in for the DB only: reports the postgresql dialect, runs nothing."""

    async def connection(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    async def execute(self, statement, *args, **kwargs):
        raise StatementCaptured(statement)


async def compile_postgres_search(chat_models, search_text: str):
    @asynccontextmanager
    async def fake_context(db=None):
        yield PostgresSessionStub()

    with patch.object(chat_models, "get_async_db_context", fake_context):
        try:
            await chat_models.Chats.get_chats_by_user_id_and_search_text(OWNER, search_text)
        except StatementCaptured as captured:
            return captured.statement.compile(dialect=postgresql.dialect())
    raise AssertionError("the search query was never executed")


@pytest.mark.asyncio
async def test_postgres_search_covers_all_three_message_stores(chat_models):
    """NARROW: pre-fix only the legacy flat array is searched."""
    compiled = await compile_postgres_search(chat_models, "needle")
    sql = str(compiled)

    assert "FROM chat_message AS message" in sql
    assert "json_each(Chat.chat#>'{history,messages}')" in sql
    assert "json_array_elements(Chat.chat->'messages')" in sql


@pytest.mark.asyncio
async def test_postgres_message_table_clause_cannot_cross_accounts(chat_models):
    """BROAD: the chat_message join is scoped by chat and by owner."""
    compiled = await compile_postgres_search(chat_models, "needle")
    sql = str(compiled)

    assert "FROM chat_message AS message" in sql
    assert "message.chat_id = Chat.id" in sql
    assert "message.user_id = Chat.user_id" in sql


@pytest.mark.asyncio
async def test_postgres_search_binds_the_needle_once(chat_models):
    """NEARBY: every EXISTS reuses a lowercased content bind, none re-binds per store.

    0.11.1 splits the match into a phrase clause plus one clause per word, so the bind
    count follows the clause count instead of being fixed at one.
    """
    compiled = await compile_postgres_search(chat_models, "NeEdLe")

    content_keys = [key for key in compiled.params if "content_key" in key]
    title_keys = [key for key in compiled.params if "title_key" in key]

    assert content_keys, "the search bound no message-content parameter"
    assert {compiled.params[key] for key in content_keys} == {"needle"}
    assert "NeEdLe" not in str(compiled.params), "the raw mixed-case text reached a bind"
    assert len(content_keys) == len(title_keys), "a match clause re-bound the needle per store"
