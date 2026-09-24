"""The two 0.11.1 chat-store fixes HTTP cannot show safely.

* `ce22e0bb1` (#28820) only `message['content']` was sanitized, so a null byte in any other
  field of a message, or in its id, reached the separate `chat_message` record raw, and
  PostgreSQL refused that write. SQLite stores the null byte, so an HTTP test on the default
  database sees nothing; the stored record is read back here instead.
* `5c79ccc9e` (#28034) `get_message_list` guarded against cycles with each message's own `id`
  field, which a message may omit, so a looping history was walked forever. Before the fix the
  walk also grows its result list without bound, which would exhaust a server's memory, so it
  is driven here through a message map that raises once a walk overruns any acyclic history.

The rest of the round is pinned over HTTP by integration/models/test_chat_store.py.

Discriminates: passes on upstream dev `bbfa876af`; with the message upsert back on sanitizing
`content` alone both null-byte tests fail, and with `get_message_list` back on the `id` field
the looping walk raises `WalkBudgetExceeded`.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.regression

OWNER = "alice"
WALK_BUDGET = 200


class WalkBudgetExceeded(RuntimeError):
    pass


class BoundedMessages(dict):
    """A message map that raises once a walk has asked for more entries than it could hold."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookups = 0

    def get(self, key, default=None):
        self.lookups += 1
        if self.lookups > WALK_BUDGET:
            raise WalkBudgetExceeded(f"{WALK_BUDGET} lookups: the walk is following a cycle")
        return super().get(key, default)


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def get_message_list(owui_module):
    return owui_module("open_webui.utils.misc").get_message_list


@pytest_asyncio.fixture
async def private_db(chat_models, owui_module, tmp_path, monkeypatch):
    """The real chat tables in a scratch SQLite file, in place of the suite's shared store."""
    chat_messages = owui_module("open_webui.models.chat_messages")
    db_path = tmp_path / "chats.db"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    chat_models.Chat.__table__.create(sync_engine)
    chat_messages.ChatMessage.__table__.create(sync_engine)
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    sessions = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(owui_module("open_webui.internal.db"), "AsyncSessionLocal", sessions)
    yield
    await async_engine.dispose()


async def save_message(chat_models, owui_module, message_id: str, message: dict):
    """Save one message to a fresh chat; returns what the `chat_message` table holds for it."""
    chat_id = str(uuid.uuid4())
    empty_chat = chat_models.ChatForm(chat={"title": "Untitled", "history": {"messages": {}}})
    await chat_models.Chats.insert_new_chat(id=chat_id, user_id=OWNER, form_data=empty_chat)

    await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
        id=chat_id, message_id=message_id, message=message
    )

    chat_messages = owui_module("open_webui.models.chat_messages").ChatMessages
    (record,) = await chat_messages.get_messages_by_chat_id(chat_id=chat_id)
    return record


# --- ce22e0bb1: null bytes outside the message content ------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("private_db")
async def test_null_bytes_are_cleaned_from_every_field_of_the_message_record(
    chat_models, owui_module
):
    record = await save_message(
        chat_models,
        owui_module,
        "m1",
        {
            "role": "assistant",
            "content": "fine",
            "model": "gpt\x004",
            "sources": [{"name": "a\x00b"}],
        },
    )

    assert record.model_id == "gpt4", "a null byte outside content reached the message record"
    assert record.sources == [{"name": "ab"}]


@pytest.mark.asyncio
@pytest.mark.usefixtures("private_db")
async def test_null_bytes_are_cleaned_from_the_message_id(chat_models, owui_module):
    record = await save_message(
        chat_models, owui_module, "m\x001", {"role": "assistant", "content": "fine"}
    )

    assert "\x00" not in record.id


@pytest.mark.asyncio
@pytest.mark.usefixtures("private_db")
async def test_content_is_still_cleaned_and_a_plain_message_is_kept_as_is(chat_models, owui_module):
    cleaned = await save_message(
        chat_models, owui_module, "m1", {"role": "assistant", "content": "be\x00fore"}
    )
    plain = await save_message(
        chat_models, owui_module, "m2", {"role": "assistant", "content": "plain", "model": "gpt-4"}
    )

    assert cleaned.content == "before"
    assert (plain.content, plain.model_id) == ("plain", "gpt-4")


# --- 5c79ccc9e: walking a looping history whose messages carry no id ----------------------


def test_a_looping_history_without_message_ids_is_walked_once(get_message_list):
    messages = BoundedMessages(
        {
            "a": {"parentId": "b", "role": "user", "content": "first"},
            "b": {"parentId": "a", "role": "assistant", "content": "second"},
        }
    )

    walked = get_message_list(messages_map=messages, message_id="a")

    assert [entry["content"] for entry in walked] == ["second", "first"]


def test_a_plain_chain_is_walked_from_the_root(get_message_list):
    messages = {
        "a": {"id": "a", "parentId": None, "content": "first"},
        "b": {"id": "b", "parentId": "a", "content": "second"},
    }

    walked = get_message_list(messages_map=messages, message_id="b")

    assert [entry["content"] for entry in walked] == ["first", "second"]
