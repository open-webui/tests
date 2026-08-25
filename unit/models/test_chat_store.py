"""Regression: the 0.11.1 round of chat-store fixes (search, current position,
cyclic histories, null bytes, shared-chat tags, concurrent saves).

Six upstream fixes land in `open_webui/models/chats.py` and its callers:

* `0800c21c6` chat search only looked at the legacy `$.messages` array on SQLite,
  so on a default install the bodies of current conversations, which live under
  `$.history.messages`, were never searched.
* `5caa91a49` reopening a compacted chat parked the user on the summary message.
  `_repair_chat_current_id` now walks to the last descendant when the current
  message carries a `contextSummary`, and editing an existing message no longer
  drags `currentId` onto it.
* `b933292d6` (#28035) `delete_message_from_history` followed `childrenIds[-1]`
  with no record of where it had been, so deleting a message in a chat whose
  reply structure loops never returned. Sibling fix `5c79ccc9e` (#28034) does the
  same for `get_message_list`, which tracked each message's own `id` field even
  though a message may omit it.
* `ce22e0bb1` (#28820) only `message['content']` was sanitized, so null bytes in
  any other field of the message, or in the message id, reached the separate
  `chat_message` record raw and PostgreSQL rejected the write.
* `7d4747dfd` (#28767) tag resolution moved into `Chats.get_chat_by_id_for_user`,
  so a chat shared with you finally loads its tags and an admin reading someone
  else's chat sees the chat's tags instead of their own.
* `1c13fedb1` (#28742) `update_chat_by_id` read the whole blob, merged in memory
  and wrote it back, so two concurrent savers each discarded the other's work and
  messages vanished from the conversation. It now patches the given keys and
  merges history.

Discriminates: passes on v0.11.1, fails on v0.11.0 (search misses
`history.messages`, the compacted chat stays on its summary, editing a message
moves the current position, the cyclic walks never terminate, null bytes reach
the message record, a share recipient is refused the chat's tags, and a partial
chat update wipes every key it did not name).

Both cyclic walks are driven through a message map that raises once it has been
asked for far more entries than an acyclic history could hold, so a checkout
without the fix fails fast instead of wedging the run.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.regression

OWNER = "alice"
VIEWER = "dana"
CHAT_ID = "chat-1"

# Any walk over the fixtures below visits a handful of messages. A walk asking
# for more than this is going round a loop.
WALK_BUDGET = 200


class WalkBudgetExceeded(RuntimeError):
    pass


class BoundedMessages(dict):
    """Message map that raises once a walk has clearly stopped making progress."""

    def __init__(self, *args, budget: int = WALK_BUDGET, **kwargs):
        super().__init__(*args, **kwargs)
        self._budget = budget
        self.lookups = 0

    def get(self, key, default=None):
        self.lookups += 1
        if self.lookups > self._budget:
            raise WalkBudgetExceeded(
                f"walk exceeded {self._budget} lookups, it is following a cycle"
            )
        return super().get(key, default)


@pytest.fixture(scope="module")
def chat_models(owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def chats_router(owui_module):
    return owui_module("open_webui.routers.chats")


@pytest.fixture(scope="module")
def misc_module(owui_module):
    return owui_module("open_webui.utils.misc")


@pytest_asyncio.fixture
async def chat_db(chat_models, owui_module, tmp_path):
    """A private SQLite database holding just the chat table.

    The shipped `AsyncSessionLocal` is swapped for one bound to it, so the model
    layer runs its real SQL without touching the suite's shared store.
    """
    internal_db = owui_module("open_webui.internal.db")
    db_path = tmp_path / "chats.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    chat_models.Chat.__table__.create(sync_engine)
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


async def seed_chat(
    chat_models, session_maker, chat: dict, *, chat_id: str = CHAT_ID, title: str = "Untitled"
):
    async with session_maker() as session:
        session.add(
            chat_models.Chat(
                id=chat_id,
                user_id=OWNER,
                title=title,
                chat=chat,
                created_at=1,
                updated_at=1,
                meta={},
            )
        )
        await session.commit()


async def stored_chat(chat_models, session_maker, chat_id: str = CHAT_ID) -> dict:
    async with session_maker() as session:
        return (await session.get(chat_models.Chat, chat_id)).chat


def message(content: str, role: str = "user", **extra) -> dict:
    return {
        "id": extra.pop("id", "m1"),
        "parentId": None,
        "childrenIds": [],
        "role": role,
        "content": content,
        **extra,
    }


# --- Narrow: 29, chat search reads the current storage format -------------------


@pytest.mark.asyncio
async def test_search_finds_message_in_current_storage_format(chat_models, chat_db):
    await seed_chat(
        chat_models,
        chat_db,
        {
            "history": {
                "currentId": "m1",
                "messages": {"m1": message("the pterodactyl migration plan")},
            }
        },
    )

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(OWNER, "pterodactyl")

    assert [chat.id for chat in found] == [CHAT_ID], (
        "chat search only read the legacy top-level messages array, so on a default "
        "SQLite install the bodies of current conversations were never searched"
    )


@pytest.mark.asyncio
async def test_search_finds_message_by_single_term_of_a_phrase(chat_models, chat_db):
    await seed_chat(
        chat_models,
        chat_db,
        {
            "history": {
                "currentId": "m1",
                "messages": {"m1": message("quarterly pterodactyl budget")},
            }
        },
    )

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(
        OWNER, "pterodactyl budget"
    )

    assert [chat.id for chat in found] == [CHAT_ID]


# --- Nearby: search behaviour that already worked ------------------------------


@pytest.mark.asyncio
async def test_search_still_finds_legacy_top_level_messages(chat_models, chat_db):
    await seed_chat(chat_models, chat_db, {"messages": [message("the pterodactyl migration plan")]})

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(OWNER, "pterodactyl")

    assert [chat.id for chat in found] == [CHAT_ID]


@pytest.mark.asyncio
async def test_search_still_matches_the_title(chat_models, chat_db):
    await seed_chat(chat_models, chat_db, {"history": {"messages": {}}}, title="Pterodactyl notes")

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(OWNER, "pterodactyl")

    assert [chat.id for chat in found] == [CHAT_ID]


@pytest.mark.asyncio
async def test_search_does_not_match_an_unrelated_chat(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"history": {"currentId": "m1", "messages": {"m1": message("hello")}}}
    )

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(OWNER, "pterodactyl")

    assert found == []


@pytest.mark.asyncio
async def test_search_of_another_users_chat_returns_nothing(chat_models, chat_db):
    await seed_chat(
        chat_models,
        chat_db,
        {"history": {"currentId": "m1", "messages": {"m1": message("pterodactyl")}}},
    )

    found = await chat_models.Chats.get_chats_by_user_id_and_search_text(VIEWER, "pterodactyl")

    assert found == []


# --- Narrow: 30, where a compacted chat reopens --------------------------------


def compacted_history() -> dict:
    return {
        "currentId": "summary",
        "messages": {
            "summary": {
                "id": "summary",
                "parentId": None,
                "childrenIds": ["reply"],
                "role": "assistant",
                "content": "summary of the earlier turns",
                "timestamp": 10,
                "contextSummary": {"tokens": 512},
            },
            "reply": {
                "id": "reply",
                "parentId": "summary",
                "childrenIds": ["followup"],
                "role": "user",
                "content": "carry on",
                "timestamp": 11,
            },
            "followup": {
                "id": "followup",
                "parentId": "reply",
                "childrenIds": [],
                "role": "assistant",
                "content": "carrying on",
                "timestamp": 12,
            },
        },
    }


@pytest.mark.asyncio
async def test_compacted_chat_reopens_after_the_summary(chat_models):
    chat = {"history": compacted_history()}

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat["history"]["currentId"] == "followup", (
        "reopening a compacted chat parked the reader on the summary message "
        "instead of the newest turn below it"
    )
    assert changed is True


@pytest.mark.asyncio
async def test_editing_an_existing_message_does_not_move_the_current_position(chat_models):
    history = {
        "currentId": "newest",
        "messages": {
            "old": {
                "id": "old",
                "parentId": None,
                "childrenIds": ["newest"],
                "role": "user",
                "content": "first",
            },
            "newest": {
                "id": "newest",
                "parentId": "old",
                "childrenIds": [],
                "role": "assistant",
                "content": "second",
            },
        },
    }

    chat_models.Chats.upsert_message_to_history(history, "old", {"content": "first, edited"})

    assert history["currentId"] == "newest", (
        "updating an existing message dragged the current position back onto it, "
        "so an edit or a late status write moved the reader off the newest turn"
    )
    assert history["messages"]["old"]["content"] == "first, edited"


# --- Nearby: current-position behaviour that must survive ----------------------


@pytest.mark.asyncio
async def test_new_message_still_becomes_the_current_position(chat_models):
    history = {
        "currentId": "old",
        "messages": {
            "old": {
                "id": "old",
                "parentId": None,
                "childrenIds": [],
                "role": "user",
                "content": "first",
            }
        },
    }

    chat_models.Chats.upsert_message_to_history(
        history, "fresh", {"parentId": "old", "content": "second"}
    )

    assert history["currentId"] == "fresh"


@pytest.mark.asyncio
async def test_repair_leaves_an_ordinary_current_message_alone(chat_models):
    history = compacted_history()
    history["messages"]["summary"].pop("contextSummary")
    chat = {"history": history}

    changed = chat_models.Chats._repair_chat_current_id(chat)

    assert chat["history"]["currentId"] == "summary"
    assert changed is False


@pytest.mark.asyncio
async def test_repair_ignores_a_chat_without_a_history(chat_models):
    assert chat_models.Chats._repair_chat_current_id({}) is False
    assert chat_models.Chats._repair_chat_current_id({"history": None}) is False


# --- Narrow: 107, deleting a message in a looping chat -------------------------


def cyclic_history() -> dict:
    """`a` and `b` are each other's child, so a walk down `childrenIds` loops."""
    return {
        "currentId": "victim",
        "messages": BoundedMessages(
            {
                "root": {
                    "id": "root",
                    "parentId": None,
                    "childrenIds": ["victim", "a"],
                    "role": "user",
                },
                "victim": {
                    "id": "victim",
                    "parentId": "root",
                    "childrenIds": [],
                    "role": "assistant",
                },
                "a": {"id": "a", "parentId": "root", "childrenIds": ["b"], "role": "user"},
                "b": {"id": "b", "parentId": "a", "childrenIds": ["a"], "role": "assistant"},
            }
        ),
    }


@pytest.mark.asyncio
async def test_deleting_a_message_terminates_when_replies_loop(chat_models):
    history = cyclic_history()

    deleted_ids = chat_models.Chats.delete_message_from_history(history, "victim")

    assert deleted_ids == {"victim"}
    assert history["currentId"] == "b", (
        "the walk to the deepest reply followed childrenIds[-1] without recording "
        "where it had been, so deleting a message in a looping chat never returned (#28035)"
    )


# --- Broad: every walk over a message graph must terminate ---------------------


@pytest.mark.asyncio
async def test_message_list_terminates_when_messages_omit_their_id(misc_module):
    # The map key is the only identity here; the message bodies carry no 'id'.
    messages_map = BoundedMessages(
        {
            "a": {"parentId": "b", "role": "user", "content": "first"},
            "b": {"parentId": "a", "role": "assistant", "content": "second"},
        }
    )

    result = misc_module.get_message_list(messages_map, "a")

    assert len(result) == 2, (
        "the cycle guard recorded each message's own 'id' field, which a message "
        "need not carry, so a looping history was walked forever (#28034)"
    )


# --- Nearby: the same walks on well-formed histories ---------------------------


@pytest.mark.asyncio
async def test_deleting_a_message_moves_to_the_deepest_remaining_reply(chat_models):
    history = {
        "currentId": "victim",
        "messages": {
            "root": {
                "id": "root",
                "parentId": None,
                "childrenIds": ["victim", "a"],
                "role": "user",
            },
            "victim": {"id": "victim", "parentId": "root", "childrenIds": [], "role": "assistant"},
            "a": {"id": "a", "parentId": "root", "childrenIds": ["b"], "role": "user"},
            "b": {"id": "b", "parentId": "a", "childrenIds": [], "role": "assistant"},
        },
    }

    deleted_ids = chat_models.Chats.delete_message_from_history(history, "victim")

    assert deleted_ids == {"victim"}
    assert history["currentId"] == "b"


@pytest.mark.asyncio
async def test_message_list_walks_a_plain_chain(misc_module):
    messages_map = {
        "a": {"id": "a", "parentId": None, "content": "first"},
        "b": {"id": "b", "parentId": "a", "content": "second"},
    }

    assert [entry["content"] for entry in misc_module.get_message_list(messages_map, "b")] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_message_list_handles_empty_and_missing_input(misc_module):
    assert misc_module.get_message_list(None, "a") == []
    assert misc_module.get_message_list({}, "a") == []
    assert misc_module.get_message_list({"a": {"id": "a"}}, "nope") == []


# --- Narrow: 137, null bytes outside message content ---------------------------


@contextmanager
def captured_message_record(chat_models):
    """Stand in for the separate `chat_message` write and keep what it was handed."""
    recorder = AsyncMock()
    with patch.object(chat_models.ChatMessages, "upsert_message", recorder):
        yield recorder


@pytest.mark.asyncio
async def test_null_bytes_are_cleaned_from_the_whole_message_record(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    with captured_message_record(chat_models) as recorder:
        await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
            CHAT_ID,
            "m1",
            {
                "role": "assistant",
                "content": "fine",
                "model": "gpt\x004",
                "sources": [{"name": "doc\x00.pdf"}],
            },
        )

    written = recorder.await_args.kwargs["data"]
    assert "\x00" not in written["model"], (
        "only message['content'] was sanitized, so null bytes in any other field "
        "reached the separate message record raw and PostgreSQL refused the write (#28820)"
    )
    assert "\x00" not in written["sources"][0]["name"]


@pytest.mark.asyncio
async def test_null_bytes_are_cleaned_from_the_message_id(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    with captured_message_record(chat_models) as recorder:
        await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
            CHAT_ID, "m\x001", {"role": "assistant", "content": "fine"}
        )

    assert "\x00" not in recorder.await_args.kwargs["message_id"]


# --- Nearby: sanitizing must not damage ordinary writes ------------------------


@pytest.mark.asyncio
async def test_ordinary_message_is_written_through_unchanged(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    with captured_message_record(chat_models) as recorder:
        await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
            CHAT_ID, "m1", {"role": "assistant", "content": "plain text", "model": "gpt-4"}
        )

    written = recorder.await_args.kwargs["data"]
    assert written["content"] == "plain text"
    assert written["model"] == "gpt-4"
    assert (await stored_chat(chat_models, chat_db))["history"]["messages"]["m1"][
        "content"
    ] == "plain text"


@pytest.mark.asyncio
async def test_null_bytes_are_still_cleaned_from_message_content(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    with captured_message_record(chat_models) as recorder:
        await chat_models.Chats.upsert_message_to_chat_by_id_and_message_id(
            CHAT_ID, "m1", {"role": "assistant", "content": "before\x00after"}
        )

    assert "\x00" not in recorder.await_args.kwargs["data"]["content"]


# --- Narrow: 141, tags on a chat shared with you -------------------------------


def build_chat(chat_models, tags: list[str], folder_id: str | None = None):
    return chat_models.ChatModel(
        id=CHAT_ID,
        user_id=OWNER,
        title="Quarterly numbers",
        chat={"messages": []},
        created_at=1,
        updated_at=2,
        folder_id=folder_id,
        meta={"tags": tags},
    )


def patch_everywhere(stack, modules, holder_name, attribute, replacement):
    """Patch a shared collaborator wherever the checkout happens to import it.

    0.11.0 resolves the tags endpoint's dependencies inside `routers.chats`;
    0.11.1 moved that into `models.chats`. Patch both so the stub is the same
    test either way.
    """
    patched = False
    for module in modules:
        holder = getattr(module, holder_name, None)
        if holder is not None and hasattr(holder, attribute):
            stack.enter_context(patch.object(holder, attribute, replacement))
            patched = True
    assert patched, f"{holder_name}.{attribute} not found in this checkout"


@contextmanager
def tag_backend(
    chat_models,
    chats_router,
    owui_module,
    *,
    chat,
    owned_by_caller=False,
    has_grant=False,
    folder_readable=False,
):
    """Stub the storage boundary the tags endpoint reads, leaving its logic alone."""
    folders_access = owui_module("open_webui.utils.access_control.folders")
    modules = (chat_models, chats_router)
    tag_lookup = AsyncMock(return_value=[])
    folder = SimpleNamespace(id=chat.folder_id, user_id=OWNER) if chat and chat.folder_id else None
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                chat_models.Chats,
                "get_chat_by_id_and_user_id",
                AsyncMock(return_value=chat if owned_by_caller else None),
            )
        )
        stack.enter_context(
            patch.object(chat_models.Chats, "get_chat_by_id", AsyncMock(return_value=chat))
        )
        patch_everywhere(
            stack, modules, "AccessGrants", "has_access", AsyncMock(return_value=has_grant)
        )
        patch_everywhere(
            stack, modules, "Folders", "get_folder_by_id", AsyncMock(return_value=folder)
        )
        stack.enter_context(
            patch.object(
                folders_access, "has_folder_access", AsyncMock(return_value=folder_readable)
            )
        )
        stack.enter_context(
            patch.object(chats_router.Tags, "get_tags_by_ids_and_user_id", tag_lookup)
        )
        yield tag_lookup


def user(role: str = "user", user_id: str = VIEWER):
    return SimpleNamespace(id=user_id, role=role)


@pytest.mark.asyncio
async def test_share_recipient_gets_the_chats_tags(chat_models, chats_router, owui_module):
    chat = build_chat(chat_models, ["tag-a"])
    with tag_backend(
        chat_models, chats_router, owui_module, chat=chat, has_grant=True
    ) as tag_lookup:
        await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(), db=None)

    assert tag_lookup.await_args.args[:2] == (["tag-a"], OWNER), (
        "the tags endpoint resolved the chat by ownership only, so opening a chat "
        "someone shared with you failed to load its tags (#28767)"
    )


@pytest.mark.asyncio
async def test_shared_folder_recipient_gets_the_chats_tags(chat_models, chats_router, owui_module):
    chat = build_chat(chat_models, ["tag-a"], folder_id="folder-1")
    with tag_backend(
        chat_models, chats_router, owui_module, chat=chat, folder_readable=True
    ) as tag_lookup:
        await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(), db=None)

    assert tag_lookup.await_args.args[:2] == (["tag-a"], OWNER)


@pytest.mark.asyncio
async def test_admin_reading_another_users_chat_sees_that_chats_tags(
    chat_models, chats_router, owui_module
):
    chat = build_chat(chat_models, ["tag-a"])
    with tag_backend(chat_models, chats_router, owui_module, chat=chat) as tag_lookup:
        await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(role="admin"), db=None)

    assert tag_lookup.await_args.args[:2] == (["tag-a"], OWNER), (
        "an admin opening someone else's chat had the chat's tag ids resolved "
        "against their own tag rows, so the sidebar showed the wrong tags (#28767)"
    )


# --- Nearby: the tags endpoint must stay closed to everyone else ---------------


@pytest.mark.asyncio
async def test_owner_still_gets_their_own_tags(chat_models, chats_router, owui_module):
    chat = build_chat(chat_models, ["tag-a"])
    with tag_backend(
        chat_models, chats_router, owui_module, chat=chat, owned_by_caller=True
    ) as tag_lookup:
        await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(user_id=OWNER), db=None)

    assert tag_lookup.await_args.args[:2] == (["tag-a"], OWNER)


@pytest.mark.asyncio
async def test_stranger_is_refused_the_chats_tags(chat_models, chats_router, owui_module):
    chat = build_chat(chat_models, ["tag-a"])
    with tag_backend(chat_models, chats_router, owui_module, chat=chat):
        with pytest.raises(Exception) as excinfo:
            await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(user_id="mallory"), db=None)

    assert getattr(excinfo.value, "status_code", None) == 401


@pytest.mark.asyncio
async def test_missing_chat_is_refused(chat_models, chats_router, owui_module):
    with tag_backend(chat_models, chats_router, owui_module, chat=None):
        with pytest.raises(Exception) as excinfo:
            await chats_router.get_chat_tags_by_id(CHAT_ID, user=user(), db=None)

    assert getattr(excinfo.value, "status_code", None) == 401


# --- Narrow: 200, a partial save must not discard the rest of the chat ---------


@pytest.mark.asyncio
async def test_partial_update_keeps_the_keys_it_did_not_name(chat_models, chat_db):
    await seed_chat(
        chat_models,
        chat_db,
        {
            "title": "Untitled",
            "models": ["gpt-4"],
            "history": {"currentId": "m1", "messages": {"m1": message("hello")}},
        },
    )

    await chat_models.Chats.update_chat_by_id(CHAT_ID, {"files": [{"id": "f1"}]}, touch=False)

    saved = await stored_chat(chat_models, chat_db)
    assert saved["files"] == [{"id": "f1"}]
    assert "m1" in saved.get("history", {}).get("messages", {}), (
        "the save rewrote the whole chat blob from the fields it was given, so "
        "attaching a file during a reply wiped the conversation (#28742)"
    )
    assert saved["models"] == ["gpt-4"]


@pytest.mark.asyncio
async def test_a_stale_writer_does_not_drop_a_message_saved_since_its_read(chat_models, chat_db):
    await seed_chat(
        chat_models,
        chat_db,
        {"title": "Untitled", "history": {"currentId": "m1", "messages": {"m1": message("hello")}}},
    )
    # Second saver landed while the first was still holding its older copy.
    await chat_models.Chats.update_chat_by_id(
        CHAT_ID,
        {
            "history": {
                "currentId": "m2",
                "messages": {"m1": message("hello"), "m2": message("world", id="m2")},
            }
        },
        touch=False,
    )

    stale_history = {"currentId": "m1", "messages": {"m1": message("hello")}}
    await chat_models.Chats.update_chat_by_id(CHAT_ID, {"history": stale_history}, touch=False)

    saved = await stored_chat(chat_models, chat_db)
    assert set(saved["history"]["messages"]) == {"m1", "m2"}, (
        "two concurrent savers each wrote a full blob built from their own earlier "
        "read, so the later write silently deleted the other's message (#28742)"
    )


# --- Nearby: ordinary updates still land ---------------------------------------


@pytest.mark.asyncio
async def test_update_writes_the_new_title(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    updated = await chat_models.Chats.update_chat_by_id(CHAT_ID, {"title": "Renamed"}, touch=False)

    assert updated.title == "Renamed"
    assert (await stored_chat(chat_models, chat_db))["title"] == "Renamed"


@pytest.mark.asyncio
async def test_update_of_a_missing_chat_returns_none(chat_models, chat_db):
    assert await chat_models.Chats.update_chat_by_id("no-such-chat", {"title": "x"}) is None


@pytest.mark.asyncio
async def test_update_strips_null_bytes_from_the_blob(chat_models, chat_db):
    await seed_chat(
        chat_models, chat_db, {"title": "Untitled", "history": {"currentId": None, "messages": {}}}
    )

    await chat_models.Chats.update_chat_by_id(CHAT_ID, {"title": "bad\x00title"}, touch=False)

    assert "\x00" not in (await stored_chat(chat_models, chat_db))["title"]
