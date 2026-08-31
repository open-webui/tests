"""Regression test for the tool approval drain lookup running on fresh chat messages.

Fix commit `9f680bb80` (PR #29142), `drain_approved_tool_calls` in `open_webui/utils/middleware.py`.

`drain_approved_tool_calls` gates on the message id it is about to read. It used to accept
`message_id` on its own, and every ordinary send carries one, so each new turn paid a chat
message lookup for an assistant message that does not exist yet and can hold no approvals. The
gate now requires `assistant_message_id`, which only a resume or continue payload sends.

Discriminates: passes on v0.11.3, fails on v0.11.1 (a fresh message id still triggers the
message lookup).
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]


@pytest.fixture(scope="session")
def middleware_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.middleware")


@pytest.fixture
def chats_stub(middleware_module: ModuleType):
    """Stub the chat store, the only I/O `drain_approved_tool_calls` reaches for on these paths."""
    stub = SimpleNamespace(
        get_message_by_id_and_message_id=AsyncMock(return_value=None),
        upsert_message_to_chat_by_id_and_message_id=AsyncMock(return_value=None),
    )
    with patch.object(middleware_module, "Chats", stub):
        yield stub


async def drain(middleware_module: ModuleType, metadata: dict) -> bool:
    return await middleware_module.drain_approved_tool_calls(
        SimpleNamespace(), {"messages": []}, SimpleNamespace(id="user-1"), {}, metadata
    )


# -----------------------------------------------------------------------------
# Narrow
# -----------------------------------------------------------------------------


async def test_fresh_message_id_does_not_trigger_the_drain_lookup(
    middleware_module: ModuleType, chats_stub
) -> None:
    drained = await drain(middleware_module, {"chat_id": "chat-1", "message_id": "msg-fresh"})

    assert drained is False
    chats_stub.get_message_by_id_and_message_id.assert_not_called()


async def test_resume_payload_still_triggers_the_drain_lookup(
    middleware_module: ModuleType, chats_stub
) -> None:
    drained = await drain(
        middleware_module, {"chat_id": "chat-1", "assistant_message_id": "msg-existing"}
    )

    assert drained is False
    chats_stub.get_message_by_id_and_message_id.assert_awaited_once_with("chat-1", "msg-existing")


async def test_resume_payload_reads_the_message_id_it_was_given(
    middleware_module: ModuleType, chats_stub
) -> None:
    """A continue payload sends both ids; the message holding the output is message_id."""
    await drain(
        middleware_module,
        {"chat_id": "chat-1", "message_id": "msg-1", "assistant_message_id": "msg-1-assistant"},
    )

    chats_stub.get_message_by_id_and_message_id.assert_awaited_once_with("chat-1", "msg-1")


# -----------------------------------------------------------------------------
# Broad: only a payload re-entering an existing assistant message reads the store
# -----------------------------------------------------------------------------


FRESH_METADATA = [
    ("message_id_only", {"chat_id": "chat-1", "message_id": "msg-fresh"}),
    (
        "message_id_with_approval_mode",
        {
            "chat_id": "chat-1",
            "message_id": "msg-fresh",
            "params": {"tool_approval_mode": "ask"},
        },
    ),
    (
        "blank_assistant_message_id",
        {"chat_id": "chat-1", "message_id": "m", "assistant_message_id": ""},
    ),
    ("no_ids_at_all", {"chat_id": "chat-1"}),
]


@pytest.mark.parametrize(
    "case,metadata", FRESH_METADATA, ids=[case for case, _ in FRESH_METADATA]
)
async def test_no_store_read_without_an_assistant_message_id(
    middleware_module: ModuleType, chats_stub, case: str, metadata: dict
) -> None:
    assert await drain(middleware_module, metadata) is False
    chats_stub.get_message_by_id_and_message_id.assert_not_called()


# -----------------------------------------------------------------------------
# Nearby
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("chat_id", ["temporary:abc", "local:abc", "channel:abc", "", None])
async def test_unsaved_chat_never_drains(
    middleware_module: ModuleType, chats_stub, chat_id
) -> None:
    metadata = {"chat_id": chat_id, "assistant_message_id": "msg-1", "message_id": "msg-1"}

    assert await drain(middleware_module, metadata) is False
    chats_stub.get_message_by_id_and_message_id.assert_not_called()


async def test_message_without_approved_calls_is_left_alone(
    middleware_module: ModuleType, chats_stub
) -> None:
    output = [
        {"type": "function_call", "call_id": "call-a", "name": "web_search", "status": "completed"},
        {"type": "function_call_output", "call_id": "call-a", "output": []},
    ]
    chats_stub.get_message_by_id_and_message_id.return_value = {"output": output}

    drained = await drain(
        middleware_module, {"chat_id": "chat-1", "assistant_message_id": "msg-1"}
    )

    assert drained is False
    assert output[0]["status"] == "completed"
    chats_stub.upsert_message_to_chat_by_id_and_message_id.assert_not_called()


async def test_approved_ask_user_call_is_released_for_the_client(
    middleware_module: ModuleType, chats_stub
) -> None:
    output = [
        {
            "type": "function_call",
            "call_id": "call-a",
            "name": "ask_user",
            "status": "queued",
            "approved": True,
        }
    ]
    chats_stub.get_message_by_id_and_message_id.return_value = {"output": output}

    with (
        patch.object(
            middleware_module, "get_event_emitter_and_caller", AsyncMock(return_value=(None, None))
        ),
        patch.object(middleware_module, "load_messages_from_db", AsyncMock(return_value=[])),
    ):
        drained = await drain(
            middleware_module, {"chat_id": "chat-1", "assistant_message_id": "msg-1"}
        )

    assert drained is True
    assert output[0]["status"] == "pending"
    assert "approved" not in output[0]
    chats_stub.upsert_message_to_chat_by_id_and_message_id.assert_awaited_once()
