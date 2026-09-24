"""A fresh chat message no longer pays a stored-message lookup for tool approvals.

Fix commit `9f680bb80` (PR #29142), `drain_approved_tool_calls` in `utils/middleware.py`. The
drain gated on `message_id` alone, which every ordinary send carries, so each new turn read a
chat message that does not exist yet and can hold no approvals. It now requires
`assistant_message_id`, which only a resume or continue payload sends.

Stays a unit test: the saved read is one database query with no effect on the reply, so no
response or stored state shows it. The chat store is a `create_autospec` stand-in, the only I/O
the drain reaches on these paths.

Discriminates: passes on bbfa876af; fails with `message_id` accepted in place of
`assistant_message_id` again (a fresh message id triggers the lookup).
"""

from __future__ import annotations

from unittest.mock import create_autospec, patch

import pytest

pytestmark = pytest.mark.regression

SAVED_CHAT_ID = "8e2b5f0c-2c3f-4d1e-9a77-0a1b2c3d4e5f"


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture
def chats(middleware_module):
    store = create_autospec(type(middleware_module.Chats), instance=True)
    store.get_message_by_id_and_message_id.return_value = None
    with patch.object(middleware_module, "Chats", store):
        yield store


async def _drain(middleware_module, metadata: dict) -> bool:
    return await middleware_module.drain_approved_tool_calls(
        request=None, form_data={"messages": []}, user=None, model={}, metadata=metadata
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"message_id": "fresh"},
        {"message_id": "fresh", "params": {"tool_approval_mode": "ask"}},
        {"message_id": "fresh", "assistant_message_id": ""},
        {},
    ],
    ids=["fresh-message", "ask-mode", "blank-assistant-id", "no-ids"],
)
async def test_a_fresh_turn_reads_no_stored_message(middleware_module, chats, metadata):
    assert await _drain(middleware_module, {"chat_id": SAVED_CHAT_ID, **metadata}) is False
    chats.get_message_by_id_and_message_id.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ids", "read"),
    [
        ({"assistant_message_id": "resumed"}, "resumed"),
        ({"assistant_message_id": "continued", "message_id": "holder"}, "holder"),
    ],
    ids=["resume", "continue"],
)
async def test_a_resumed_turn_reads_the_message_holding_the_output(
    middleware_module, chats, ids, read
):
    await _drain(middleware_module, {"chat_id": SAVED_CHAT_ID, **ids})

    chats.get_message_by_id_and_message_id.assert_awaited_once_with(SAVED_CHAT_ID, read)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["temporary:abc", "local:abc", "channel:abc", "", None])
async def test_an_unsaved_chat_never_reads_the_store(middleware_module, chats, chat_id):
    metadata = {"chat_id": chat_id, "assistant_message_id": "m-1", "message_id": "m-1"}

    assert await _drain(middleware_module, metadata) is False
    chats.get_message_by_id_and_message_id.assert_not_called()
