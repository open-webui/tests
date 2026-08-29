"""Regression coverage for open-webui/open-webui#26257.

Continuing an existing assistant response must preserve its content and output
while restoring a missing parent link. The continuation guard previously
skipped persistence entirely, leaving an orphaned assistant message.

Discriminates: passes on fix/assistant-parent-id, fails on upstream/dev because the
assistant receives no parent-link update.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture
def main_module(owui_module):
    return owui_module("open_webui.main")


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_parent", [None, "user-message-id"])
async def test_continuation_repairs_orphaned_assistant_parent(
    main_module, owui_module, monkeypatch, existing_parent
):
    """An orphan is repaired while a correctly linked response is left alone."""
    assistant_message_id = "assistant-message-id"
    user_message_id = "user-message-id"
    chat_id = "chat-id"

    messages = {
        user_message_id: {"id": user_message_id, "role": "user", "childrenIds": []},
        assistant_message_id: {
            "id": assistant_message_id,
            "parentId": existing_parent,
            "role": "assistant",
            "content": "existing response",
            "output": [{"type": "message", "content": "existing response"}],
            "done": False,
        },
    }

    async def get_message_by_id_and_message_id(requested_chat_id, message_id):
        assert requested_chat_id == chat_id
        return messages.get(message_id)

    async def create_task(redis, process, *, id, task_id):
        process.close()
        return task_id, None

    upsert_message = AsyncMock()
    monkeypatch.setattr(main_module.Models, "get_model_by_id", AsyncMock(return_value=None))
    monkeypatch.setattr(main_module, "check_model_access", AsyncMock())
    monkeypatch.setattr(
        main_module.Config, "get", AsyncMock(side_effect=lambda key, default=None: default)
    )
    monkeypatch.setattr(main_module.Chats, "is_chat_owner", AsyncMock(return_value=True))
    monkeypatch.setattr(main_module.Chats, "update_chat_by_id", AsyncMock())
    monkeypatch.setattr(main_module.Chats, "update_chat_variables_by_id", AsyncMock())
    monkeypatch.setattr(
        main_module.Chats,
        "get_message_by_id_and_message_id",
        get_message_by_id_and_message_id,
    )
    monkeypatch.setattr(
        main_module.Chats, "upsert_message_to_chat_by_id_and_message_id", upsert_message
    )
    monkeypatch.setattr(main_module, "emit_chat_list_event", AsyncMock())
    monkeypatch.setattr(main_module, "publish_event", AsyncMock())
    monkeypatch.setattr(main_module, "create_task", create_task)
    monkeypatch.setattr(main_module, "get_event_emitter", AsyncMock(return_value=None))

    timers = owui_module("open_webui.utils.timers")
    monkeypatch.setattr(timers, "cancel_timers_for_chat", AsyncMock())

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                MODELS={"chat-model": {"id": "chat-model"}},
                redis=None,
            )
        ),
        headers={"user-agent": "test"},
        state=SimpleNamespace(internal=False),
    )
    user = SimpleNamespace(id="user-id", role="admin")
    form_data = {
        "model": "chat-model",
        "chat_id": chat_id,
        "message_ids": [
            {"model_id": "chat-model", "message_id": assistant_message_id, "modelIdx": 0}
        ],
        "assistant_message_id": assistant_message_id,
        "session_id": "test-session",
        "chat_variables": {},
        "user_message": {
            "id": user_message_id,
            "parentId": None,
            "childrenIds": [],
            "role": "user",
            "content": "hello",
        },
        "params": {},
    }

    await main_module.chat_completion(request, form_data, user)

    assistant_writes = [
        call.args[2]
        for call in upsert_message.await_args_list
        if call.args[1] == assistant_message_id
    ]
    expected_writes = [{"parentId": user_message_id}] if existing_parent is None else []
    assert assistant_writes == expected_writes
