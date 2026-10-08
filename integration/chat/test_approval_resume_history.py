"""Approving a tool call sends the provider the chat as it stood plus the one paused response.

Commit 639139aa7 (`utils/middleware.py`): the resume of a paused response used to rebuild the
chat history from the database after the approved tools ran. It now builds the history once,
compacts it, takes the paused assistant message out of it and appends that message again after
the tools ran. The follow-up must carry each earlier turn once, the paused call and its result
once, and after a compaction the summary and the kept turns only.

Discriminates: nearby tests that pass before and after the commit; in a backend copy that skips
the append of the paused message after the tools ran, both fail (the follow-up never carries the
call and its result).
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message
from harness.chat_history import seed_chat

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SUMMARY = "SUMMARY OF THE EARLIER TURNS"


@pytest.fixture
def chat_settings(admin, preserve):
    """`chat_settings(**changes)` updates the admin chat settings for the rest of the test."""
    preserve(CHAT_CONFIG)

    def change(**changes) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            client.post(CHAT_CONFIG[1], json={**current, **changes}).raise_for_status()

    return change


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def turns(count: int, filler: int = 400) -> list[dict]:
    messages = []
    for number in range(1, count + 1):
        messages.append({"role": "user", "content": f"question {number} " + "q" * filler})
        messages.append({"role": "assistant", "content": f"answer {number} " + "a" * filler})
    return [{**message, "id": str(uuid.uuid4())} for message in messages]


def approve_the_pending_call(client: httpx.Client, turn: ChatTurn) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
        pending = [item for item in output if item.get("status") == "pending"]
        if pending:
            resolved = client.post(
                f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
                json={"call_id": pending[0]["call_id"], "action": "approve"},
            )
            assert resolved.status_code == 200, resolved.text
            return
        time.sleep(0.1)
    raise AssertionError("the tool call never waited for approval")


def follow_up_after_approval(upstream) -> list[dict]:
    """The messages of the model call that carries the approved call's result."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for body in upstream.chat_requests():
            if body.get("stream") and body["messages"][-1]["role"] == "tool":
                return body["messages"]
        time.sleep(0.2)
    raise AssertionError("the model was never called again after the approval")


def roles(messages: list[dict]) -> list[str]:
    return [message["role"] for message in messages]


def test_the_follow_up_repeats_no_earlier_turn_and_the_paused_call_once(
    chat_settings, make_user, upstream
):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(
        reply.text("Hello there."),
        reply.tool_call("get_current_timestamp", {}),
        reply.text("It is late."),
    )
    with make_user().client() as client:
        first, _ = ask(client, "hello")
        turn = send_message(
            client,
            "what time is it?",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            params={"tool_approval_mode": "ask"},
        )
        approve_the_pending_call(client, turn)
        sent = follow_up_after_approval(upstream)

    assert roles(sent) == ["user", "assistant", "user", "assistant", "tool"]
    assert [m["content"] for m in sent[:3]] == ["hello", "Hello there.", "what time is it?"]
    assert [call["id"] for call in sent[3]["tool_calls"]] == ["call_1"]
    assert sent[4]["tool_call_id"] == "call_1"


def test_the_follow_up_after_a_compaction_carries_the_summary_and_the_kept_turns(
    chat_settings, make_user, upstream
):
    chat_settings(
        ENABLE_TOOL_PERMISSIONS=True,
        ENABLE_CONTEXT_COMPACTION=True,
        CONTEXT_COMPACTION_TOKEN_THRESHOLD=50,
        CONTEXT_COMPACTION_TOKEN_CAP=50,
        CONTEXT_COMPACTION_RETENTION_PERCENTAGE=50,
    )
    upstream.queue(
        reply.text(SUMMARY, match=is_summary_request),
        reply.text(SUMMARY, match=is_summary_request),
        reply.tool_call("get_current_timestamp", {}),
        reply.text("It is late."),
    )
    history = turns(4)
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, history)
        turn = send_message(
            client,
            "what time is it?",
            chat_id=chat_id,
            parent_id=last_id,
            history=[{"role": "system", "content": "You are a careful travel agent."}],
            params={"tool_approval_mode": "ask"},
        )
        approve_the_pending_call(client, turn)
        sent = follow_up_after_approval(upstream)

    assert roles(sent) == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in sent[0]["content"]
    assert [m["content"][:10] for m in sent[1:4]] == ["question 4", "answer 4 a", "what time "]
    assert [call["id"] for call in sent[4]["tool_calls"]] == ["call_1"]
