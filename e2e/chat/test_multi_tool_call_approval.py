"""Regression: with tool approval on, only the first of several calls in a turn could be allowed.

Fix `35dda256f` (open-webui/open-webui#31315, issue open-webui/open-webui#29293): when the model
called two tools in one turn, only the first got Allow and Deny buttons; the second stayed on
"Executing..." for good, never ran, and the reply never finished. Every call now gets its own
approval in turn. Browser twin of integration/chat/test_multi_tool_call_approval.py. Runs in the
main chat and in a side chat (`utils/chat_surface.py`).

Discriminates: passes on dev efe63bd34; with 35dda256f reverted in a backend copy (on the clean
build) the test fails: after the first Allow the second call is never put up for approval.
"""

from __future__ import annotations

import time
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    conversation,
    expect_reply,
    send,
    set_tool_approval_mode,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
APPROVAL_WAIT = 20.0


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


def _wait_until_second_call_is_asked(account) -> None:
    """Wait for the server to put the second call up for approval, so the next Allow is its."""
    deadline = time.monotonic() + APPROVAL_WAIT
    statuses: dict = {}
    with account.client() as client:
        while time.monotonic() < deadline:
            [chat] = client.get("/api/v1/chats/").json()[:1]
            history = client.get(f"/api/v1/chats/{chat['id']}").json()["chat"]["history"]
            output = history["messages"][history["currentId"]].get("output") or []
            statuses = {
                item.get("call_id"): item.get("status")
                for item in output
                if item.get("type") == "function_call"
            }
            if statuses.get("call_2") == "pending":
                return
            time.sleep(0.2)
    raise AssertionError(
        f"after the first Allow the second call never asked for approval: {statuses} (#29293)"
    )


def test_both_calls_of_a_turn_are_allowed_one_after_the_other(
    tool_approval_on, chat_for, make_user, upstream
):
    question = f"what time is it twice? {uuid.uuid4().hex[:6]}"
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "get_current_timestamp", "arguments": "{}"},
        }
        for call_id in ("call_1", "call_2")
    ]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=reply.answering(question)),
        reply.text("Checked the clock twice.", match=reply.answering(question)),
    )
    account = make_user()
    chat = chat_for(account)
    set_tool_approval_mode(chat, "Ask for approval")

    send(chat, question)
    allow = conversation(chat).get_by_role("button", name="Allow", exact=True)
    expect(allow).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    allow.click()
    _wait_until_second_call_is_asked(account)
    expect(allow).to_be_enabled()
    allow.click()

    expect_reply(chat, "Checked the clock twice.")
    tool_results = [
        entry.get("tool_call_id")
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    ]
    assert tool_results == ["call_1", "call_2"]
