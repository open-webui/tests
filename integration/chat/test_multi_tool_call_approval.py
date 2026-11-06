"""Regression: with tool approval on, only the first of several tool calls in one turn was asked.

Fix `35dda256f` (open-webui/open-webui#31315, issue open-webui/open-webui#29293): when the model
called several tools in one turn, the approval pause made the first call `pending` and queued
the others only while they were still streaming. By the time it ran every call was already
complete, so the second and later calls were neither pending nor queued: they never got an
approval card, never ran, answered "already resolved" when approved, and the model was called
again without their results. Every unanswered call is now queued and asked in turn.

Discriminates: passes on dev efe63bd34; with 35dda256f reverted in a backend copy both
multi-call tests fail (the second call never waits for approval). The single-call test passes on
both.
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

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


def _two_calls() -> reply.Reply:
    """One turn in which the model asks for the time twice."""
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "get_current_timestamp", "arguments": "{}"},
        }
        for call_id in ("call_1", "call_2")
    ]
    return reply.Reply(tool_calls=calls)


def _output(client: httpx.Client, turn: ChatTurn) -> list[dict]:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    return chat["history"]["messages"][turn.assistant_message_id].get("output") or []


def _wait_until_pending(client: httpx.Client, turn: ChatTurn, call_id: str) -> None:
    deadline = time.monotonic() + APPROVAL_WAIT
    statuses: dict = {}
    while time.monotonic() < deadline:
        calls = [item for item in _output(client, turn) if item.get("type") == "function_call"]
        statuses = {item.get("call_id"): item.get("status") for item in calls}
        if statuses.get(call_id) == "pending":
            return
        time.sleep(0.2)
    raise AssertionError(
        f"{call_id} never waited for approval; the calls stood at {statuses} (#29293)"
    )


def _resolve(client: httpx.Client, turn: ChatTurn, call_id: str, action: str) -> None:
    resolved = client.post(
        f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
        json={"call_id": call_id, "action": action},
    )
    assert resolved.status_code == 200, f"{action} {call_id}: {resolved.text}"


def _tool_results(upstream) -> list[str]:
    last = conversation(upstream.chat_requests()[-1])
    return [entry.get("tool_call_id") for entry in last if entry["role"] == "tool"]


def test_every_call_of_a_turn_is_asked_in_turn(tool_approval_on, make_user, upstream):
    upstream.queue(_two_calls(), reply.text("Checked twice."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        _wait_until_pending(client, turn, "call_1")
        _resolve(client, turn, "call_1", "approve")
        _wait_until_pending(client, turn, "call_2")
        _resolve(client, turn, "call_2", "approve")
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("Checked twice.")
    assert _tool_results(upstream) == ["call_1", "call_2"], (
        "the model was called again without the result of every approved call"
    )


def test_rejecting_the_first_call_still_asks_about_the_second(
    tool_approval_on, make_user, upstream
):
    upstream.queue(_two_calls(), reply.text("Checked once."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        _wait_until_pending(client, turn, "call_1")
        _resolve(client, turn, "call_1", "reject")
        _wait_until_pending(client, turn, "call_2")
        _resolve(client, turn, "call_2", "approve")
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("Checked once.")
    statuses = {
        item["call_id"]: item.get("status")
        for item in message.get("output") or []
        if item.get("type") == "function_call"
    }
    assert statuses.get("call_1") == "rejected"


def test_a_single_call_is_asked_and_runs_once_approved(tool_approval_on, make_user, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with make_user().client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        _wait_until_pending(client, turn, "call_1")
        _resolve(client, turn, "call_1", "approve")
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("It is late.")
    assert _tool_results(upstream) == ["call_1"]
