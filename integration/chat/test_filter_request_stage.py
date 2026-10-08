"""The `request` filter stage, which runs right before every model call.

Fix commit `2daa610cb` in `utils/middleware.py`. Filters could only see the payload at `inlet`,
before retrieval, tool definitions, the system prompt and message normalisation were added, so
no plugin could inspect or edit what the model was actually asked. A `request` stage now runs
before every model call: the first one, the continuation after a tool call, and the follow-up
that `drain_approved_tool_calls` makes once the user approves a tool call.

Commit 639139aa7 moved the resume of approved calls into `process_chat_payload`, so the request
stage must also run before those tools do: a filter that refuses the resumed request has to stop
the approved tool from running (nearby test, it passes before and after the commit).

Twin of unit/chat/test_filter_request_stage.py.

Discriminates: with the request filter removed from the resume step the refusal test fails (the
tool runs first); with 2daa610cb reverted the three request-stage tests fail (no `request` hook
ever runs); removing only the hook in `drain_approved_tool_calls` fails only the approval test.
The inlet tests pass on both.
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message, wait_for_reply
from harness.plugins import installed_function
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")

# Stamps the last message at each stage with whether the payload already carries the tools.
STAGE_STAMPS = """
class Filter:
    async def inlet(self, body):
        body["messages"][-1]["content"] += f" [inlet tools={'tools' in body}]"
        return body

    async def request(self, body):
        body["messages"][-1]["content"] += f" [request tools={'tools' in body}]"
        return body
"""


def model_calls(upstream) -> list[list[dict]]:
    return [request["messages"] for request in upstream.chat_requests()]


def test_the_request_stage_sees_the_assembled_payload(admin, user, upstream):
    with installed_function(admin, STAGE_STAMPS, is_global=True), user.client() as client:
        ask(client, "hello")

    [messages] = model_calls(upstream)
    assert messages[-1]["content"] == "hello [inlet tools=False] [request tools=True]"


def test_the_request_stage_runs_again_after_a_tool_call(admin, user, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with installed_function(admin, STAGE_STAMPS, is_global=True), user.client() as client:
        _, message = ask(client, "what time is it?")

    first_call, continuation = model_calls(upstream)
    assert first_call[-1]["content"].endswith("[request tools=True]")
    assert continuation[-1]["role"] == "tool"
    assert continuation[-1]["content"].endswith("[request tools=True]")
    assert message["content"] == "It is late."


def pending_tool_call(client: httpx.Client, turn: ChatTurn) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
        pending = [item for item in output if item.get("status") == "pending"]
        if pending:
            return pending[0]
        time.sleep(0.1)
    raise AssertionError("the tool call never waited for approval")


def settled_message(client: httpx.Client, turn: ChatTurn) -> dict:
    """The stored reply once it is done or carries an error."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        message = chat["history"]["messages"][turn.assistant_message_id]
        if message.get("error") or message.get("done"):
            return message
        time.sleep(0.2)
    raise AssertionError("the resumed reply never settled")


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


def test_the_request_stage_runs_before_the_call_after_an_approval(
    admin, user, upstream, tool_approval_on
):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with installed_function(admin, STAGE_STAMPS, is_global=True), user.client() as client:
        turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
        call = pending_tool_call(client, turn)
        resolved = client.post(
            f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
            json={"call_id": call["call_id"], "action": "approve"},
        )
        assert resolved.status_code == 200, resolved.text
        message = wait_for_reply(client, turn)

    follow_up = model_calls(upstream)[-1]
    assert follow_up[-1]["role"] == "tool"
    assert follow_up[-1]["content"].endswith("[request tools=True]")
    assert message["content"] == "It is late."


# only a resumed response carries assistant_message_id, so the first model call passes
REFUSE_A_RESUMED_RESPONSE = """
class Filter:
    async def request(self, body, __metadata__):
        if __metadata__.get("assistant_message_id"):
            raise RuntimeError("request refused by policy")
        return body
"""


def test_a_refusing_request_filter_stops_an_approved_tool_from_running(
    admin, user, upstream, tool_approval_on
):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with installed_function(admin, REFUSE_A_RESUMED_RESPONSE, is_global=True):
        with user.client() as client:
            turn = send_message(client, "what time is it?", params={"tool_approval_mode": "ask"})
            call = pending_tool_call(client, turn)
            client.post(
                f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
                json={"call_id": call["call_id"], "action": "approve"},
            )
            message = settled_message(client, turn)

    ran = [
        item for item in message.get("output") or [] if item.get("type") == "function_call_output"
    ]
    assert not ran, "the approved tool ran although the request filter refused the request"
    assert len(model_calls(upstream)) == 1, "the model was asked again after the filter refused"


INLET_ONLY = """
class Filter:
    async def inlet(self, body):
        body["messages"][-1]["content"] += " [inlet]"
        return body
"""

RAISING_INLET = """
class Filter:
    async def inlet(self, body):
        raise RuntimeError("inlet exploded")
"""


def test_an_inlet_only_filter_still_shapes_the_payload(admin, user, upstream):
    with installed_function(admin, INLET_ONLY, is_global=True), user.client() as client:
        ask(client, "hello")

    assert model_calls(upstream)[-1][-1]["content"] == "hello [inlet]"


def test_a_failing_inlet_still_stops_the_chat(admin, user, upstream):
    with installed_function(admin, RAISING_INLET, is_global=True), user.client() as client:
        refused = client.post(
            "/api/chat/completions",
            json={"model": MOCK_MODEL_ID, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert refused.status_code == 400
    assert "inlet exploded" in refused.text
    assert model_calls(upstream) == []
