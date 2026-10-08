"""Regression: sub-agents and chats resuming after a tool approval lost the personal tool servers.

Fix `5d4f9b957` (open-webui/open-webui#31424, issue open-webui/open-webui#29893): a tool server a
user adds in their own settings (Open Terminal among them) reaches the server as `tool_servers`
in the chat request, each entry carrying its tool specs. Setting up the tools for the main model
popped those specs out of the request's own entries, so everything that later read the entries
again found no tools: a foreground sub-agent the model delegated to, and the continuation of a
chat after the user approved a tool call. Each now sees the same personal tools as the main
model.

The personal server's address is never called: its tools run in the browser, and only the specs
the model is offered are checked here.

`test_a_subagent_is_offered_the_personal_tools` is red on dev 62f70a844: since de73bb830 a chat
request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev efe63bd34; with 5d4f9b957 reverted in a backend copy the sub-agent
and the approval-resume tests fail (the personal tool is missing from those requests). The
main-model test passes on both.
"""

from __future__ import annotations

import time

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SUBAGENT_TASK = "Find out whether it rains in Graz tomorrow."
PERSONAL_TOOL = "lookup_weather"
PERSONAL_SERVER = {
    "url": "http://127.0.0.1:9/personal-tools",
    "key": "",
    "info": {"title": "My weather", "description": "A tool server on my own machine"},
    "specs": [
        {
            "name": PERSONAL_TOOL,
            "description": "Look up the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city."}},
                "required": ["city"],
            },
        }
    ],
}


def _switch_on(client, config: tuple[str, str], key: str) -> None:
    current = client.get(config[0]).json()
    client.post(config[1], json={**current, key: True}).raise_for_status()


@pytest.fixture
def subagents_enabled(preserve, admin):
    preserve(SUBAGENTS)
    with admin.client() as client:
        _switch_on(client, SUBAGENTS, "ENABLE_SUBAGENTS")


@pytest.fixture
def tool_approval_on(preserve, admin):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        _switch_on(client, CHAT_CONFIG, "ENABLE_TOOL_PERMISSIONS")


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _is_subagent(request: dict) -> bool:
    users = [entry for entry in request["messages"] if entry["role"] == "user"]
    return bool(users) and SUBAGENT_TASK in str(users[-1]["content"])


def test_the_main_model_is_offered_the_personal_tools(make_user, upstream):
    with make_user(role="admin").client() as client:
        ask(client, "what can you do?", tool_servers=[PERSONAL_SERVER])

    assert PERSONAL_TOOL in _offered(upstream.chat_requests()[-1])


def test_a_subagent_is_offered_the_personal_tools(subagents_enabled, make_user, upstream):
    upstream.queue(
        reply.tool_call("delegate_task", {"task": SUBAGENT_TASK}),
        reply.text("It stays dry.", match=_is_subagent),
        reply.text("The helper says it stays dry."),
    )
    with make_user(role="admin").client() as client:
        _, message = ask(client, "ask a helper about the weather", tool_servers=[PERSONAL_SERVER])

    [subagent_request] = [request for request in upstream.chat_requests() if _is_subagent(request)]
    main_request = upstream.chat_requests()[0]
    assert PERSONAL_TOOL in _offered(main_request)
    assert PERSONAL_TOOL in _offered(subagent_request), (
        "the sub-agent was offered none of the personal tool server's tools the main model had "
        "(#29893)"
    )
    assert message["content"].endswith("The helper says it stays dry.")


def test_a_chat_resumed_after_an_approval_keeps_the_personal_tools(
    tool_approval_on, make_user, upstream
):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with make_user(role="admin").client() as client:
        turn = send_message(
            client,
            "what time is it?",
            params={"tool_approval_mode": "ask"},
            tool_servers=[PERSONAL_SERVER],
        )
        call_id = _pending_call_id(client, turn)
        resolved = client.post(
            f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
            json={"call_id": call_id, "action": "approve"},
        )
        assert resolved.status_code == 200, resolved.text
        wait_for_reply(client, turn)

    first, resumed = upstream.chat_requests()[0], upstream.chat_requests()[-1]
    assert resumed["messages"][-1]["role"] == "tool"
    assert PERSONAL_TOOL in _offered(first)
    assert PERSONAL_TOOL in _offered(resumed), (
        "after the approval the chat went on without the personal tool server's tools (#29893)"
    )


def _pending_call_id(client, turn) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
        pending = [item for item in output if item.get("status") == "pending"]
        if pending:
            return pending[0]["call_id"]
        time.sleep(0.1)
    raise AssertionError("the tool call never waited for approval")
