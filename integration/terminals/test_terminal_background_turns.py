"""Journey: the turns Open WebUI starts by itself in a chat keep the chat's terminal.

A chat with a terminal picked hands that terminal on to what the model sets off: a sub-agent it
gives a task to, in the foreground or in the background, the turn that brings a background
sub-agent's report back into the chat, and a timer. Each of them is offered the terminal's tools
and its commands reach the terminal. A background sub-agent runs with no page open, so it loses
the tools that run in the person's browser (a tool server from their own settings) and keeps the
terminal the server reaches itself. A report that comes back while the chat is still writing
waits for that reply and then starts its turn with the terminal of the reply it waited for.

The terminal is the fake of `harness.terminal_server`, connected by the admin as a plain
terminal. `test_terminals_background_runs.py` has timers and sub-agents on a terminal of a
Terminals orchestrator, and `integration/tools/test_personal_tools_in_subagents.py` the
personal tools of a foreground sub-agent.

Discriminates: passes on dev c15f28649; each mutation below, made in a backend copy, turns its
tests red. The terminal left out of what a delegation hands its sub-agent fails both sub-agent
tests and the report to an idle chat (the report's turn takes it from there); a background
sub-agent keeping the browser's tool servers fails the background case alone. The terminal
left out of what a finished reply passes to a waiting report fails the still-writing case
alone, left out of the report's turn both report cases. The terminal left out of what a timer
keeps fails the timer test.
"""

from __future__ import annotations

import threading
import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat import ask, send_message, wait_for_reply
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TURN_WAIT = 60.0
PRINTED = "three boats in"
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {
                "operationId": "run_command",
                "description": "Run a shell command.",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}
PERSONAL_TOOL = "lookup_weather"
PERSONAL_SERVER = {
    "url": "http://127.0.0.1:9/personal-tools",
    "key": "",
    "info": {"title": "My weather", "description": "A tool server on my own machine"},
    "specs": [
        {
            "name": PERSONAL_TOOL,
            "description": "Look up the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
}


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("POST", "/execute", json_answer({"output": [{"data": PRINTED}]}))
        yield server


@pytest.fixture
def subagents_on(admin, preserve):
    """Sub-agents, in the background as well, and timers, which the first switch enables too."""
    preserve(SUBAGENTS)
    switches = {"ENABLE_SUBAGENTS": True, "SUBAGENTS_BACKGROUND_ENABLED": True}
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, **switches}).raise_for_status()


def _connect(admin, server: FakeTerminalServer, *people) -> str:
    connection = server.connection(
        config={"access_grants": [read_grant(person.id) for person in people]}
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    server.clear()
    return connection["id"]


def _run(**options) -> reply.Reply:
    return reply.tool_call("run_command", {"command": "ls", "wait": 10}, "command", **options)


def _report_of(task: str):
    """A `match` for the turn that brings the finished sub-agent's report back to the chat."""

    def matches(request: dict) -> bool:
        users = [entry for entry in request.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def _handed(task: str):
    """A `match` for the sub-agent's own requests."""
    return lambda request: reply.answering(task)(request) and not _report_of(task)(request)


def _requests(upstream, match) -> list[dict]:
    return [request for request in upstream.chat_requests() if match(request)]


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _told(request: dict) -> str:
    results = [entry["content"] for entry in conversation(request) if entry["role"] == "tool"]
    assert results, f"no tool ran; the model was sent {conversation(request)}"
    return results[-1]


def _report_waiting_in(client, chat_id: str) -> bool:
    """Whether a sub-agent's report is stored in the chat, waiting for the reply to end."""
    chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    return any(
        (message.get("meta") or {}).get("type") == "subagent"
        for message in chat["history"]["messages"].values()
    )


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
def test_a_sub_agent_keeps_the_terminal_and_in_the_background_loses_the_browsers_tools(
    background, subagents_on, admin, make_user, terminal, upstream, direct_integrations
):
    direct_integrations(True)
    # tool servers of one's own need the direct tool servers permission, which admins have
    person = make_user(role="admin")
    terminal_id = _connect(admin, terminal, person)
    tag = uuid.uuid4().hex[:8]
    prompt, task = f"hand this over {tag}", f"count the boats {tag}"
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": task, "background": background}, match=reply.answering(prompt)
        ),
        reply.text("Handed over.", match=reply.answering(prompt)),
        _run(match=_handed(task)),
        reply.text("Counted.", match=_handed(task)),
        reply.text("Noted.", match=_report_of(task)),
    )

    with person.client() as client:
        ask(client, prompt, terminal_id=terminal_id, tool_servers=[PERSONAL_SERVER])
    counted = wait_until(lambda: len(_requests(upstream, _handed(task))) == 2, TURN_WAIT)

    delegated = _requests(upstream, _handed(task))
    assert counted, f"the sub-agent never came back from its command: {delegated}"
    [asked] = _requests(upstream, reply.answering(prompt))[:1]
    assert {"run_command", PERSONAL_TOOL} <= _offered(asked), (
        f"the chat itself was not offered the terminal and the personal tool: {_offered(asked)}"
    )
    assert "run_command" in _offered(delegated[0]), (
        f"the sub-agent was not offered the chat's terminal: {sorted(_offered(delegated[0]))}"
    )
    assert PRINTED in _told(delegated[-1]), "the sub-agent's command did not reach the terminal"
    assert (PERSONAL_TOOL in _offered(delegated[0])) is (not background), (
        f"a {'background' if background else 'foreground'} sub-agent was offered "
        f"{sorted(_offered(delegated[0]))}"
    )
    assert len(terminal.requests_to("/execute")) == 1, (
        f"the terminal ran {len(terminal.requests_to('/execute'))} commands for one call"
    )


@pytest.mark.parametrize("chat", ["idle", "still writing"])
def test_the_turn_a_background_report_starts_keeps_the_chats_terminal(
    chat, subagents_on, admin, make_user, terminal, upstream
):
    person = make_user()
    terminal_id = _connect(admin, terminal, person)
    tag = uuid.uuid4().hex[:8]
    prompt, task = f"hand this over {tag}", f"count the boats {tag}"
    release = threading.Event()
    waits_for_the_report = {"hold": release} if chat == "still writing" else {}
    waits_for_the_reply = {"hold": release} if chat == "idle" else {}
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": task, "background": True}, match=reply.answering(prompt)
        ),
        reply.text("Handed over.", match=reply.answering(prompt), **waits_for_the_report),
        reply.text("Counted.", match=_handed(task), **waits_for_the_reply),
        _run(match=_report_of(task)),
        reply.text("Noted.", match=_report_of(task)),
    )

    try:
        with person.client() as client:
            turn = send_message(client, prompt, terminal_id=terminal_id)
            if chat == "still writing":
                arrived = wait_until(lambda: _report_waiting_in(client, turn.chat_id), TURN_WAIT)
                assert arrived, "the report never arrived while the chat was writing"
            else:
                wait_for_reply(client, turn)
            before_release = len(_requests(upstream, _report_of(task)))
            release.set()
            wait_for_reply(client, turn)
            noted = wait_until(lambda: len(_requests(upstream, _report_of(task))) == 2, TURN_WAIT)
    finally:
        release.set()

    reported = _requests(upstream, _report_of(task))
    assert before_release == 0, "the report's turn started before the test let it"
    assert noted, f"the report's turn never came back from its command: {reported}"
    assert "run_command" in _offered(reported[0]), (
        f"the report's turn was not offered the chat's terminal: {sorted(_offered(reported[0]))}"
    )
    assert PRINTED in _told(reported[-1]), "the report turn's command did not reach the terminal"
    [call] = terminal.requests_to("/execute")
    assert call.headers.get("x-session-id") == turn.chat_id, (
        f"the command was not run for the chat: {call.headers.get('x-session-id')}"
    )


def test_a_timer_fires_with_the_terminal_of_its_chat(
    subagents_on, admin, make_user, terminal, upstream
):
    person = make_user()
    terminal_id = _connect(admin, terminal, person)
    tag = uuid.uuid4().hex[:8]
    prompt, ring = f"wake me at high tide {tag}", f"High tide {tag}."
    upstream.queue(
        reply.tool_call("timer", {"prompt": ring, "at": "2s"}, match=reply.answering(prompt)),
        reply.text("Timer set.", match=reply.answering(prompt)),
        _run(match=reply.answering(ring)),
        reply.text("Rang.", match=reply.answering(ring)),
    )

    with person.client() as client:
        turn, message = ask(client, prompt, terminal_id=terminal_id)
    # due after two seconds and polled every second
    fired = wait_until(lambda: len(_requests(upstream, reply.answering(ring))) == 2, TURN_WAIT)

    rung = _requests(upstream, reply.answering(ring))
    assert message.get("content", "").endswith("Timer set."), f"no timer was set: {message}"
    assert fired, f"the timer never came back from its command: {rung}"
    assert "run_command" in _offered(rung[0]), (
        f"the timer's turn was not offered the chat's terminal: {sorted(_offered(rung[0]))}"
    )
    assert PRINTED in _told(rung[-1]), "the timer's command did not reach the terminal"
    [call] = terminal.requests_to("/execute")
    assert call.headers.get("x-session-id") == turn.chat_id, (
        f"the timer's command was not run for its chat: {call.headers.get('x-session-id')}"
    )
