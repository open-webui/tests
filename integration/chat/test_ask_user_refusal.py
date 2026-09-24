"""A refused `ask_user` call ended the turn with no reply.

Fix commit `be958d7b0` (PR #29252, issue #29077) in `utils/ask_user.py` and
`utils/middleware.py`. When a model called `ask_user` in a shape Open WebUI refuses (twice in
one turn, next to another tool, or with arguments it cannot use), staging gave only the first
`ask_user` call an error result and the middleware then skipped the rest of the round, so the
model was never called again and the user was left with a dead turn. Every refused call now
gets its own error result, the other tool calls still run, and the model is called again with
all the results.

Twin of unit/chat/test_ask_user_refusal.py.

Discriminates: with be958d7b0 reverted every refusal test fails (no second provider call, and
a second refused call gets no result); the accepted call stays pending on both.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message
from harness.upstream import Reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def question(question_id: str = "q1", **overrides) -> dict:
    options = [{"label": "A", "description": "first"}, {"label": "B", "description": "second"}]
    return {"id": question_id, "question": "Which one?", "options": options, **overrides}


def ask_user_call(call_id: str, arguments: dict | str) -> dict:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ask_user", "arguments": encoded},
    }


def timestamp_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_current_timestamp", "arguments": "{}"},
    }


def tool_results(message: dict) -> dict[str, str]:
    return {
        item["call_id"]: item["output"][0]["text"]
        for item in message["output"]
        if item["type"] == "function_call_output"
    }


def replayed_tool_messages(upstream) -> dict[str, str]:
    requests = upstream.chat_requests()
    assert len(requests) == 2, f"the model was called {len(requests)} time(s), not twice"
    return {
        entry["tool_call_id"]: entry["content"]
        for entry in requests[-1]["messages"]
        if entry["role"] == "tool"
    }


def test_every_refused_ask_user_call_gets_a_result_and_the_turn_goes_on(user, upstream):
    two_questions = Reply(
        tool_calls=[
            ask_user_call("call_a", {"questions": [question()]}),
            ask_user_call("call_b", {"questions": [question()]}),
        ]
    )
    upstream.queue(two_questions, reply.text("I will ask one question at a time."))
    with user.client() as client:
        _, message = ask(client, "help me choose")

    results = tool_results(message)
    assert set(results) == {"call_a", "call_b"}, message["output"]
    assert all(text.startswith("Error:") and "did not run" in text for text in results.values())
    assert replayed_tool_messages(upstream) == results
    assert message["content"] == "I will ask one question at a time."


def test_a_refused_ask_user_does_not_stop_the_other_tool_call(user, upstream):
    mixed = Reply(
        tool_calls=[
            ask_user_call("call_ask", {"questions": [question()]}),
            timestamp_call("call_time"),
        ]
    )
    upstream.queue(mixed, reply.text("done"))
    with user.client() as client:
        _, message = ask(client, "what time is it, and which one?")

    replayed = replayed_tool_messages(upstream)
    assert set(replayed) == {"call_ask", "call_time"}
    assert replayed["call_ask"].startswith("Error:")
    assert not replayed["call_time"].startswith("Error:")
    assert message["content"] == "done"


def option_list(count: int) -> list[dict]:
    return [{"label": f"L{index}", "description": "d"} for index in range(count)]


REFUSED_ARGUMENTS = [
    pytest.param("{not json", id="malformed-json"),
    pytest.param("[]", id="not-an-object"),
    pytest.param({"questions": []}, id="no-questions"),
    pytest.param(
        {"questions": [question(f"q{index}") for index in range(4)]}, id="too-many-questions"
    ),
    pytest.param({"questions": [question(id="")]}, id="missing-question-id"),
    pytest.param({"questions": [question(options=option_list(1))]}, id="too-few-options"),
    pytest.param(
        {"questions": [question(options=[{"label": "A"}, {"label": "B", "description": "d"}])]},
        id="option-without-description",
    ),
    pytest.param({"questions": [question("q1"), question("q1")]}, id="duplicate-question-id"),
    pytest.param({"questions": ["not an object"]}, id="question-not-an-object"),
    pytest.param(
        {"questions": [question(options=["A", {"label": "B", "description": "d"}])]},
        id="option-not-an-object",
    ),
    pytest.param({"questions": [question(question="  ")]}, id="empty-question-text"),
]


@pytest.mark.parametrize("arguments", REFUSED_ARGUMENTS)
def test_a_refusal_reaches_the_model_as_a_tool_result(user, upstream, arguments):
    upstream.queue(Reply(tool_calls=[ask_user_call("call_a", arguments)]), reply.text("retrying"))
    with user.client() as client:
        _, message = ask(client, "ask me something")

    results = tool_results(message)
    assert list(results) == ["call_a"]
    assert results["call_a"].startswith("Error:")
    assert replayed_tool_messages(upstream) == results


def pending_ask_user(client: httpx.Client, turn: ChatTurn, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
        pending = [item for item in output if item.get("status") == "pending"]
        if pending:
            return pending[0]
        time.sleep(0.1)
    raise AssertionError("the ask_user call never reached the pending state")


def test_an_accepted_ask_user_call_waits_for_the_user(user, upstream):
    upstream.queue(reply.tool_call("ask_user", {"questions": [question(header="Scope")]}))
    with user.client() as client:
        turn = send_message(client, "ask me something")
        call = pending_ask_user(client, turn)

    arguments = json.loads(call["arguments"])
    assert call["name"] == "ask_user"
    assert arguments["timeout_ms"] == 120_000
    assert arguments["questions"][0]["header"] == "Scope"
    assert arguments["questions"][0]["options"][0] == {"label": "A", "description": "first"}
    time.sleep(1)  # nothing more should reach the model while the question is open
    assert len(upstream.chat_requests()) == 1
