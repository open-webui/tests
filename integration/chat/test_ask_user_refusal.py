"""A refused `ask_user` call ended the turn with no reply.

Fix commit `be958d7b0` (PR #29252, issue #29077) in `utils/ask_user.py` and
`utils/middleware.py`. When a model called `ask_user` in a shape Open WebUI refuses (twice in
one turn, next to another tool, or with arguments it cannot use), staging gave only the first
`ask_user` call an error result and the middleware then skipped the rest of the round, so the
model was never called again and the user was left with a dead turn. Every refused call now
gets its own error result, the other tool calls still run, and the model is called again with
all the results.

Since 794e57cf7 (#32727, issue #32662) questions sent as something other than a list, such as a
sentence, get their own error saying they must be an array of question objects, where
the model used to be told "ask_user requires 1-3 questions" about the right number of questions.
Since d87837372 (0.13.1) tool arguments are read leniently, so questions sent as JSON text are
decoded and asked as questions; questions as plain prose still get the array error.

Twin of unit/chat/test_ask_user_refusal.py.

Discriminates: with be958d7b0 reverted every refusal test fails (no second provider call, and
a second refused call gets no result); the accepted call stays pending on both. On dev f36490bf2
with 794e57cf7 reverted in a backend copy the prose-questions case fails (the model gets the
count error); a wrong count gets the count error on both. The JSON-text case passes on dev
da9bca0fc and fails on d87837372^ (the call is refused, nothing is asked).
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message
from harness.tool_calls import without_hidden_calls
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
        for item in without_hidden_calls(message["output"])
        if item["type"] == "function_call_output"
    }


def replayed_tool_messages(upstream, prompt: str) -> dict[str, str]:
    requests = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    assert len(requests) == 2, f"the model was called {len(requests)} time(s), not twice"
    return {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(requests[-1])
        if entry["role"] == "tool"
    }


def test_every_refused_ask_user_call_gets_a_result_and_the_turn_goes_on(make_user, upstream):
    prompt = "help me choose"
    two_questions = Reply(
        tool_calls=[
            ask_user_call("call_a", {"questions": [question()]}),
            ask_user_call("call_b", {"questions": [question()]}),
        ],
        match=reply.answering(prompt),
    )
    closing = reply.text("I will ask one question at a time.", match=reply.answering(prompt))
    upstream.queue(two_questions, closing)
    with make_user().client() as client:
        _, message = ask(client, prompt)

    results = tool_results(message)
    assert set(results) == {"call_a", "call_b"}, without_hidden_calls(message["output"])
    refused = all(text.startswith("Error:") and "did not run" in text for text in results.values())
    assert refused, f"a refused ask_user call got no refusal: {results}"
    assert replayed_tool_messages(upstream, prompt) == results
    assert message["content"] == "I will ask one question at a time."


def test_a_refused_ask_user_does_not_stop_the_other_tool_call(make_user, upstream):
    prompt = "what time is it, and which one?"
    mixed = Reply(
        tool_calls=[
            ask_user_call("call_ask", {"questions": [question()]}),
            timestamp_call("call_time"),
        ],
        match=reply.answering(prompt),
    )
    upstream.queue(mixed, reply.text("done", match=reply.answering(prompt)))
    with make_user().client() as client:
        _, message = ask(client, prompt)

    replayed = replayed_tool_messages(upstream, prompt)
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
def test_a_refusal_reaches_the_model_as_a_tool_result(make_user, upstream, arguments):
    prompt = "ask me something"
    upstream.queue(
        Reply(tool_calls=[ask_user_call("call_a", arguments)], match=reply.answering(prompt)),
        reply.text("retrying", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        _, message = ask(client, prompt)

    results = tool_results(message)
    assert list(results) == ["call_a"]
    assert results["call_a"].startswith("Error:")
    assert replayed_tool_messages(upstream, prompt) == results


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


def test_an_accepted_ask_user_call_waits_for_the_user(make_user, upstream):
    prompt = "ask me something"
    upstream.queue(
        reply.tool_call(
            "ask_user", {"questions": [question(header="Scope")]}, match=reply.answering(prompt)
        )
    )
    with make_user().client() as client:
        turn = send_message(client, prompt)
        try:
            call = pending_ask_user(client, turn)
            time.sleep(1)  # nothing more should reach the model while the question is open
            sent = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
        finally:
            # the open question would otherwise wait two minutes on the shared instance
            client.post(f"/api/tasks/chat/{turn.chat_id}/stop")

    arguments = json.loads(call["arguments"])
    assert call["name"] == "ask_user"
    assert arguments["timeout_ms"] == 120_000
    assert arguments["questions"][0]["header"] == "Scope"
    assert arguments["questions"][0]["options"][0] == {"label": "A", "description": "first"}
    assert len(sent) == 1, f"the model was called again while the question was open: {sent}"


QUESTIONS_AS_JSON_TEXT = {"questions": json.dumps([question(header="Harbour")])}
QUESTIONS_AS_PROSE = {"questions": "Which harbour, Trieste or Split?"}


def test_questions_sent_as_json_text_are_asked(make_user, upstream):
    prompt = "ask me which harbour, as text"
    upstream.queue(
        Reply(
            tool_calls=[ask_user_call("call_a", QUESTIONS_AS_JSON_TEXT)],
            match=reply.answering(prompt),
        )
    )
    with make_user().client() as client:
        turn = send_message(client, prompt)
        try:
            call = pending_ask_user(client, turn)
        finally:
            # the open question would otherwise wait two minutes on the shared instance
            client.post(f"/api/tasks/chat/{turn.chat_id}/stop")

    questions = json.loads(call["arguments"])["questions"]
    assert [entry["header"] for entry in questions] == ["Harbour"], (
        f"the questions sent as JSON text were not asked as questions: {questions}"
    )


@pytest.mark.parametrize(
    ("arguments", "told", "not_told"),
    [
        pytest.param(QUESTIONS_AS_PROSE, "array of question objects", "1-3 questions", id="string"),
        pytest.param(
            {"questions": [question(f"q{index}") for index in range(4)]},
            "1-3 questions",
            "array of question objects",
            id="four",
        ),
    ],
)
def test_the_refusal_names_what_is_wrong_with_the_questions(
    make_user, upstream, arguments, told, not_told
):
    """#32727: questions sent as a string were refused as the wrong count."""
    prompt = "ask me which harbour"
    upstream.queue(
        Reply(tool_calls=[ask_user_call("call_a", arguments)], match=reply.answering(prompt)),
        reply.text("retrying", match=reply.answering(prompt)),
    )
    with make_user().client() as client:
        ask(client, prompt)

    result = replayed_tool_messages(upstream, prompt)["call_a"]
    assert told in result and not_told not in result, f"the model was told: {result}"
