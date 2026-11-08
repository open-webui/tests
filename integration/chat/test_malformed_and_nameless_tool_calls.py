"""Chat-pipeline fixes around failed turns and broken tool calls.

* `dbb17a572`: an assistant turn that failed with nothing written was replayed to the model as
  an empty assistant message on the next turn. Such rows are now dropped from the replay.
* `fed94c9f5`: tool-call arguments that parse to something other than an object (a list, a
  string) crashed the reply. The tool now answers with an `Error: ` result saying the arguments
  must be a JSON object (a "complete" one for a string, since dev d87837372) and the model is
  called again.
* `66e021a92` (PR #29690, issue #29686): a streamed tool call whose `function.name` was JSON null
  kept the null, which was stored and sent back for the provider to reject. It is now `""`, so
  the call fails once with the ordinary unknown-tool error.
* `c384d8e51` (PR #33779, issue #33778): a streamed call carrying two argument objects glued
  together ran as two calls, but the original call also stayed in the reply with no result,
  showing an extra unfinished call. The first part now keeps the original call's id.
* `d87837372` (0.13.1, issue #33868, titled `refac`): tool calls are read more forgivingly.
  Arguments wrapped in a code fence or encoded twice run, a number or a boolean sent as text
  runs as the number or boolean the tool declares (and a number sent for text as text), and a
  reply made only of `<tool_call>` or `<function=...>` markup naming an offered tool runs as that
  call. Markup next to prose, or naming a tool the chat does not offer, stays text.

Twin of unit/chat/test_malformed_and_nameless_tool_calls.py.

Discriminates: reverting dbb17a572 replays the empty failed turn; reverting fed94c9f5 ends the
reply in an error ("'list' object has no attribute 'items'") with no tool result; reverting
66e021a92 stores and replays the name as null; reverting c384d8e51 on dev f659c0d12 stores
three calls, one of them unanswered. Each revert fails only its own tests; the nearby tests pass
on all of them. On dev e8c0ad85e in a backend copy: the code fence left in place fails the fence
test, no second decode fails the twice-encoded test, no coercion by the tool's schema fails the
text-values test and no recovery of markup from the reply text fails both markup tests; the
prose and unoffered-tool controls pass on all of them.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.python_tools import python_tool
from harness.tool_calls import without_hidden_calls
from harness.upstream import Reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def tool_results(message: dict) -> list[str]:
    return [
        item["output"][0]["text"]
        for item in without_hidden_calls(message.get("output") or [])
        if item["type"] == "function_call_output"
    ]


def replayed(upstream) -> list[dict]:
    return reply.conversation(upstream.chat_requests()[-1])


def test_an_empty_failed_turn_is_not_replayed(user, upstream):
    upstream.queue(reply.error(500, "quota exceeded"), reply.text("fine now"))
    with user.client() as client:
        failed_turn, failed = ask(client, "first question")
        ask(
            client,
            "second question",
            chat_id=failed_turn.chat_id,
            parent_id=failed_turn.assistant_message_id,
        )

    assert failed["error"] and not failed["content"]
    assert [(entry["role"], entry["content"]) for entry in replayed(upstream)] == [
        ("user", "first question"),
        ("user", "second question"),
    ]


def test_a_failed_turn_with_partial_text_is_still_replayed(user, upstream):
    with user.client() as client:
        chat_id, failed_id = seed_chat(
            client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "partial answer", "error": {"content": "cut off"}},
            ],
        )
        ask(client, "go on", chat_id=chat_id, parent_id=failed_id)

    roles_and_text = [(entry["role"], entry["content"]) for entry in replayed(upstream)]
    assert ("assistant", "partial answer") in roles_and_text


@pytest.mark.parametrize("arguments", ["[1, 2, 3]", '"just a string"'])
def test_non_object_arguments_come_back_as_an_error_and_the_turn_goes_on(user, upstream, arguments):
    upstream.queue(reply.tool_call("get_current_timestamp", arguments), reply.text("let me retry"))
    with user.client() as client:
        _, message = ask(client, "what time is it?")

    results = tool_results(message)
    assert len(results) == 1 and results[0].startswith("Error: ") and "JSON object" in results[0], (
        message
    )
    tool_messages = [entry["content"] for entry in replayed(upstream) if entry["role"] == "tool"]
    assert tool_messages == results
    assert message["content"] == "let me retry"


@pytest.mark.parametrize("arguments", ["{}", ""])
def test_object_and_empty_arguments_still_run_the_tool(user, upstream, arguments):
    upstream.queue(reply.tool_call("get_current_timestamp", arguments), reply.text("it is late"))
    with user.client() as client:
        _, message = ask(client, "what time is it?")

    [result] = tool_results(message)
    assert "current_timestamp" in result
    assert message["content"] == "it is late"


def test_a_tool_call_with_a_null_name_is_stored_and_replayed_as_empty(user, upstream):
    nameless = {"id": "call_1", "type": "function", "function": {"name": None, "arguments": "{}"}}
    upstream.queue(Reply(tool_calls=[nameless]), reply.text("sorry"))
    with user.client() as client:
        _, message = ask(client, "do something")

    calls = [
        item for item in without_hidden_calls(message["output"]) if item["type"] == "function_call"
    ]
    assert [call["name"] for call in calls] == [""]
    [result] = tool_results(message)
    assert "not found" in result
    replayed_calls = [
        call["function"]["name"]
        for entry in replayed(upstream)
        for call in entry.get("tool_calls") or []
    ]
    assert replayed_calls == [""]


def test_a_call_carrying_two_argument_sets_is_stored_as_two_answered_calls(user, upstream):
    # GPT-5.4 glues two argument objects into one streamed call (issue #33778)
    upstream.queue(reply.tool_call("get_current_timestamp", "{}{}"), reply.text("both ran"))
    with user.client() as client:
        _, message = ask(client, "what time is it, twice?")

    output = without_hidden_calls(message["output"])
    calls = [item["call_id"] for item in output if item["type"] == "function_call"]
    answered = [item["call_id"] for item in output if item["type"] == "function_call_output"]
    assert len(calls) == 2 and sorted(answered) == sorted(calls), (
        f"the glued call was stored as calls {calls} with results for {answered}"
    )
    assert "call_1" in calls, f"the original call id was replaced: {calls}"
    assert all("current_timestamp" in result for result in tool_results(message)), message
    assert message["content"] == "both ran"


# --- forgiving argument reading (#33868) ---------------------------------------------------------

CATCH_TOOL_SOURCE = '''
import json


class Tools:
    def log_catch(self, fish: str, count: int, weight: float, released: bool) -> str:
        """Log a catch in the harbour book.

        :param fish: The kind of fish.
        :param count: How many were caught.
        :param weight: The total weight in kilograms.
        :param released: Whether they were let go again.
        """
        return json.dumps({"fish": fish, "count": count, "weight": weight, "released": released})
'''

CATCH = {"fish": "cod", "count": 2, "weight": 1.5, "released": False}


@pytest.fixture(scope="module")
def catch_tool(admin):
    with python_tool(admin, CATCH_TOOL_SOURCE, name="Harbour book") as tool_id:
        yield tool_id


def logged_catch(user, upstream, catch_tool: str, first_reply: Reply) -> tuple[dict, list[str]]:
    """The stored reply and the tool results after the model answered first with `first_reply`."""
    prompt = f"log my catch ({uuid.uuid4().hex[:8]})"
    first_reply.match = reply.answering(prompt)
    upstream.queue(first_reply, reply.text("logged", match=reply.answering(prompt)))
    with user.client() as client:
        _, message = ask(client, prompt, tool_ids=[catch_tool])
    return message, tool_results(message)


def test_arguments_wrapped_in_a_code_fence_run_the_tool(user, upstream, catch_tool):
    fenced = f"```json\n{json.dumps(CATCH)}\n```"
    _, results = logged_catch(user, upstream, catch_tool, reply.tool_call("log_catch", fenced))

    assert [json.loads(result) for result in results] == [CATCH], f"the fenced call: {results}"


def test_arguments_encoded_twice_run_the_tool(user, upstream, catch_tool):
    twice = json.dumps(json.dumps(CATCH))
    _, results = logged_catch(user, upstream, catch_tool, reply.tool_call("log_catch", twice))

    assert [json.loads(result) for result in results] == [CATCH], f"twice encoded: {results}"


def test_numbers_and_booleans_sent_as_text_run_as_what_the_tool_declares(
    user, upstream, catch_tool
):
    as_text = {"fish": 7, "count": "2", "weight": "1.5", "released": "true"}
    _, results = logged_catch(user, upstream, catch_tool, reply.tool_call("log_catch", as_text))

    assert [json.loads(result) for result in results] == [
        {"fish": "7", "count": 2, "weight": 1.5, "released": True}
    ], f"the text values reached the tool as: {results}"


@pytest.mark.parametrize(
    "markup",
    [
        f"<tool_call>{json.dumps({'name': 'log_catch', 'arguments': CATCH})}</tool_call>",
        "<function=log_catch><parameter=fish>cod</parameter><parameter=count>2</parameter>"
        "<parameter=weight>1.5</parameter><parameter=released>false</parameter></function>",
    ],
    ids=["tool_call", "function"],
)
def test_a_reply_written_as_tool_call_markup_runs_the_call(markup, user, upstream, catch_tool):
    message, results = logged_catch(user, upstream, catch_tool, reply.text(markup))

    assert [json.loads(result) for result in results] == [CATCH], (
        f"the markup was not run as a call: {message['output']}"
    )
    calls = [
        item for item in without_hidden_calls(message["output"]) if item["type"] == "function_call"
    ]
    assert [call["name"] for call in calls] == ["log_catch"]
    assert message["content"] == "logged", f"the markup stayed in the answer: {message['content']}"


@pytest.mark.parametrize(
    "markup",
    [
        "Here is how I would call it: "
        f"<tool_call>{json.dumps({'name': 'log_catch', 'arguments': CATCH})}</tool_call>",
        f"<tool_call>{json.dumps({'name': 'sink_boat', 'arguments': {}})}</tool_call>",
    ],
    ids=["beside_prose", "unoffered_tool"],
)
def test_markup_beside_prose_or_for_a_tool_not_offered_stays_text(
    markup, user, upstream, catch_tool
):
    prompt = f"show me the markup ({uuid.uuid4().hex[:8]})"
    upstream.queue(reply.text(markup, match=reply.answering(prompt)))
    with user.client() as client:
        _, message = ask(client, prompt, tool_ids=[catch_tool])

    assert tool_results(message) == [], f"text markup was run: {message['output']}"
    assert message["content"] == markup
