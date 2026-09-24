"""Three chat-pipeline fixes from 0.11.4 around failed turns and broken tool calls.

* `dbb17a572`: an assistant turn that failed with nothing written was replayed to the model as
  an empty assistant message on the next turn. Such rows are now dropped from the replay.
* `fed94c9f5`: tool-call arguments that parse to something other than an object (a list, a
  string) crashed the reply. The tool now answers with a "must be a JSON object" error and the
  model is called again.
* `66e021a92` (PR #29690, issue #29686): a streamed tool call whose `function.name` was JSON null
  kept the null, which was stored and sent back for the provider to reject. It is now `""`, so
  the call fails once with the ordinary unknown-tool error.

Twin of unit/chat/test_malformed_and_nameless_tool_calls.py.

Discriminates: reverting dbb17a572 replays the empty failed turn; reverting fed94c9f5 ends the
reply in an error ("'list' object has no attribute 'items'") with no tool result; reverting
66e021a92 stores and replays the name as null. Each revert fails only its own tests; the
nearby tests pass on all three.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.upstream import Reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def tool_results(message: dict) -> list[str]:
    return [
        item["output"][0]["text"]
        for item in message.get("output") or []
        if item["type"] == "function_call_output"
    ]


def replayed(upstream) -> list[dict]:
    return upstream.chat_requests()[-1]["messages"]


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
    assert len(results) == 1 and "must be a JSON object" in results[0], message
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

    calls = [item for item in message["output"] if item["type"] == "function_call"]
    assert [call["name"] for call in calls] == [""]
    [result] = tool_results(message)
    assert "not found" in result
    replayed_calls = [
        call["function"]["name"]
        for entry in replayed(upstream)
        for call in entry.get("tool_calls") or []
    ]
    assert replayed_calls == [""]
