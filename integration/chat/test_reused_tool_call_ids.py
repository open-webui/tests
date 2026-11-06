"""Regression: a model that reuses tool call ids across rounds mixed up its calls, #28305.

Fix 88720c692 (PR #31887). Some providers (Kimi K3 on OpenRouter) number tool calls from zero
again in every round of one reply. The second round's call then took the place of the first one
with the same id: the stored reply showed the first call with the second call's arguments, and
the model was sent its first result next to the wrong arguments. A call whose id is already used
in the reply now gets a fresh id, so every call keeps its own arguments and result.

Discriminates: passes on dev b859124f9, fails with 88720c692 reverted (the first call is stored
and sent with the second call's arguments and result).
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import python_tool
from harness.tool_calls import without_hidden_calls
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SHOUTING_TOOL = '''
class Tools:
    def shout(self, text: str) -> str:
        """Shout the text."""
        return text.upper()
'''


def shout(text: str, call_id: str) -> reply.Reply:
    return reply.tool_call("shout", {"text": text}, call_id=call_id, match=reply.answering("shout"))


def shout_twice(admin, make_user, upstream, first_id: str, second_id: str) -> dict:
    """One reply that shouts "first", then "second", in two rounds with the given call ids."""
    upstream.queue(
        shout("first", first_id),
        shout("second", second_id),
        reply.text("all shouted", match=reply.answering("shout")),
    )
    with python_tool(admin, SHOUTING_TOOL) as tool_id, make_user().client() as client:
        _, message = ask(client, "shout both words", tool_ids=[tool_id])
    return message


def stored_pairs(message: dict) -> list[tuple[str, str]]:
    """Each stored call's text argument with the result stored under its id."""
    output = without_hidden_calls(message["output"])
    results = {
        item["call_id"]: item["output"][0]["text"]
        for item in output
        if item["type"] == "function_call_output"
    }
    return [
        (json.loads(item["arguments"])["text"], results.get(item["call_id"]))
        for item in output
        if item["type"] == "function_call"
    ]


def sent_pairs(upstream) -> list[tuple[str, str]]:
    """Each call the provider was sent last, by its text argument, with the result sent for it."""
    messages = conversation(upstream.chat_requests()[-1])
    results = {
        entry["tool_call_id"]: entry["content"] for entry in messages if entry["role"] == "tool"
    }
    return [
        (json.loads(call["function"]["arguments"])["text"], results.get(call["id"]))
        for entry in messages
        for call in entry.get("tool_calls") or []
    ]


def test_calls_reusing_an_id_keep_their_own_arguments_and_results(admin, make_user, upstream):
    message = shout_twice(admin, make_user, upstream, "call_0", "call_0")

    assert stored_pairs(message) == [("first", "FIRST"), ("second", "SECOND")]
    assert message["content"].endswith("all shouted")


def test_calls_reusing_an_id_reach_the_model_paired_with_their_results(admin, make_user, upstream):
    shout_twice(admin, make_user, upstream, "call_0", "call_0")

    assert sent_pairs(upstream) == [("first", "FIRST"), ("second", "SECOND")]
    call_ids = [
        call["id"]
        for entry in conversation(upstream.chat_requests()[-1])
        for call in entry.get("tool_calls") or []
    ]
    assert len(set(call_ids)) == 2, call_ids


# ---------------------------------------------------------------- nearby


def test_unique_call_ids_are_kept_as_the_provider_sent_them(admin, make_user, upstream):
    message = shout_twice(admin, make_user, upstream, "call_a", "call_b")

    assert stored_pairs(message) == [("first", "FIRST"), ("second", "SECOND")]
    stored_ids = [
        item["call_id"]
        for item in without_hidden_calls(message["output"])
        if item["type"] == "function_call"
    ]
    assert stored_ids == ["call_a", "call_b"]
