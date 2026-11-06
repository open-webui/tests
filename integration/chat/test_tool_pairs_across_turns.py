"""Regression: a tool result whose id matched a call elsewhere in the chat reached the provider.

Issue open-webui/open-webui#28937, fix 25604d707 (PR open-webui/open-webui#31431). Before a
chat's history goes to the model, calls without results and results without calls are left out.
The check matched ids across the whole conversation, so a result stored under a block whose
call sat in an earlier block (providers that number calls per round reuse `call_0`), or a result
stored before its call, was kept. Anthropic and Bedrock rejected every following message in that
chat. Calls and results now pair only within one assistant block and the results right after it.

Discriminates: passes on dev efe63bd34, fails with 25604d707 reverted (the leftover result under
the later block, and the result saved before its call, are sent unpaired).
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REUSED_ID = "call_0"


def text(value: str) -> dict:
    return {"type": "message", "content": [{"type": "output_text", "text": value}]}


def call(call_id: str) -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "kb_lookup",
        "arguments": "{}",
        "status": "completed",
    }


def result(call_id: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": f"result for {call_id}"}],
        "status": "completed",
    }


def assistant(*output: dict) -> dict:
    visible = "\n".join(
        part["text"] for item in output if item["type"] == "message" for part in item["content"]
    )
    return {"role": "assistant", "content": visible, "output": list(output)}


def replay(user, upstream, *replies: dict) -> list[dict]:
    """Seed one user turn per stored reply, send the next message, return what the model got."""
    history = []
    for index, stored in enumerate(replies):
        history += [{"role": "user", "content": f"question {index}"}, stored]
    upstream.queue(reply.text("next answer"))
    with user.client() as client:
        chat_id, last_id = seed_chat(client, history)
        ask(client, "and now?", chat_id=chat_id, parent_id=last_id)
    return conversation(upstream.chat_requests()[-1])


def unpaired_pieces(messages: list[dict]) -> list[str]:
    """Each call not answered right after its assistant message, and each result not asked for."""
    problems = []
    open_calls: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            if message["tool_call_id"] not in open_calls:
                problems.append(f"result {message['tool_call_id']} without its call")
            open_calls.discard(message["tool_call_id"])
            continue
        problems += [f"call {call_id} without its result" for call_id in sorted(open_calls)]
        open_calls = {entry["id"] for entry in message.get("tool_calls") or []}
    problems += [f"call {call_id} without its result" for call_id in sorted(open_calls)]
    return problems


def tool_call_ids(messages: list[dict]) -> list[str]:
    return [
        entry["id"]
        for message in messages
        if message["role"] == "assistant"
        for entry in message.get("tool_calls") or []
    ]


ANSWERED = assistant(call(REUSED_ID), result(REUSED_ID), text("Found it."))


def test_leftover_results_under_a_later_call_block_are_left_out(user, upstream):
    # the shape in the issue: a second round of one reply declares one call, gets two results
    rounds = assistant(
        call(REUSED_ID),
        call("call_1"),
        result(REUSED_ID),
        result("call_1"),
        text("One more look."),
        call(REUSED_ID),
        result(REUSED_ID),
        result("call_1"),
        text("Found it."),
    )

    messages = replay(user, upstream, rounds)

    assert unpaired_pieces(messages) == [], messages
    assert tool_call_ids(messages) == [REUSED_ID, "call_1", REUSED_ID]
    assert [message["role"] for message in messages].count("tool") == 3


def test_a_result_saved_before_its_reused_call_is_left_out(user, upstream):
    later = assistant(result(REUSED_ID), text("Checking again."), call(REUSED_ID))

    messages = replay(user, upstream, ANSWERED, later)

    assert unpaired_pieces(messages) == [], messages
    assert tool_call_ids(messages) == [REUSED_ID]


@pytest.mark.parametrize(
    "later_reply",
    [
        pytest.param(assistant(call(REUSED_ID)), id="bare-call"),
        pytest.param(assistant(result(REUSED_ID)), id="bare-result"),
        pytest.param(
            assistant(call(REUSED_ID), call("call_1"), result("call_1")), id="partial-batch"
        ),
        pytest.param(
            assistant(call(REUSED_ID), result(REUSED_ID), result("call_1")), id="leftover-result"
        ),
    ],
)
def test_no_broken_shape_after_an_answered_call_reaches_the_provider(user, upstream, later_reply):
    messages = replay(user, upstream, ANSWERED, later_reply)

    assert unpaired_pieces(messages) == [], messages


def test_the_text_of_a_reply_with_a_dropped_call_survives(user, upstream):
    messages = replay(user, upstream, ANSWERED, assistant(text("Checking again."), call(REUSED_ID)))

    assistant_text = [message["content"] for message in messages if message["role"] == "assistant"]
    assert "Checking again." in assistant_text


def test_well_formed_replies_reusing_an_id_are_sent_whole(user, upstream):
    messages = replay(
        user,
        upstream,
        ANSWERED,
        assistant(call(REUSED_ID), result(REUSED_ID), text("Found it again.")),
    )

    assert unpaired_pieces(messages) == [], messages
    assert tool_call_ids(messages) == [REUSED_ID, REUSED_ID]
    assert [message["role"] for message in messages].count("tool") == 2
