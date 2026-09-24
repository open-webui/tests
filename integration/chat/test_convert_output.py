"""A stored turn with an unanswered tool call was replayed to the provider unbalanced.

Regression for open-webui/open-webui#24758, fixed in PR #24798 (`reconcile_tool_pairs` in
`utils/misc.py`). The next turn rebuilds the conversation from the `output` items stored on
each assistant message. When a `function_call_output` was missing (a knowledge base updated
mid-chat, a call interrupted before its result was written) or a result survived without its
call, the rebuilt history carried an assistant `tool_calls` entry with no `tool` message after
it, or the reverse, and strict providers (Anthropic, Bedrock Converse) answered 400. Three
layers now keep the pairs balanced: only resolved calls with a result are replayed (0.11.1),
the reconciliation strips what is left unpaired, and `sanitize_tool_pairs` checks the final
payload.

Twin of unit/chat/test_convert_output.py.

Discriminates: every layer alone keeps these tests green, so reverting PR #24798 by itself
changes nothing here; with all three removed (every stored call and result replayed, no
reconciliation, no sanitizing) every unbalanced shape fails, while the well-formed shapes and
the surviving assistant text pass.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def text(value: str) -> dict:
    return {"type": "message", "content": [{"type": "output_text", "text": value}]}


def call(call_id: str, status: str = "completed") -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "kb_lookup",
        "arguments": "{}",
        "status": status,
    }


def result(call_id: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": f"result for {call_id}"}],
        "status": "completed",
    }


def replay_stored_output(user, upstream, output: list[dict]) -> list[dict]:
    """Seed a turn with `output`, send the next message, return what the provider got."""
    upstream.queue(reply.text("next answer"))
    visible_text = "\n".join(
        part["text"] for item in output if item["type"] == "message" for part in item["content"]
    )
    with user.client() as client:
        chat_id, assistant_id = seed_chat(
            client,
            [
                {"role": "user", "content": "look it up"},
                {"role": "assistant", "content": visible_text, "output": output},
            ],
        )
        ask(client, "and now?", chat_id=chat_id, parent_id=assistant_id)
    return upstream.chat_requests()[-1]["messages"]


def requested_call_ids(messages: list[dict]) -> set[str]:
    return {
        tool_call["id"]
        for message in messages
        if message["role"] == "assistant"
        for tool_call in message.get("tool_calls") or []
    }


def answered_call_ids(messages: list[dict]) -> set[str]:
    return {message["tool_call_id"] for message in messages if message["role"] == "tool"}


STORED_OUTPUTS = [
    pytest.param([text("Let me look that up."), call("orphan")], set(), id="call-without-result"),
    pytest.param(
        [call("answered"), call("orphan"), result("answered")], {"answered"}, id="partial-batch"
    ),
    pytest.param(
        [text("First answer."), result("stale"), call("orphan")], set(), id="orphans-both-ways"
    ),
    pytest.param([text("Answer."), result("no_call")], set(), id="result-without-call"),
    pytest.param(
        [call("pending", status="in_progress"), result("pending")], set(), id="unresolved-call"
    ),
    pytest.param([call("c1"), result("c1")], {"c1"}, id="well-formed"),
    pytest.param(
        [call("c1"), call("c2"), result("c1"), result("c2")], {"c1", "c2"}, id="well-formed-batch"
    ),
    pytest.param(
        [text("thinking"), call("c1"), result("c1"), text("done")],
        {"c1"},
        id="well-formed-with-text",
    ),
]


@pytest.mark.parametrize("output,surviving_calls", STORED_OUTPUTS)
def test_only_paired_tool_calls_reach_the_provider(user, upstream, output, surviving_calls):
    messages = replay_stored_output(user, upstream, output)

    assert requested_call_ids(messages) == answered_call_ids(messages) == surviving_calls
    empty_assistants = [
        message
        for message in messages
        if message["role"] == "assistant"
        and not (message.get("content") or "").strip()
        and not message.get("tool_calls")
    ]
    assert not empty_assistants, messages


@pytest.mark.parametrize(
    "output",
    [
        pytest.param([text("Here is what I found."), call("orphan")], id="call-without-result"),
        pytest.param([text("Here is what I found."), result("no_call")], id="result-without-call"),
    ],
)
def test_the_assistant_text_survives_a_stripped_orphan(user, upstream, output):
    messages = replay_stored_output(user, upstream, output)

    assistant_text = [message["content"] for message in messages if message["role"] == "assistant"]
    assert assistant_text == ["Here is what I found."]
