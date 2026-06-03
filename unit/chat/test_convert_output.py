"""Regression tests for open-webui/open-webui#24758.

`convert_output_to_messages` reconstructs an OpenAI-format chat history
from the Responses-API-native `output` items stored on a chat. The
stored output can be incomplete — a `function_call_output` may be
missing (lost when a knowledge base is updated mid-chat, or a tool
call interrupted before its result is written), or a `function_call`
may be missing while its output survived.

Before the fix (PR #24798), the final `flush_pending()` emitted an
assistant message carrying `tool_calls` with no following `tool` role
message. Strict providers (Anthropic, AWS Bedrock Converse) reject
that history with a 400:

    tool_use ids were found without tool_result blocks

The fix adds `_reconcile_tool_pairs()` after `flush_pending()`: any
assistant `tool_calls` entry whose id has no matching `tool` message
is stripped, an assistant message left empty by that stripping is
dropped, and a `tool` message whose `tool_call_id` has no surviving
tool call is dropped too. Well-formed output is a no-op.

These tests assert the provider-facing invariant directly: in the
reconstructed message list, every assistant tool_call id has a
matching later `tool` message, and every `tool` message's
tool_call_id has a matching assistant tool_call — regardless of how
incomplete the stored output was.
"""

from __future__ import annotations

from types import ModuleType

import pytest

# -----------------------------------------------------------------------------
# Stored-output item builders (Responses-API-native shapes)
# -----------------------------------------------------------------------------


def _msg(text: str) -> dict:
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _call(call_id: str, name: str = "kb_lookup") -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _call_output(call_id: str, text: str = "result") -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": text}],
    }


# -----------------------------------------------------------------------------
# Invariant checker — the actual provider contract
# -----------------------------------------------------------------------------


def _assert_tool_pairs_balanced(messages: list[dict]) -> None:
    """Every assistant tool_call id has a matching `tool` message, and
    every `tool` message's tool_call_id has a matching assistant
    tool_call. This is exactly what Anthropic / Bedrock Converse
    enforce."""
    called_ids = {
        tc.get("id")
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc.get("id")
    }
    answered_ids = {
        m.get("tool_call_id") for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
    }

    orphan_calls = called_ids - answered_ids
    orphan_results = answered_ids - called_ids
    assert not orphan_calls, (
        f"Regression of open-webui/open-webui#24758: assistant tool_calls "
        f"with no matching tool result: {sorted(orphan_calls)}. Strict "
        f"providers (Anthropic/Bedrock) 400 on this."
    )
    assert not orphan_results, (
        f"tool message(s) with no matching assistant tool_call: "
        f"{sorted(orphan_results)}. Strict providers reject this too."
    )


# -----------------------------------------------------------------------------
# Regressions
# -----------------------------------------------------------------------------


@pytest.mark.regression
def test_orphan_function_call_is_stripped(misc_module: ModuleType) -> None:
    """The exact #24758 scenario: a function_call whose
    function_call_output never made it into stored output."""
    output = [
        _msg("Let me look that up."),
        _call("tooluse_orphan_1"),
        # function_call_output for tooluse_orphan_1 is MISSING
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)


@pytest.mark.regression
def test_partial_batch_keeps_only_answered_calls(misc_module: ModuleType) -> None:
    """Two tool calls in one assistant turn, only one answered. The
    answered call must survive; the orphan must be stripped."""
    output = [
        _call("tooluse_answered"),
        _call("tooluse_orphan"),
        _call_output("tooluse_answered", "ok"),
        # no output for tooluse_orphan
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)

    # The answered call should still be present and usable.
    called_ids = {
        tc["id"]
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    assert "tooluse_answered" in called_ids, messages


@pytest.mark.regression
def test_assistant_text_preserved_when_orphan_call_stripped(
    misc_module: ModuleType,
) -> None:
    """If the assistant turn had real text *and* an orphan tool call,
    strip the call but keep the message + its text."""
    output = [
        _msg("Here is what I found."),
        _call("tooluse_orphan"),
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)

    assistant_text = " ".join(
        m.get("content", "")
        for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    )
    assert "Here is what I found." in assistant_text, messages


@pytest.mark.regression
def test_assistant_message_dropped_when_only_orphan_call(
    misc_module: ModuleType,
) -> None:
    """An assistant turn that was *only* the now-orphaned tool call
    (no text) should be dropped entirely, not left as an empty
    assistant message."""
    output = [
        _msg("First answer."),
        _call_output("tooluse_old", "leftover"),  # paired below? no — orphan result
        _call("tooluse_orphan"),
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)

    # No assistant message should be empty (no content, no tool_calls).
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content")
            has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
            assert has_content or m.get("tool_calls") or m.get("reasoning_content"), (
                f"Empty assistant message left in output: {m!r}"
            )


@pytest.mark.regression
def test_orphan_tool_result_is_stripped(misc_module: ModuleType) -> None:
    """Symmetric case: a function_call_output whose function_call is
    missing. The dangling tool message must be dropped."""
    output = [
        _msg("Answer."),
        _call_output("tooluse_no_call", "dangling result"),
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)


@pytest.mark.regression
@pytest.mark.parametrize(
    "output",
    [
        # single well-formed call+result
        [_call("c1"), _call_output("c1", "r1")],
        # multi-call batch, all answered
        [
            _call("c1"),
            _call("c2"),
            _call_output("c1", "r1"),
            _call_output("c2", "r2"),
        ],
        # text + call + result + follow-up text
        [
            _msg("thinking"),
            _call("c1"),
            _call_output("c1", "r1"),
            _msg("done"),
        ],
    ],
)
def test_wellformed_output_is_noop(misc_module: ModuleType, output) -> None:
    """Reconciliation must not disturb well-formed histories: every id
    pairs, so nothing is stripped and all calls survive."""
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)

    expected_call_ids = {item["call_id"] for item in output if item["type"] == "function_call"}
    surviving_call_ids = {
        tc["id"]
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    assert surviving_call_ids == expected_call_ids, (
        f"Well-formed output should be a no-op; expected calls "
        f"{sorted(expected_call_ids)}, got {sorted(surviving_call_ids)}"
    )
