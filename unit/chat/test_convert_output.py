"""Regression tests for open-webui/open-webui#24758.

`convert_output_to_messages` reconstructs an OpenAI-format chat history
from the Responses-API-native `output` items stored on a chat. The
stored output can be incomplete: a `function_call_output` may be
missing (lost when a knowledge base is updated mid-chat, or a tool
call interrupted before its result is written), or a `function_call`
may be missing while its output survived.

Before the fix (PR #24798), the final `flush_pending()` emitted an
assistant message carrying `tool_calls` with no following `tool` role
message. Strict providers (Anthropic, AWS Bedrock Converse) reject
that history with a 400:

    tool_use ids were found without tool_result blocks

The fix adds `reconcile_tool_pairs()` after `flush_pending()`: any
assistant `tool_calls` entry whose id has no matching `tool` message
is stripped, an assistant message left empty by that stripping is
dropped, and a `tool` message whose `tool_call_id` has no surviving
tool call is dropped too. Well-formed output is a no-op.

0.11.1 hardened the same invariant one layer earlier: only a
`function_call` that both carries a resolved `status`
(completed/failed/rejected) and has a matching `function_call_output`
is replayed at all, so an unfinished or rejected call never reaches
the message list. Stored output always carries `status`, written by
the streaming path in `utils/middleware.py`.

These tests assert the provider-facing invariant directly: in the
reconstructed message list, every assistant tool_call id has a
matching later `tool` message, and every `tool` message's
tool_call_id has a matching assistant tool_call, regardless of how
incomplete the stored output was. `reconcile_tool_pairs` is also driven
directly, because 0.11.1's earlier status-and-result filter already
balances everything `convert_output_to_messages` can emit, so the
end-to-end cases alone would not notice the reconciliation disappearing.

Discriminates: passes on v0.11.0 and v0.11.1, fails on the pre-#24798 ref
where `reconcile_tool_pairs` does not exist and the reconstructed history
keeps its unpaired tool calls.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.regression

# -----------------------------------------------------------------------------
# Stored-output item builders (Responses-API-native shapes)
# -----------------------------------------------------------------------------


def _msg(text: str) -> dict:
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _call(call_id: str, name: str = "kb_lookup", status: str = "completed") -> dict:
    # status is always written by the streaming path; 0.11.1 only replays resolved calls
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
        "status": status,
    }


def _call_output(call_id: str, text: str = "result") -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": [{"type": "input_text", "text": text}],
    }


# -----------------------------------------------------------------------------
# Invariant checker: the actual provider contract
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


def test_assistant_message_dropped_when_only_orphan_call(
    misc_module: ModuleType,
) -> None:
    """An assistant turn that was *only* the now-orphaned tool call
    (no text) should be dropped entirely, not left as an empty
    assistant message."""
    output = [
        _msg("First answer."),
        _call_output("tooluse_old", "leftover"),  # no matching call: orphan result
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


def test_orphan_tool_result_is_stripped(misc_module: ModuleType) -> None:
    """Symmetric case: a function_call_output whose function_call is
    missing. The dangling tool message must be dropped."""
    output = [
        _msg("Answer."),
        _call_output("tooluse_no_call", "dangling result"),
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)


def test_unresolved_call_is_not_replayed(misc_module: ModuleType) -> None:
    """A call still `in_progress` (interrupted, or awaiting approval)
    must not be replayed even if some output for it exists."""
    output = [
        _call("tooluse_pending", status="in_progress"),
        _call_output("tooluse_pending", "partial"),
    ]
    messages = misc_module.convert_output_to_messages(output)
    _assert_tool_pairs_balanced(messages)

    surviving_call_ids = {
        tc["id"]
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    assert not surviving_call_ids, messages


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


# -----------------------------------------------------------------------------
# Direct reconciliation: the fix itself, driven with hand-built histories
# -----------------------------------------------------------------------------


def _tool_call(call_id: str, name: str = "kb_lookup") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _tool_result(call_id: str, text: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def test_reconcile_strips_both_orphan_directions(misc_module: ModuleType) -> None:
    """An unbalanced history that `convert_output_to_messages` cannot produce
    on this ref, so it reaches the reconciliation unfiltered."""
    messages = [
        {"role": "user", "content": "look it up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("answered"), _tool_call("orphan_call")],
        },
        _tool_result("answered", "ok"),
        _tool_result("orphan_result", "dangling"),
    ]

    reconciled = misc_module.reconcile_tool_pairs(messages)
    _assert_tool_pairs_balanced(reconciled)

    assert [m["role"] for m in reconciled] == ["user", "assistant", "tool"], reconciled
    assert [tc["id"] for tc in reconciled[1]["tool_calls"]] == ["answered"], reconciled


def test_reconcile_drops_an_assistant_left_with_nothing(misc_module: ModuleType) -> None:
    messages = [{"role": "assistant", "content": "", "tool_calls": [_tool_call("orphan")]}]

    assert misc_module.reconcile_tool_pairs(messages) == []


def test_reconcile_keeps_assistant_text_and_drops_the_orphan_call(
    misc_module: ModuleType,
) -> None:
    messages = [
        {"role": "assistant", "content": "Here is what I found.", "tool_calls": [_tool_call("x")]}
    ]

    assert misc_module.reconcile_tool_pairs(messages) == [
        {"role": "assistant", "content": "Here is what I found."}
    ]


def test_reconcile_is_a_noop_on_a_balanced_history(misc_module: ModuleType) -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c1"), _tool_call("c2")]},
        _tool_result("c1", "r1"),
        _tool_result("c2", "r2"),
        {"role": "assistant", "content": "done"},
    ]

    assert misc_module.reconcile_tool_pairs(messages) == messages


def test_convert_output_returns_through_the_reconciliation(misc_module: ModuleType) -> None:
    """The call site, not just the helper: dropping it reopens #24758 for every
    history the earlier status filter does not already balance."""
    tree = ast.parse(Path(misc_module.__file__).read_text(encoding="utf-8"))
    converter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "convert_output_to_messages"
    )

    reconciled_returns = [
        node
        for node in ast.walk(converter)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "reconcile_tool_pairs"
    ]

    assert reconciled_returns, (
        "convert_output_to_messages no longer returns through reconcile_tool_pairs, "
        "so an unbalanced stored output is handed straight to the provider (#24758)"
    )
