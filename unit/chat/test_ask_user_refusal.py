"""Regression tests for a refused `ask_user` call ending the turn with no reply.

Fix commit `be958d7b0` (PR #29252, issue #29077), `open_webui/utils/ask_user.py`.

When a model called `ask_user` in a shape Open WebUI refuses (alongside another tool, or twice
in one turn), the staging helper looked at only the first `ask_user` call and produced a single
error result, so any further refused call was left with a `function_call` item and no
`function_call_output`. The middleware then skipped the rest of the round, and the turn ended
with nothing said. Staging now walks every refused `ask_user` call, gives each its own error
result, and hands the error back to the caller so the remaining tool calls still run.

Discriminates: passes on v0.11.3, fails on v0.11.1 (a second refused ask_user call never gets a
result, and the refusal text does not tell the model the call did not run).
"""

from __future__ import annotations

import itertools
from types import ModuleType

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def ask_user_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.ask_user")


output_id_counter = itertools.count()


def make_output_id(prefix: str) -> str:
    return f"{prefix}-{next(output_id_counter)}"


def valid_arguments(question_id: str = "q1") -> str:
    return (
        '{"questions": [{"id": "%s", "header": "Scope", "question": "Which one?",'
        ' "options": [{"label": "A", "description": "first"},'
        ' {"label": "B", "description": "second"}]}]}' % question_id
    )


def ask_user_call(call_id: str, arguments: str) -> dict:
    return {"id": call_id, "function": {"name": "ask_user", "arguments": arguments}}


def other_tool_call(call_id: str = "call-web") -> dict:
    return {"id": call_id, "function": {"name": "web_search", "arguments": '{"query": "x"}'}}


def stage(module: ModuleType, tool_calls: list[dict], output: list[dict]) -> str | None:
    """Drive whichever staging entrypoint the checkout ships; returns the refusal text or None."""
    staging = getattr(module, "stage_ask_user_tool_calls", None) or module.stage_ask_user_tool_call
    result = staging(tool_calls, output, make_output_id)
    if isinstance(result, tuple):
        return result[1]
    return result["error"] if result else None


def error_texts_by_call_id(output: list[dict]) -> dict[str, str]:
    return {
        item["call_id"]: item["output"][0]["text"]
        for item in output
        if item.get("type") == "function_call_output"
    }


# -----------------------------------------------------------------------------
# Narrow
# -----------------------------------------------------------------------------


def test_every_refused_ask_user_call_gets_its_own_error_result(ask_user_module: ModuleType) -> None:
    """Two ask_user calls in one turn are both refused, so both need a result to unblock it."""
    output: list[dict] = []
    tool_calls = [
        ask_user_call("call-a", valid_arguments("q1")),
        ask_user_call("call-b", valid_arguments("q2")),
    ]

    error = stage(ask_user_module, tool_calls, output)

    assert error
    staged = [item for item in output if item.get("type") == "function_call"]
    assert [item["call_id"] for item in staged] == ["call-a", "call-b"], output
    assert [item["status"] for item in staged] == ["completed", "completed"], output
    results = error_texts_by_call_id(output)
    assert set(results) == {"call-a", "call-b"}, output
    assert all(text.startswith("Error:") for text in results.values()), results


def test_refusal_text_tells_the_model_the_call_did_not_run(ask_user_module: ModuleType) -> None:
    """The model has to know the question was never asked, otherwise it waits for an answer."""
    output: list[dict] = []
    tool_calls = [ask_user_call("call-a", valid_arguments()), other_tool_call()]

    error = stage(ask_user_module, tool_calls, output)

    assert "did not run" in error, error
    assert error_texts_by_call_id(output)["call-a"] == error


# -----------------------------------------------------------------------------
# Broad: every refusal reason comes back as a usable tool result
# -----------------------------------------------------------------------------


REFUSAL_CASES = [
    ("mixed_with_another_tool", [ask_user_call("call-a", valid_arguments()), other_tool_call()]),
    ("malformed_json", [ask_user_call("call-a", "{not json")]),
    ("arguments_not_an_object", [ask_user_call("call-a", "[]")]),
    ("no_questions", [ask_user_call("call-a", '{"questions": []}')]),
    (
        "too_many_questions",
        [
            ask_user_call(
                "call-a",
                '{"questions": [%s]}'
                % ", ".join(
                    '{"id": "q%d", "question": "Q", "options": [{"label": "A", "description": "d"},'
                    ' {"label": "B", "description": "d"}]}' % index
                    for index in range(4)
                ),
            )
        ],
    ),
    (
        "missing_question_id",
        [
            ask_user_call(
                "call-a",
                '{"questions": [{"question": "Q", "options": [{"label": "A", "description": "d"},'
                ' {"label": "B", "description": "d"}]}]}',
            )
        ],
    ),
    (
        "too_few_options",
        [ask_user_call("call-a", '{"questions": [{"id": "q1", "question": "Q", "options": []}]}')],
    ),
    (
        "option_without_description",
        [
            ask_user_call(
                "call-a",
                '{"questions": [{"id": "q1", "question": "Q", "options": [{"label": "A"},'
                ' {"label": "B", "description": "d"}]}]}',
            )
        ],
    ),
    (
        "duplicate_question_id",
        [
            ask_user_call(
                "call-a",
                '{"questions": [%s, %s]}'
                % (
                    '{"id": "q1", "question": "Q", "options": [{"label": "A", "description": "d"},'
                    ' {"label": "B", "description": "d"}]}',
                    '{"id": "q1", "question": "Q2", "options": [{"label": "A", "description": "d"},'
                    ' {"label": "B", "description": "d"}]}',
                ),
            )
        ],
    ),
    ("question_not_an_object", [ask_user_call("call-a", '{"questions": ["not an object"]}')]),
    (
        "option_not_an_object",
        [
            ask_user_call(
                "call-a",
                '{"questions": [{"id": "q1", "question": "Q", "options": ["A",'
                ' {"label": "B", "description": "d"}]}]}',
            )
        ],
    ),
    (
        "empty_question_text",
        [
            ask_user_call(
                "call-a",
                '{"questions": [{"id": "q1", "question": "  ", "options":'
                ' [{"label": "A", "description": "d"}, {"label": "B", "description": "d"}]}]}',
            )
        ],
    ),
]


@pytest.mark.parametrize("case,tool_calls", REFUSAL_CASES, ids=[case for case, _ in REFUSAL_CASES])
def test_refusal_reaches_the_model_as_a_tool_result(
    ask_user_module: ModuleType, case: str, tool_calls: list[dict]
) -> None:
    output: list[dict] = []

    error = stage(ask_user_module, tool_calls, output)

    assert error and error.startswith("Error:"), (case, error)
    staged = [item for item in output if item.get("type") == "function_call"]
    assert [item["status"] for item in staged] == ["completed"] * len(staged), output
    assert error_texts_by_call_id(output) == {item["call_id"]: error for item in staged}, output


# -----------------------------------------------------------------------------
# Nearby
# -----------------------------------------------------------------------------


def test_accepted_ask_user_call_is_staged_pending_with_no_error_result(
    ask_user_module: ModuleType,
) -> None:
    output: list[dict] = []

    error = stage(ask_user_module, [ask_user_call("call-a", valid_arguments())], output)

    assert error is None
    assert [item["type"] for item in output] == ["function_call"]
    assert output[0]["status"] == "pending"
    assert output[0]["call_id"] == "call-a"


def test_accepted_ask_user_arguments_are_normalized(ask_user_module: ModuleType) -> None:
    from open_webui.utils.json_codec import JSONCodec

    output: list[dict] = []

    stage(ask_user_module, [ask_user_call("call-a", valid_arguments())], output)

    arguments = JSONCodec.loads(output[0]["arguments"])
    assert arguments["timeout_ms"] == 120_000
    assert arguments["questions"][0]["header"] == "Scope"
    assert arguments["questions"][0]["options"][0] == {"label": "A", "description": "first"}


def test_refused_batch_leaves_the_other_tool_call_intact(ask_user_module: ModuleType) -> None:
    """The caller filters ask_user out and runs the rest, so the other call must survive."""
    output: list[dict] = []
    other = other_tool_call()
    tool_calls = [ask_user_call("call-a", valid_arguments()), other]

    assert stage(ask_user_module, tool_calls, output)
    assert tool_calls[1] is other
    assert not [item for item in output if item.get("call_id") == other["id"]], output


def test_a_turn_without_ask_user_stages_nothing(ask_user_module: ModuleType) -> None:
    output: list[dict] = []

    error = stage(ask_user_module, [other_tool_call(), other_tool_call("call-two")], output)

    assert error is None
    assert output == []


def test_no_tool_calls_at_all_stages_nothing(ask_user_module: ModuleType) -> None:
    output: list[dict] = []

    assert stage(ask_user_module, [], output) is None
    assert output == []
