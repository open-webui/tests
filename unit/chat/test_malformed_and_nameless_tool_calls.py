"""Regression tests for three 0.11.4 chat-pipeline fixes in utils/middleware.py.

* "Empty failed replies in history" (`dbb17a572`): an assistant turn that ended in an
  error with nothing written was handed back to the model as part of the conversation
  on the next message. `load_messages_from_db` now drops replay rows that carry an
  error and no content and no output, and `process_messages_with_output` skips an
  output-less, content-less assistant message rather than sending an empty turn.
* "Malformed tool calls" (`fed94c9f5`): a tool call whose arguments were not an object
  at all (a bare list or string) crashed the reply, because `parse_tool_params` returned
  the parsed non-dict and downstream code assumed dict shape. It now raises ValueError
  and the executor answers with a readable "must be a JSON object, please try again"
  error instead of stopping the reply.
* "Nameless tool calls fail once" (`66e021a92`, PR #29690, issue #29686): an endpoint
  that sends a streamed tool call with `function.name` set to JSON null left the null
  in place, which was stored with the message and sent back on the next turn for the
  endpoint to reject; a null name is now normalised to '' on the spot so the call fails
  locally with the ordinary unknown-tool error.

`streaming_chat_response_handler` is a ~2000-line closure that cannot be constructed
in isolation, so the shipped statements are lifted out of the real middleware source
with ast and executed against a chosen namespace (same harness as
unit/chat/test_stream_event_handling.py). Nothing is reimplemented: the code under test
is the code that ships.

Discriminates: passes on dev 344ea5306; on the pre-fix refs an empty failed turn is
replayed, a non-object arguments crash escapes parse_tool_params, and a null tool name
survives the stream assembly untouched.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def open_webui_backend(open_webui_backend) -> Path:
    return open_webui_backend


def _lift_function(source: str, name: str, indent: str = "") -> types.FunctionType:
    """Extract one shipped function body from the middleware source and make it callable."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            node.decorator_list = []
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict = {}
            exec(compile(module, "<lifted>", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in middleware source")


# ── empty failed replies stay out of the replayed conversation ──────────────


@pytest.mark.asyncio
async def test_failed_empty_assistant_row_is_dropped_from_replay(
    middleware_module, monkeypatch
):
    """The dbb17a572 filter in load_messages_from_db drops an empty failed row."""
    rows = [
        {"id": "u1", "role": "user", "content": "hi"},
        {
            "id": "a1",
            "role": "assistant",
            "content": "",
            "error": {"content": "quota"},
            "parentId": "u1",
        },
        {"id": "a2", "role": "assistant", "content": "hello", "parentId": "a1"},
    ]

    async def fake_map(chat_id):
        return {row["id"]: row for row in rows}

    monkeypatch.setattr(
        middleware_module.Chats,
        "get_messages_map_by_chat_id",
        fake_map,
    )

    replay = await middleware_module.load_messages_from_db("chat-1", "a2")

    assert replay is not None
    assert all(
        not (m.get("error") and not m.get("content") and not m.get("output")) for m in replay
    ), (
        "an assistant turn that failed with nothing written was replayed to the model "
        "as part of the conversation (dbb17a572)"
    )


@pytest.mark.asyncio
async def test_failed_reply_with_content_stays_in_replay(middleware_module, monkeypatch):
    """Partial output before the failure is still the model's context."""
    rows = [
        {"id": "u1", "role": "user", "content": "hi"},
        {
            "id": "a1",
            "role": "assistant",
            "content": "partial answer",
            "error": {"content": "cut off"},
            "parentId": "u1",
        },
    ]

    async def fake_map(chat_id):
        return {row["id"]: row for row in rows}

    monkeypatch.setattr(
        middleware_module.Chats,
        "get_messages_map_by_chat_id",
        fake_map,
    )

    replay = await middleware_module.load_messages_from_db("chat-1", "a1")

    assert any(m.get("content") == "partial answer" for m in replay)


@pytest.mark.asyncio
async def test_successful_assistant_turns_are_untouched(middleware_module, monkeypatch):
    rows = [
        {"id": "u1", "role": "user", "content": "hi"},
        {"id": "a1", "role": "assistant", "content": "hello", "parentId": "u1"},
    ]

    async def fake_map(chat_id):
        return {row["id"]: row for row in rows}

    monkeypatch.setattr(
        middleware_module.Chats,
        "get_messages_map_by_chat_id",
        fake_map,
    )

    replay = await middleware_module.load_messages_from_db("chat-1", "a1")

    assert [m["role"] for m in replay] == ["user", "assistant"]


# ── malformed tool-call arguments answer instead of crashing ───────────────


@pytest.fixture(scope="session")
def parse_tool_params(middleware_module, open_webui_backend):
    """The closure lives inside streaming_chat_response_handler; ship source is lifted."""
    source = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text()
    start = source.index("def parse_tool_params(tool_call):")
    end = source.index("\n", source.index("return params", start))
    block = source[start:end]
    header, _, body = block.partition("\n")
    indented = "\n".join("    " + line if line.strip() else line for line in body.splitlines())
    namespace = {**vars(middleware_module)}
    exec(f"{header}\n{indented}", namespace)
    return namespace["parse_tool_params"]


def test_list_arguments_are_refused_with_a_message(middleware_module, parse_tool_params):
    tool_call = {"id": "call_1", "function": {"name": "search", "arguments": "[1, 2, 3]"}}

    with pytest.raises(ValueError, match="JSON object"):
        parse_tool_params(tool_call)


def test_string_arguments_are_refused_with_a_message(parse_tool_params):
    tool_call = {"id": "call_1", "function": {"name": "search", "arguments": '"just a string"'}}

    with pytest.raises(ValueError, match="JSON object"):
        parse_tool_params(tool_call)


def test_object_arguments_still_parse(parse_tool_params):
    tool_call = {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "a"}'}}

    assert parse_tool_params(tool_call) == {"q": "a"}


def test_empty_arguments_still_parse(parse_tool_params):
    tool_call = {"id": "call_1", "function": {"name": "search", "arguments": ""}}

    assert parse_tool_params(tool_call) == {}


# ── a tool call whose name arrives as null is normalised on the spot ───────


@pytest.fixture(scope="session")
def stream_assembly_statements(open_webui_backend):
    """The new-tool-call branch of the delta handler, lifted verbatim from ship source."""
    source = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text()
    start = source.index("if current_response_tool_call is None:")
    end = source.index("response_tool_calls.append(delta_tool_call)", start)
    return source[start : end + len("response_tool_calls.append(delta_tool_call)")]


def _assemble_one_delta(middleware_module, block, delta_tool_call, existing):
    """Run the shipped new-tool-call branch against one delta."""
    namespace = {
        "JSONCodec": middleware_module.JSONCodec,
        "output_id": lambda prefix: f"{prefix}-1",
        "response_tool_calls": existing,
        "delta_tool_call": delta_tool_call,
        "tool_call_index": 0,
        "current_response_tool_call": None,
    }
    exec(_indent_tail(block), namespace)
    return namespace["response_tool_calls"]


def _indent_tail(block: str) -> str:
    """Dedent the lifted block to column 0 so it runs as a bare statement list."""
    lines = block.splitlines()
    dedented = [line[4:] if line.startswith("    ") else line for line in lines]
    return "\n".join(dedented)


def test_null_tool_call_name_is_normalised_to_empty(
    middleware_module, stream_assembly_statements
):
    """PR #29690: `function.name: null` in a streamed tool call must become ''.

    Pre-fix the null survived assembly, was stored with the message and sent back on
    the next turn for the endpoint to reject; with '' the call fails once locally
    through the ordinary unknown-tool path.
    """
    assembled = _assemble_one_delta(
        middleware_module,
        stream_assembly_statements,
        {"index": 0, "id": "call_1", "function": {"name": None, "arguments": "{}"}},
        [],
    )

    assert assembled[0]["function"]["name"] == "", (
        "a null tool call name survived stream assembly and was stored with the message "
        "for the endpoint to reject on the next turn (#29686)"
    )


def test_missing_name_still_defaults_to_empty(middleware_module, stream_assembly_statements):
    assembled = _assemble_one_delta(
        middleware_module,
        stream_assembly_statements,
        {"index": 0, "id": "call_1", "function": {"arguments": "{}"}},
        [],
    )

    assert assembled[0]["function"]["name"] == ""
