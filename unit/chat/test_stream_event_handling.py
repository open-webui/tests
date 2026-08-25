"""Regression tests for the 0.11.1 Responses-API streaming repairs.

Six separate breakages in `open_webui/utils/middleware.py`, all on the path that turns a
provider's SSE stream into the stored `output` items:

* #27800 / #27810 (`74a790282`) — `response.completed` overwrote the accumulated reply with
  the terminal event's `output` whenever that key was merely present, so a provider sending
  `output: []` at the end erased the whole reply: the message vanished, an empty message was
  persisted and replayed as next turn's context. The same commit moved the dedicated
  `response.output_item.done` branch above the generic `response.*.done` branch, which had
  been shadowing it since the feature landed, so the finished item (and the annotations that
  only arrive with it) is applied instead of dropped.
* #28312 (`fc8a9b8ed`) — the generic `response.*.delta` branch bound `new_output` only inside
  the guard checking the target item exists but returned it outside, so a delta arriving
  before its output item raised `UnboundLocalError`; and a two-part event name fell off the
  end of the branch, returning `None` for a caller that unpacks a 2-tuple. Either way the
  reply was cut short and the end-of-message filter hooks never ran.
* #28872 (`883c7434f`) — a stream ending inside a reasoning item with no answer text hit
  `int(ended_at - started_at)` with `started_at` absent or `None`, failing the whole turn.
* #28016 (`f3f76095d`) — every tool result was stamped `completed`. The new
  `_is_tool_result_error` helper recognises error payloads so a failed call is stamped
  `failed`.
* #28633 (`a610d77137`) — when the follow-up request made after a tool ran came back >= 400,
  the handler just `break`-ed and the reply ended with nothing said. The new
  `get_response_error_detail` in `utils/misc.py` extracts the provider's message and the
  handler emits it. `main.py` dropped its inline copy of the same parsing.

`handle_responses_streaming_event` and `get_response_error_detail` are module-level and are
driven directly. The other three sites live inside `streaming_chat_response_handler`, a
~2000-line coroutine that cannot be constructed in isolation, so the shipped statements are
lifted out of the real middleware source with `ast` and executed against a chosen namespace.
Nothing is reimplemented: the code under test is the code that ships.

Discriminates: passes on v0.11.1, fails on v0.11.0 (empty terminal output erases the reply,
`output_item.done` is ignored, an orphan delta crashes, a started_at-less reasoning item
crashes, error tool results are stamped completed, an HTTP error after a tool is silent).
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def handle_event(middleware_module):
    return middleware_module.handle_responses_streaming_event


@pytest.fixture(scope="session")
def middleware_source(open_webui_backend: Path) -> str:
    return (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text(
        encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Lifting the shipped statements out of `streaming_chat_response_handler`
# -----------------------------------------------------------------------------


def _handler(source: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(source)
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "streaming_chat_response_handler"
    ]
    assert len(handlers) == 1, (
        f"expected one `streaming_chat_response_handler` in middleware.py, found {len(handlers)}"
    )
    return handlers[0]


def _unique(source: str, predicate, what: str) -> ast.stmt:
    hits = [node for node in ast.walk(_handler(source)) if predicate(node)]
    assert len(hits) == 1, (
        f"expected exactly one {what} in `streaming_chat_response_handler`, found {len(hits)}; "
        f"the extraction below no longer matches the shipped code"
    )
    return hits[0]


def _no_args() -> ast.arguments:
    return ast.arguments(
        posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
    )


def _run(statements: list[ast.stmt], namespace: dict) -> dict:
    """Execute lifted statements verbatim; assigned names land back in `namespace`."""
    module = ast.Module(body=list(statements), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "middleware.py", "exec"), namespace)  # noqa: S102
    return namespace


def _run_async_loop_body(statements: list[ast.stmt], namespace: dict):
    """Same, for statements that `await` and `break` out of the enclosing loop."""
    once = ast.For(
        target=ast.Name(id="_once", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(1)], keywords=[]
        ),
        body=list(statements),
        orelse=[],
    )
    # The real branch rebinds these in an enclosing scope; without this they would become
    # locals of the wrapper and the lifted code would read an unbound name.
    rebound = ast.Global(names=["output", "prior_output"])
    function = ast.AsyncFunctionDef(
        name="_extracted",
        args=_no_args(),
        body=[rebound, once],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "middleware.py", "exec"), namespace)  # noqa: S102
    return namespace["_extracted"]()


def _close_reasoning_block(source: str) -> list[ast.stmt]:
    """The stream-end cleanup that closes a still-open reasoning item (#28872)."""
    test = "output[-1].get('type') == 'reasoning'"
    node = _unique(
        source,
        lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == test,
        "reasoning-close branch",
    )
    return [node]


def _tool_result_output_loop(source: str) -> list[ast.stmt]:
    """The loop building `function_call_output` items from tool results (#28016)."""
    node = _unique(
        source,
        lambda n: isinstance(n, ast.For)
        and ast.unparse(n.target) == "result"
        and ast.unparse(n.iter) == "results",
        "tool-result output loop",
    )
    return [node]


def _follow_up_response_branches(source: str) -> list[ast.If]:
    """Both dispatches on the response to the post-tool follow-up request (#28633)."""
    hits = [
        node
        for node in ast.walk(_handler(source))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "isinstance(res, StreamingResponse)"
    ]
    assert len(hits) == 2, (
        f"expected two follow-up response dispatches in `streaming_chat_response_handler`, "
        f"found {len(hits)}"
    )
    return hits


def _nested_function(source: str, name: str) -> ast.stmt:
    return _unique(
        source,
        lambda n: isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name,
        f"nested `{name}`",
    )


# -----------------------------------------------------------------------------
# Event builders
# -----------------------------------------------------------------------------


def _message_item(text: str, annotations: list | None = None) -> dict:
    part = {"type": "output_text", "text": text}
    if annotations is not None:
        part["annotations"] = annotations
    return {"type": "message", "id": "msg_1", "role": "assistant", "content": [part]}


def _completed(output) -> dict:
    return {"type": "response.completed", "response": {"id": "resp_1", "output": output}}


def _text(output: list) -> str:
    return "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


# =============================================================================
# NARROW — #27800 / #27810: an empty terminal output must not erase the reply
# =============================================================================


def test_completed_with_empty_output_keeps_the_streamed_reply(handle_event):
    streamed = [_message_item("the answer is 42")]
    new_output, metadata = handle_event(_completed([]), streamed)

    assert _text(new_output) == "the answer is 42", (
        "a provider reporting `output: []` on response.completed erased the accumulated reply "
        "(#27800): the message vanishes and an empty one is persisted as next turn's context"
    )
    assert metadata["done"] is True


def test_completed_with_empty_output_keeps_mid_stream_citations(handle_event):
    streamed = [_message_item("cited", annotations=[{"type": "url_citation", "url": "https://x"}])]
    new_output, _ = handle_event(_completed([]), streamed)

    annotations = new_output[0]["content"][0].get("annotations")
    assert annotations == [{"type": "url_citation", "url": "https://x"}], (
        "citations that arrived mid-stream were dropped with the rest of the reply (#27800)"
    )


def test_completed_with_empty_output_keeps_reasoning_and_tool_items(handle_event):
    streamed = [
        {"type": "reasoning", "id": "r1", "status": "in_progress", "summary": []},
        {"type": "function_call", "id": "fc1", "call_id": "call_1", "name": "search"},
        _message_item("done"),
    ]
    new_output, _ = handle_event(_completed([]), streamed)

    assert [item["type"] for item in new_output] == ["reasoning", "function_call", "message"]


# =============================================================================
# NARROW — #27810: response.output_item.done must be applied, not shadowed
# =============================================================================


def test_output_item_done_replaces_the_accumulated_item(handle_event):
    streamed = [_message_item("partial")]
    finished = _message_item("partial and complete", annotations=[{"type": "url_citation"}])

    new_output, metadata = handle_event(
        {"type": "response.output_item.done", "output_index": 0, "item": finished},
        streamed,
    )

    assert _text(new_output) == "partial and complete", (
        "the generic `response.*.done` branch shadowed the dedicated output_item.done branch "
        "(#27810), so a provider that skips content_part events left the reply unextractable"
    )
    assert new_output[0]["content"][0]["annotations"] == [{"type": "url_citation"}]
    assert metadata == {}


def test_output_item_done_appends_an_item_the_stream_never_announced(handle_event):
    finished = {"type": "reasoning", "id": "r1", "status": "completed", "summary": [{"text": "hm"}]}

    new_output, _ = handle_event(
        {"type": "response.output_item.done", "output_index": 7, "item": finished},
        [],
    )

    assert new_output == [finished]


# =============================================================================
# NARROW — #28312: an out-of-order delta must not crash the handler
# =============================================================================


@pytest.mark.parametrize(
    "current_output, output_index",
    [
        ([], 0),
        ([_message_item("hi")], 5),
        ([_message_item("hi")], -3),
    ],
    ids=["no-item-yet", "index-past-end", "negative-index"],
)
def test_delta_without_its_output_item_returns_the_accumulated_output(
    handle_event, current_output, output_index
):
    result = handle_event(
        {"type": "response.output_text.delta", "output_index": output_index, "delta": "x"},
        current_output,
    )

    assert result == (current_output, None), (
        "a delta arriving before its output item left `new_output` unbound and raised "
        "UnboundLocalError, cutting the reply short and skipping the outlet filters (#28312)"
    )


def test_two_part_delta_event_name_still_returns_a_tuple(handle_event):
    result = handle_event({"type": "response.delta", "delta": "x"}, [_message_item("hi")])

    assert result is not None, (
        "a two-part `response.*.delta` event name fell off the end of the branch and returned "
        "None, so both call sites raised TypeError unpacking it (#28312)"
    )
    new_output, metadata = result
    assert _text(new_output) == "hi"
    assert metadata is None


def test_two_part_done_event_name_still_returns_a_tuple(handle_event):
    result = handle_event({"type": "response.done"}, [_message_item("hi")])

    assert result is not None
    new_output, _ = result
    assert _text(new_output) == "hi"


# =============================================================================
# NARROW — #28872: closing a reasoning item that never recorded a start time
# =============================================================================


@pytest.mark.parametrize(
    "started_at",
    [None, "absent"],
    ids=["started-at-none", "started-at-missing"],
)
def test_reasoning_item_without_start_time_closes_cleanly(middleware_source, started_at):
    item = {"type": "reasoning", "id": "r1", "status": "in_progress", "ended_at": None}
    if started_at is None:
        item["started_at"] = None

    import time

    namespace = {"output": [item], "time": time}
    _run(_close_reasoning_block(middleware_source), namespace)

    assert item["status"] == "completed", (
        "a stream ending inside a reasoning item with no answer text hit "
        "int(ended_at - started_at) with no start time and failed the whole turn (#28872)"
    )
    assert item["ended_at"] is not None
    assert "duration" not in item, "no start time means no duration, not a fabricated one"


def test_reasoning_item_with_start_time_still_gets_a_duration(middleware_source):
    import time

    item = {
        "type": "reasoning",
        "id": "r1",
        "status": "in_progress",
        "ended_at": None,
        "started_at": time.time() - 5,
    }
    _run(_close_reasoning_block(middleware_source), {"output": [item], "time": time})

    assert item["status"] == "completed"
    assert item["duration"] >= 4


def test_already_closed_reasoning_item_is_left_alone(middleware_source):
    import time

    item = {
        "type": "reasoning",
        "id": "r1",
        "status": "completed",
        "ended_at": 100.0,
        "started_at": 90.0,
        "duration": 10,
    }
    _run(_close_reasoning_block(middleware_source), {"output": [item], "time": time})

    assert item["ended_at"] == 100.0
    assert item["duration"] == 10


# =============================================================================
# NARROW / BROAD — #28016: recognising a failed tool result
# =============================================================================

ERROR_RESULTS = [
    "Error: connection refused",
    "  error: connection refused  ",
    "ERROR: CONNECTION REFUSED",
    "Exception: KeyError('id')",
    "Traceback (most recent call last):\n  File ...",
    "HTTP Error! status: 500",
    '{"error": "boom"}',
    '{"error": {"code": 500, "message": "boom"}}',
    '{"error": ["boom"]}',
    '{"status": "error"}',
    '{"status": "Failed"}',
    '{"success": false, "message": "could not fetch"}',
    '{"ok": false, "message": {"reason": "denied"}}',
    '"{\\"error\\": \\"double encoded\\"}"',
    {"error": "boom"},
    {"status": "FAILED"},
    {"success": False, "message": "nope"},
]

SUCCESSFUL_RESULTS = [
    "",
    "   ",
    "The weather in Paris is 18C.",
    "no error occurred",
    "errors are unlikely here",
    '{"result": 42}',
    '{"error": ""}',
    '{"error": "   "}',
    '{"error": null}',
    '{"error": {}}',
    '{"error": []}',
    '{"status": "success"}',
    '{"status": "ok"}',
    '{"success": true}',
    '{"success": false}',
    '{"ok": false}',
    '{"ok": false, "message": ""}',
    "[1, 2, 3]",
    "42",
    None,
    42,
    [],
    {},
    {"content": "fine"},
]


@pytest.fixture(scope="session")
def is_tool_result_error(middleware_module):
    helper = getattr(middleware_module, "_is_tool_result_error", None)
    assert helper is not None, (
        "middleware._is_tool_result_error is missing: every tool result is stamped completed, "
        "so a failed call is reported to the user and the model as a success (#28016)"
    )
    return helper


@pytest.mark.parametrize("value", ERROR_RESULTS, ids=lambda v: repr(v)[:48])
def test_error_shaped_tool_results_are_recognised(is_tool_result_error, value):
    assert is_tool_result_error(value) is True


@pytest.mark.parametrize("value", SUCCESSFUL_RESULTS, ids=lambda v: repr(v)[:48])
def test_successful_tool_results_are_not_flagged(is_tool_result_error, value):
    assert is_tool_result_error(value) is False


def test_failed_tool_result_is_stamped_failed_at_the_call_site(
    middleware_source, middleware_module
):
    """Drive the shipped loop that turns tool results into `function_call_output` items."""
    namespace = {
        "results": [{"tool_call_id": "call_1", "content": "Error: connection refused"}],
        "output": [],
        "output_id": lambda prefix: f"{prefix}_1",
        "result_status_by_call_id": {},
        "_is_tool_result_error": getattr(middleware_module, "_is_tool_result_error", None),
    }
    _run(_tool_result_output_loop(middleware_source), namespace)

    assert namespace["output"][0]["status"] == "failed", (
        "a tool call that raised was written to the stored output as completed (#28016), so the "
        "UI showed a green tool call and the model was told it succeeded"
    )


def test_successful_tool_result_is_still_stamped_completed_at_the_call_site(
    middleware_source, middleware_module
):
    namespace = {
        "results": [{"tool_call_id": "call_1", "content": '{"temp": 18}'}],
        "output": [],
        "output_id": lambda prefix: f"{prefix}_1",
        "result_status_by_call_id": {},
        "_is_tool_result_error": getattr(middleware_module, "_is_tool_result_error", None),
    }
    _run(_tool_result_output_loop(middleware_source), namespace)

    item = namespace["output"][0]
    assert item["status"] == "completed"
    assert item["call_id"] == "call_1"
    assert item["output"] == [{"type": "input_text", "text": '{"temp": 18}'}]


# =============================================================================
# NARROW / BROAD — #28633: reporting an HTTP error on the post-tool follow-up
# =============================================================================


@pytest.fixture(scope="session")
def get_response_error_detail(misc_module):
    helper = getattr(misc_module, "get_response_error_detail", None)
    assert helper is not None, (
        "misc.get_response_error_detail is missing: a >= 400 response to the follow-up request "
        "made after a tool ran was swallowed and the reply ended silently (#28633)"
    )
    return helper


def _response(body, status_code=502):
    return types.SimpleNamespace(status_code=status_code, body=body)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_response(b'{"error": {"message": "upstream exploded"}}'), "upstream exploded"),
        (_response('{"error": {"message": "upstream exploded"}}'), "upstream exploded"),
        (_response(b'{"error": "rate limited"}'), "rate limited"),
        (_response(b'{"detail": "model not found"}'), "model not found"),
        (_response(b'{"message": "bad request"}'), "bad request"),
        (_response(b'{"error": {"detail": "nested detail"}}'), "nested detail"),
        (_response(b'"a bare string body"'), "a bare string body"),
        (_response(b'{"error": {"code": 500}}'), "{'code': 500}"),
        (_response(b'{"message": 7}'), "7"),
        (_response(b"not json at all"), "Provider returned HTTP 502"),
        (_response(b""), "Provider returned HTTP 502"),
        (_response(b'{"unrelated": "shape"}'), "{'unrelated': 'shape'}"),
        (types.SimpleNamespace(status_code=429), "Provider returned HTTP 429"),
        (types.SimpleNamespace(), "Provider returned an error"),
    ],
    ids=[
        "bytes-error-message",
        "str-error-message",
        "error-string",
        "detail",
        "message",
        "nested-detail",
        "bare-string-body",
        "unwalkable-dict",
        "non-string-leaf",
        "non-json-body",
        "empty-body",
        "no-known-key",
        "no-body",
        "no-status-code",
    ],
)
def test_provider_error_detail_extraction(get_response_error_detail, response, expected):
    assert get_response_error_detail(response) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("branch_index", [0, 1], ids=["with-prior-output", "plain"])
async def test_http_error_after_a_tool_ran_is_reported_to_the_user(
    middleware_source, misc_module, branch_index
):
    """Drive the shipped follow-up dispatch with a 502 instead of a stream."""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    emitted = []

    async def emit_message_error(error_content):
        emitted.append(error_content)

    async def stream_body_handler(res, form_data):
        raise AssertionError("a 502 must not be treated as a stream")

    namespace = {
        "res": _response(b'{"error": {"message": "upstream exploded"}}'),
        "StreamingResponse": StreamingResponse,
        "HTTPException": HTTPException,
        "stream_body_handler": stream_body_handler,
        "emit_message_error": emit_message_error,
        "get_response_error_detail": getattr(misc_module, "get_response_error_detail", None),
        "new_form_data": {},
        "output": [],
        "prior_output": [],
        "full_output": lambda: [],
    }
    _run([_nested_function(middleware_source, "get_message_error_content")], namespace)

    branch = _follow_up_response_branches(middleware_source)[branch_index]
    await _run_async_loop_body([branch], namespace)

    assert emitted == ["upstream exploded"], (
        "the handler just broke out of the loop on a >= 400 follow-up response (#28633), so the "
        "turn ended with the tool run shown and nothing said afterwards"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("branch_index", [0, 1], ids=["with-prior-output", "plain"])
async def test_successful_follow_up_response_is_still_streamed(
    middleware_source, misc_module, branch_index
):
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    streamed = []

    async def body():
        yield b"data: [DONE]\n\n"

    async def stream_body_handler(res, form_data):
        streamed.append(res)

    async def emit_message_error(error_content):
        raise AssertionError("a streaming response must not be reported as an error")

    namespace = {
        "res": StreamingResponse(body()),
        "StreamingResponse": StreamingResponse,
        "HTTPException": HTTPException,
        "stream_body_handler": stream_body_handler,
        "emit_message_error": emit_message_error,
        "get_response_error_detail": getattr(misc_module, "get_response_error_detail", None),
        "new_form_data": {},
        "output": [],
        "prior_output": [],
        "full_output": lambda: [],
    }
    _run([_nested_function(middleware_source, "get_message_error_content")], namespace)

    branch = _follow_up_response_branches(middleware_source)[branch_index]
    await _run_async_loop_body([branch], namespace)

    assert len(streamed) == 1


# =============================================================================
# NEARBY — the streaming paths that were already correct
# =============================================================================


def test_completed_with_a_real_output_replaces_the_accumulation(handle_event):
    final = [_message_item("the provider's own final copy")]
    new_output, metadata = handle_event(_completed(final), [_message_item("partial")])

    assert _text(new_output) == "the provider's own final copy"
    assert metadata["response_id"] == "resp_1"


def test_completed_without_an_output_key_keeps_the_accumulation(handle_event):
    new_output, _ = handle_event(_completed(None), [_message_item("kept")])

    assert _text(new_output) == "kept"


def test_completed_marks_open_reasoning_items_completed(handle_event):
    streamed = [{"type": "reasoning", "id": "r1", "status": "in_progress"}, _message_item("hi")]
    new_output, _ = handle_event(_completed(None), streamed)

    assert new_output[0]["status"] == "completed"


def test_text_deltas_accumulate_onto_their_message_item(handle_event):
    output = [{"type": "message", "id": "msg_1", "role": "assistant"}]
    for chunk in ("Hel", "lo ", "world"):
        output, _ = handle_event(
            {"type": "response.output_text.delta", "output_index": 0, "delta": chunk}, output
        )

    assert _text(output) == "Hello world"


def test_function_call_argument_deltas_accumulate(handle_event):
    output = [{"type": "function_call", "id": "fc1", "call_id": "call_1", "name": "search"}]
    for chunk in ('{"q":', ' "cats"}'):
        output, _ = handle_event(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": chunk,
            },
            output,
        )

    assert output[0]["arguments"] == '{"q": "cats"}'


def test_reasoning_summary_deltas_land_on_the_summary(handle_event):
    output = [{"type": "reasoning", "id": "r1", "status": "in_progress"}]
    output, _ = handle_event(
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "delta": "thinking",
        },
        output,
    )

    assert output[0]["summary"][0]["text"] == "thinking"


def test_output_item_added_appends_the_item(handle_event):
    item = {"type": "message", "id": "msg_1", "role": "assistant", "content": []}
    new_output, metadata = handle_event(
        {"type": "response.output_item.added", "output_index": 0, "item": item}, []
    )

    assert new_output == [item]
    assert metadata is None


def test_unknown_event_is_a_no_op(handle_event):
    streamed = [_message_item("hi")]
    assert handle_event({"type": "response.something.unheard_of"}, streamed) == (streamed, None)
    assert handle_event({}, streamed) == (streamed, None)


def test_response_failed_surfaces_the_error(handle_event):
    streamed = [_message_item("hi")]
    new_output, metadata = handle_event(
        {"type": "response.failed", "response": {"error": {"message": "nope"}}}, streamed
    )

    assert new_output == streamed
    assert metadata == {"error": {"message": "nope"}}


def test_the_handler_never_mutates_the_output_it_was_given(handle_event):
    streamed = [_message_item("original")]
    snapshot = [_message_item("original")]

    handle_event({"type": "response.output_text.delta", "output_index": 0, "delta": "x"}, streamed)
    handle_event(
        {"type": "response.output_item.done", "output_index": 0, "item": _message_item("z")},
        streamed,
    )

    assert streamed == snapshot
