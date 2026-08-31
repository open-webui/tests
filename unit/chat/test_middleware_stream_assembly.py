"""Regression tests for six 0.11.0 and two 0.11.2 repairs in `open_webui/utils/middleware.py`.

All of them sit on the path that assembles a chat turn: the payload the model is asked with,
the items a stream accumulates, and what survives a page reload.

* #27414 / #27017 (`381149ea5`, `outlet_filter_handler`): the outlet payload embedded
  `m['output']` by reference, so a filter mutating `output` in place mutated the stored
  baseline too. The change check then compared the object with itself, found nothing changed
  and never persisted: the edit showed up live and vanished on reload. Fixed with
  `copy.deepcopy`.
* #26687 / #26645 (`051a1f6`, `streaming_chat_response_handler`): reasoning arriving after a
  `message` item already existed was appended after it, so the thinking block rendered below
  the answer; and a `reasoning_details` payload carrying no text, summary or data opened an
  empty thinking block. The fix inserts late reasoning at the message index (stamped
  completed) and only opens a block for details that carry content.
* #26857 / #26836 (`d3cfcd801`, `process_chat_payload`): `metadata['system_prompt']` read the
  model default from `form_data['params']['system']` after `apply_params_to_form_data` had
  already popped `params`, so it was always empty. The native tool-call loop runs with
  `bypass_system_prompt=True` and restores from that metadata, so from the second round the
  model's system prompt was gone, leaving only the injected `<memory_context>` block.
* #26986 (`b9d7274`, `process_chat_payload`): `skill_ids` was a raw set union, so the order
  skills were injected in varied per process and defeated provider prompt caching. Now
  `sorted`.
* #27426 / #27411 (`8ab44ed`, `streaming_chat_response_handler`): a reply that raised while
  continuing after a tool call or a code-interpreter run logged at debug and broke out, ending
  mid-answer with nothing said. The fix emits `chat:message:error` and persists it.
* #27365 / #27074 (`dd514ee20`, `streaming_chat_response_handler`): the plain-JSON (no
  `data:` prefix) error branch called the message upsert without `await`, so the coroutine
  never ran and the error vanished on reload.
* #29053 / #29035 (`3749e7dc7`, `streaming_chat_response_handler`): a chunk carrying both
  reasoning text and `reasoning_details`, which is what OpenRouter sends, had its already
  built `response.reasoning_text.delta` cleared by the details branch, so nothing was emitted
  for the whole reasoning phase and the answer only appeared once generation finished. The
  event is now dropped only when the details were all the chunk reported.
* #29052 / #29040 (`87bed3f0b`, `streaming_chat_response_handler`): a finished tool round
  appended an empty `message` item for the next response. Reasoning routing keys off the first
  `message` item, so the next round's thinking was folded into the previous, already closed
  Thoughts block instead of opening a live one. The pre-append is gone.

`outlet_filter_handler` and `process_chat_payload` are module-level and are driven for real.
The other sites live inside `streaming_chat_response_handler` and `process_chat_payload`,
functions too large to construct in isolation, so the shipped statements are lifted out of the
real middleware source with `ast` and executed against a namespace prefilled from the module's
own globals. Nothing is reimplemented: the code under test is the code that ships.

Two 0.11.1 refactors are absorbed at runtime rather than pinned: `dbf715cb6` (#28798) folded
`Skills.get_skills_by_user_id` into `get_skills(user_id=, ids=)`, and `d198d950c` (#28821)
made the reasoning branch await the stream save, so the lifted block now runs in a coroutine.
Neither changes what the assertions check.

Discriminates: passes on v0.11.2, fails on v0.10.2 for the 0.11.0 sections (an outlet edit to
`output` is never persisted, late reasoning lands below the answer, a contentless
reasoning_details opens an empty thinking block, the model system prompt is missing from the
tool-call metadata, `skill_ids` iterates in set order, a continuation failure is silent, and a
plain-JSON error line is never written to the chat) and fails on v0.11.1 for the 0.11.2
sections (a reasoning delta that arrives with reasoning_details is emitted as nothing, and the
message item a tool round pre-appends diverts the next round's thinking into a closed Thoughts
block).
"""

from __future__ import annotations

import ast
import copy
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def filter_module(owui_module):
    return owui_module("open_webui.utils.filter")


@pytest.fixture(scope="session")
def middleware_source(open_webui_backend: Path) -> str:
    return (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text(
        encoding="utf-8"
    )


@pytest.fixture(scope="session")
def user_model(owui_module):
    return owui_module("open_webui.models.users").UserModel


@pytest.fixture(scope="session")
def groups_model(owui_module):
    return owui_module("open_webui.models.groups").Groups


@pytest.fixture(scope="session")
def skills_model(owui_module):
    return owui_module("open_webui.models.skills").Skills


def _user(user_model, role="user"):
    return user_model(
        id="alice",
        name="Alice",
        email="alice@example.com",
        role=role,
        profile_image_url="",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


def _request():
    app = types.SimpleNamespace(state=types.SimpleNamespace(MODELS={"m": {"id": "m"}}))
    return types.SimpleNamespace(
        app=app,
        cookies={},
        headers={},
        state=types.SimpleNamespace(direct=False, internal=False),
    )


# -----------------------------------------------------------------------------
# Lifting shipped statements out of the oversized coroutines
# -----------------------------------------------------------------------------


def _function(source: str, name: str) -> ast.AST:
    hits = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(hits) == 1, f"expected one `{name}` in middleware.py, found {len(hits)}"
    return hits[0]


def _blocks(node: ast.AST):
    for inner in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(inner, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _assigns(block: list[ast.stmt], target: str):
    return [
        index
        for index, node in enumerate(block)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == target
    ]


def _namespace(middleware_module, statements: list[ast.stmt], **overrides) -> dict:
    """Prefill every free name the lifted code reads from the module's own globals.

    Keeps the extraction working across refs that renamed a dependency (`json` vs
    `JSONCodec`) without the test having to know which one this checkout ships.
    """
    module_globals = vars(middleware_module)
    namespace = {"__builtins__": __builtins__}
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in module_globals:
                namespace[node.id] = module_globals[node.id]
    namespace.update(overrides)
    return namespace


def _no_args() -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def _exec(statements: list[ast.stmt], namespace: dict) -> dict:
    """Execute lifted statements verbatim; assigned names land back in `namespace`."""
    module = ast.Module(body=list(statements), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "middleware.py", "exec"), namespace)  # noqa: S102
    return namespace


def _rebound_names(statements: list[ast.stmt]) -> list[str]:
    """Names the lifted code assigns; they must stay in the namespace, not become locals."""
    names = {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return sorted(names)


def _exec_async_loop_body(statements: list[ast.stmt], namespace: dict):
    """Same, for statements that `await` and may `break` or `continue`."""
    once = ast.For(
        target=ast.Name(id="_once", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(1)], keywords=[]
        ),
        body=list(statements),
        orelse=[],
    )
    function = ast.AsyncFunctionDef(
        name="_extracted",
        args=_no_args(),
        body=[ast.Global(names=_rebound_names(statements)), once],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "middleware.py", "exec"), namespace)  # noqa: S102
    return namespace["_extracted"]()


def _reasoning_delta_block(source: str) -> list[ast.stmt]:
    """The `delta.reasoning*` accumulation, from the content read to the end of its branch."""
    for block in _blocks(_function(source, "streaming_chat_response_handler")):
        if not _assigns(block, "reasoning_details"):
            continue
        start = _assigns(block, "reasoning_content")[0]
        end = next(
            index
            for index, node in enumerate(block)
            if index > start
            and isinstance(node, ast.If)
            and ast.unparse(node.test).startswith("reasoning_content or")
        )
        return block[start : end + 1]
    raise AssertionError("reasoning delta block no longer matches the shipped middleware")


def _skill_ids_block(source: str) -> list[ast.stmt]:
    """The two statements resolving which skills this turn injects, in order."""
    for block in _blocks(_function(source, "process_chat_payload")):
        targets = _assigns(block, "skill_ids")
        if not targets:
            continue
        mentioned = _assigns(block, "mentioned_skill_ids")[0]
        return block[mentioned : targets[0] + 1]
    raise AssertionError("skill_ids resolution no longer matches the shipped middleware")


def _continuation_error_handlers(source: str) -> list[ast.ExceptHandler]:
    """The two `except` arms guarding the post-tool and post-code-interpreter continuations."""
    hits = [
        node
        for node in ast.walk(_function(source, "streaming_chat_response_handler"))
        if isinstance(node, ast.ExceptHandler)
        and node.body
        and isinstance(node.body[-1], ast.Break)
    ]
    assert len(hits) == 2, (
        f"expected two break-ending except arms in `streaming_chat_response_handler`, "
        f"found {len(hits)}"
    )
    return hits


def _message_error_helpers(source: str) -> list[ast.stmt]:
    """`get_message_error_content` / `emit_message_error`; absent before `8ab44ed`."""
    return [
        node
        for node in ast.walk(_function(source, "streaming_chat_response_handler"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("get_message_error_content", "emit_message_error")
    ]


def _raw_json_error_branch(source: str) -> list[ast.stmt]:
    """The try that normalizes a plain-JSON (non-SSE) error line into an error event."""
    hits = [
        node
        for node in ast.walk(_function(source, "streaming_chat_response_handler"))
        if isinstance(node, ast.Try)
        and node.body
        and isinstance(node.body[0], ast.Assign)
        and ast.unparse(node.body[0].targets[0]) == "raw_obj"
    ]
    assert len(hits) == 1, f"expected one raw-JSON error branch, found {len(hits)}"
    return hits


def _delta_block(source: str) -> list[ast.stmt]:
    """The whole per-chunk delta branch: `value = delta['content']` through `if value:`."""
    for block in _blocks(_function(source, "streaming_chat_response_handler")):
        if not _assigns(block, "reasoning_details"):
            continue
        start = _assigns(block, "value")[0]
        end = next(
            index
            for index, node in enumerate(block)
            if index > start and isinstance(node, ast.If) and ast.unparse(node.test) == "value"
        )
        return block[start : end + 1]
    raise AssertionError("delta block no longer matches the shipped middleware")


def _tool_round_block(source: str) -> list[ast.stmt]:
    """What a finished tool round leaves in `output`, up to the citation emit."""
    for block in _blocks(_function(source, "streaming_chat_response_handler")):
        targets = _assigns(block, "result_status_by_call_id")
        if not targets:
            continue
        end = next(
            index
            for index, node in enumerate(block)
            if index > targets[0]
            and isinstance(node, ast.If)
            and ast.unparse(node.test) == "citations_enabled"
        )
        return block[targets[0] : end]
    raise AssertionError("tool round block no longer matches the shipped middleware")


# =============================================================================
# #27414: an outlet filter's in-place edit to `output` must be persisted
# =============================================================================

CHAT_ID = "chat-outlet-aliasing"
MESSAGE_ID = "msg-outlet-aliasing"


def _stored_message(text: str) -> dict:
    return {
        "id": MESSAGE_ID,
        "parentId": None,
        "role": "assistant",
        "content": text,
        "output": [
            {
                "type": "message",
                "id": "out-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


class _FilterFunction:
    """The DB row the filter chain iterates; only `.id` is read."""

    def __init__(self, function_id: str) -> None:
        self.id = function_id


@contextmanager
def _outlet_pipeline(middleware_module, filter_module, outlet, upsert, messages_map):
    """Serve the plugin, pipeline and chat-store boundaries; the real filter chain runs.

    The two refs load filters differently (`get_filter_functions` versus
    `get_sorted_filter_ids` plus a `Functions` lookup), so whichever the checkout ships is
    the one that gets served.
    """
    chats = types.SimpleNamespace(
        get_messages_map_by_chat_id=AsyncMock(return_value=messages_map),
        upsert_message_to_chat_by_id_and_message_id=upsert,
    )
    filters = [_FilterFunction("edit_output")]

    async def passthrough_pipeline(request, form_data, user, models):
        return form_data

    async def function_module(request, function_id, **kwargs):
        return types.SimpleNamespace(outlet=outlet)

    boundaries = {
        middleware_module: {
            "Chats": chats,
            "ENABLE_PLUGINS": True,
            "process_pipeline_outlet_filter": AsyncMock(side_effect=passthrough_pipeline),
            "get_filter_functions": AsyncMock(return_value=filters),
            "get_sorted_filter_ids": AsyncMock(return_value=["edit_output"]),
            "Functions": types.SimpleNamespace(
                get_functions_by_ids=AsyncMock(return_value=filters)
            ),
        },
        filter_module: {
            "ENABLE_PLUGINS": True,
            "get_function_module": function_module,
        },
    }
    with ExitStack() as stack:
        for module, replacements in boundaries.items():
            for name, value in replacements.items():
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, value))
        yield


async def _run_outlet(middleware_module, filter_module, user_model, outlet, messages_map):
    upsert = AsyncMock(return_value=None)
    emitted = []

    async def event_emitter(event):
        emitted.append(event)

    ctx = {
        "request": _request(),
        "user": _user(user_model),
        "model": {"id": "m"},
        "metadata": {"chat_id": CHAT_ID, "message_id": MESSAGE_ID, "filter_ids": ["edit_output"]},
        "event_emitter": event_emitter,
        "event_caller": None,
    }
    with _outlet_pipeline(middleware_module, filter_module, outlet, upsert, messages_map):
        await middleware_module.outlet_filter_handler(ctx)
    # The handler swallows its own exceptions, so prove it reached the end before reading
    # anything into the absence of an upsert.
    assert [event["type"] for event in emitted] == ["chat:outlet"], (
        "outlet_filter_handler did not run to completion; the harness, not the code, is wrong"
    )
    return upsert


def _append_part(body):
    """An outlet filter that edits the structured output in place, as plugins do."""
    body["messages"][-1]["output"][0]["content"].append(
        {"type": "output_text", "text": " [reviewed]"}
    )
    return body


@pytest.mark.asyncio
async def test_outlet_edit_to_output_is_persisted(middleware_module, filter_module, user_model):
    messages_map = {MESSAGE_ID: _stored_message("hello")}
    upsert = await _run_outlet(
        middleware_module, filter_module, user_model, _append_part, messages_map
    )

    assert upsert.await_count == 1, (
        "an outlet filter's in-place edit to the structured output was never persisted "
        "(#27414): the outlet payload aliased messages_map, so the change check compared the "
        "mutated object with itself and the edit was lost on reload"
    )
    chat_id, message_id, update = upsert.await_args.args
    assert (chat_id, message_id) == (CHAT_ID, MESSAGE_ID)
    assert update["output"][0]["content"][-1]["text"] == " [reviewed]"


@pytest.mark.asyncio
async def test_outlet_payload_does_not_alias_the_stored_baseline(
    middleware_module, filter_module, user_model
):
    messages_map = {MESSAGE_ID: _stored_message("hello")}
    baseline = copy.deepcopy(messages_map[MESSAGE_ID]["output"])
    await _run_outlet(middleware_module, filter_module, user_model, _append_part, messages_map)

    assert messages_map[MESSAGE_ID]["output"] == baseline, (
        "the filter mutated the pre-filter baseline itself, which is what blinded the change "
        "check and what corrupts originalContent for output-only messages (#27414)"
    )


@pytest.mark.asyncio
async def test_outlet_rewrite_of_output_text_is_persisted(
    middleware_module, filter_module, user_model
):
    def rewrite_text(body):
        body["messages"][-1]["output"][0]["content"][0]["text"] = "REDACTED"
        return body

    messages_map = {MESSAGE_ID: _stored_message("secret")}
    upsert = await _run_outlet(
        middleware_module, filter_module, user_model, rewrite_text, messages_map
    )

    assert upsert.await_count == 1, (
        "a redacting outlet filter that rewrites output text in place was silently dropped "
        "on reload (#27414)"
    )
    update = upsert.await_args.args[2]
    assert update["output"][0]["content"][0]["text"] == "REDACTED"
    assert update["originalContent"] == "secret"


# --- Nearby: unchanged output, and content-only edits, behave as before ------


@pytest.mark.asyncio
async def test_outlet_without_changes_persists_nothing(
    middleware_module, filter_module, user_model
):
    def noop(body):
        return body

    messages_map = {MESSAGE_ID: _stored_message("hello")}
    upsert = await _run_outlet(middleware_module, filter_module, user_model, noop, messages_map)

    assert upsert.await_count == 0


@pytest.mark.asyncio
async def test_outlet_content_edit_is_still_persisted(
    middleware_module, filter_module, user_model
):
    def rewrite_content(body):
        body["messages"][-1]["content"] = "rewritten"
        return body

    messages_map = {MESSAGE_ID: _stored_message("hello")}
    upsert = await _run_outlet(
        middleware_module, filter_module, user_model, rewrite_content, messages_map
    )

    assert upsert.await_count == 1
    assert upsert.await_args.args[2]["content"] == "rewritten"


# =============================================================================
# #26687: late reasoning must sit above the answer, empty details add nothing
# =============================================================================


def _message_item(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "status": "in_progress",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


async def _apply_reasoning_delta(middleware_module, middleware_source, delta, output):
    # 0.11.1 (`d198d950c`) awaits the stream save inside this block, so it has to run in a
    # coroutine; the placement logic it wraps is unchanged.
    statements = _reasoning_delta_block(middleware_source)
    namespace = _namespace(
        middleware_module,
        statements,
        delta=delta,
        output=output,
        full_output=lambda: output,
        save_current_response_stream=AsyncMock(return_value=None),
    )
    await _exec_async_loop_body(statements, namespace)
    return output


@pytest.mark.asyncio
async def test_late_reasoning_is_placed_above_the_answer(middleware_module, middleware_source):
    output = await _apply_reasoning_delta(
        middleware_module,
        middleware_source,
        {"reasoning_content": "second thoughts"},
        [_message_item("the answer is 42")],
    )

    assert [item["type"] for item in output] == ["reasoning", "message"], (
        "reasoning that arrived after the answer text was appended below it, so the thinking "
        "block rendered underneath the answer (#26645)"
    )


@pytest.mark.asyncio
async def test_late_reasoning_is_stamped_completed(middleware_module, middleware_source):
    output = await _apply_reasoning_delta(
        middleware_module,
        middleware_source,
        {"reasoning_content": "second thoughts"},
        [_message_item("the answer is 42")],
    )

    reasoning = next(item for item in output if item["type"] == "reasoning")
    assert reasoning["status"] == "completed", (
        "a reasoning block opened after the answer is never closed by the content branch, so "
        "it stays spinning forever unless it is stamped completed on insert (#26645)"
    )
    assert reasoning["duration"] == 0


@pytest.mark.asyncio
async def test_contentless_reasoning_details_open_no_thinking_block(
    middleware_module, middleware_source
):
    output = await _apply_reasoning_delta(
        middleware_module,
        middleware_source,
        {"reasoning_details": [{"type": "reasoning.text", "index": 0}]},
        [],
    )

    assert output == [], (
        "a reasoning_details payload carrying no text, summary or data opened an empty "
        "thinking block (#26645)"
    )


# --- Nearby: ordinary reasoning accumulation is untouched --------------------


@pytest.mark.asyncio
async def test_reasoning_before_any_message_is_appended(middleware_module, middleware_source):
    output = await _apply_reasoning_delta(
        middleware_module, middleware_source, {"reasoning_content": "thinking"}, []
    )

    assert len(output) == 1
    assert output[0]["type"] == "reasoning"
    assert output[0]["status"] == "in_progress"
    assert output[0]["content"][0]["text"] == "thinking"


@pytest.mark.asyncio
async def test_reasoning_chunks_accumulate_into_one_item(middleware_module, middleware_source):
    output = []
    for chunk in ("think", "ing", " hard"):
        await _apply_reasoning_delta(
            middleware_module, middleware_source, {"reasoning": chunk}, output
        )

    assert len(output) == 1
    assert output[0]["content"][0]["text"] == "thinking hard"


@pytest.mark.asyncio
async def test_reasoning_details_with_text_still_open_a_block(middleware_module, middleware_source):
    output = await _apply_reasoning_delta(
        middleware_module,
        middleware_source,
        {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "step one"}]},
        [],
    )

    assert len(output) == 1
    assert output[0]["type"] == "reasoning"
    assert output[0]["reasoning_details"][0]["text"] == "step one"


# =============================================================================
# #26857: the system prompt a tool-call round is rebuilt from
# =============================================================================

MODEL_SYSTEM_PROMPT = "You are Ada, a terse assistant."
MEMORY_BLOCK = "<memory_context>\nUser likes short answers.\n</memory_context>"


SKILL_ACCESSORS = ("get_skills_by_user_id", "get_skills")


@contextmanager
def _no_skills(skills_model):
    """0.11.1 (`dbf715cb6`) folded `get_skills_by_user_id` into `get_skills(user_id=, ids=)`."""
    present = [name for name in SKILL_ACCESSORS if hasattr(skills_model, name)]
    assert present, f"none of {SKILL_ACCESSORS} on this checkout's Skills table"
    with ExitStack() as stack:
        for name in present:
            stack.enter_context(patch.object(skills_model, name, AsyncMock(return_value=[])))
        yield


async def _payload_metadata(middleware_module, groups_model, skills_model, user, form_data, model):
    """Run the real `process_chat_payload` and hand back the metadata it built."""
    metadata = {
        "chat_id": "",
        "session_id": "session-1",
        "message_id": "assistant-1",
        "params": {"function_calling": "native"},
        "features": {},
    }
    with (
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
        _no_skills(skills_model),
    ):
        _form_data, metadata, _events = await middleware_module.process_chat_payload(
            _request(), form_data, user, metadata, model
        )
    return metadata


def _payload(system: str | None, messages: list[dict]) -> dict:
    params = {"function_calling": "native"}
    if system is not None:
        params["system"] = system
    return {"model": "m", "params": params, "messages": messages}


@pytest.mark.asyncio
async def test_model_system_prompt_survives_into_tool_call_metadata(
    middleware_module, groups_model, skills_model, user_model
):
    form_data = _payload(
        MODEL_SYSTEM_PROMPT,
        [{"role": "system", "content": MEMORY_BLOCK}, {"role": "user", "content": "hi"}],
    )
    metadata = await _payload_metadata(
        middleware_module,
        groups_model,
        skills_model,
        _user(user_model),
        form_data,
        {"id": "m"},
    )

    assert MODEL_SYSTEM_PROMPT in (metadata["system_prompt"] or ""), (
        "the model's system prompt was read from form_data['params'] after "
        "apply_params_to_form_data had popped it, so the tool-call loop (which runs with "
        "bypass_system_prompt=True and restores from this metadata) lost it from round two "
        "and was left with the injected memory block alone (#26836)"
    )


@pytest.mark.asyncio
async def test_model_system_prompt_is_captured_without_a_memory_block(
    middleware_module, groups_model, skills_model, user_model
):
    form_data = _payload(MODEL_SYSTEM_PROMPT, [{"role": "user", "content": "hi"}])
    metadata = await _payload_metadata(
        middleware_module,
        groups_model,
        skills_model,
        _user(user_model),
        form_data,
        {"id": "m"},
    )

    assert metadata["system_prompt"] == MODEL_SYSTEM_PROMPT, (
        "without a memory block there was no system message to capture at all, so the "
        "tool-call loop ran with no system prompt whatsoever (#26836)"
    )


# --- Nearby: the cases that were already right ------------------------------


@pytest.mark.asyncio
async def test_memory_block_is_kept_alongside_the_model_prompt(
    middleware_module, groups_model, skills_model, user_model
):
    form_data = _payload(
        MODEL_SYSTEM_PROMPT,
        [{"role": "system", "content": MEMORY_BLOCK}, {"role": "user", "content": "hi"}],
    )
    metadata = await _payload_metadata(
        middleware_module,
        groups_model,
        skills_model,
        _user(user_model),
        form_data,
        {"id": "m"},
    )

    assert MEMORY_BLOCK in metadata["system_prompt"]


@pytest.mark.asyncio
async def test_memory_block_alone_is_still_captured(
    middleware_module, groups_model, skills_model, user_model
):
    form_data = _payload(
        None, [{"role": "system", "content": MEMORY_BLOCK}, {"role": "user", "content": "hi"}]
    )
    metadata = await _payload_metadata(
        middleware_module,
        groups_model,
        skills_model,
        _user(user_model),
        form_data,
        {"id": "m"},
    )

    assert metadata["system_prompt"] == MEMORY_BLOCK


@pytest.mark.asyncio
async def test_no_system_prompt_anywhere_stays_none(
    middleware_module, groups_model, skills_model, user_model
):
    form_data = _payload(None, [{"role": "user", "content": "hi"}])
    metadata = await _payload_metadata(
        middleware_module,
        groups_model,
        skills_model,
        _user(user_model),
        form_data,
        {"id": "m"},
    )

    assert metadata["system_prompt"] is None


# =============================================================================
# #26986: skill order must not depend on set iteration
# =============================================================================

# Enough ids that a set landing in sorted order by chance is not a real risk.
REQUEST_SKILL_IDS = [f"skill-{index:02d}" for index in range(20)]
MODEL_SKILL_IDS = [f"model-skill-{index:02d}" for index in range(10)]


def _resolved_skill_ids(middleware_module, middleware_source, request_ids, model_ids):
    statements = _skill_ids_block(middleware_source)
    namespace = _namespace(
        middleware_module,
        statements,
        form_data={"messages": [{"role": "user", "content": "hi"}], "skill_ids": list(request_ids)},
        model={"info": {"meta": {"skillIds": list(model_ids)}}},
    )
    _exec(statements, namespace)
    return list(namespace["skill_ids"])


def test_skill_ids_are_resolved_in_sorted_order(middleware_module, middleware_source):
    resolved = _resolved_skill_ids(
        middleware_module,
        middleware_source,
        list(reversed(REQUEST_SKILL_IDS)),
        list(reversed(MODEL_SKILL_IDS)),
    )

    assert resolved == sorted(REQUEST_SKILL_IDS + MODEL_SKILL_IDS), (
        "skill_ids came straight off a set union, so the order skills were injected in varied "
        "per process and defeated the provider's prompt-prefix caching (#26986)"
    )


# --- Nearby: the union itself is unchanged ----------------------------------


def test_skill_ids_are_deduplicated(middleware_module, middleware_source):
    resolved = _resolved_skill_ids(
        middleware_module, middleware_source, ["skill-a", "skill-b", "skill-a"], ["skill-b"]
    )

    assert sorted(resolved) == ["skill-a", "skill-b"]


def test_no_skills_resolves_to_nothing(middleware_module, middleware_source):
    assert _resolved_skill_ids(middleware_module, middleware_source, [], []) == []


def test_model_skills_are_included(middleware_module, middleware_source):
    resolved = _resolved_skill_ids(middleware_module, middleware_source, [], ["model-only"])

    assert list(resolved) == ["model-only"]


# =============================================================================
# #27426: a failure while continuing after a tool call must be reported
# =============================================================================


class _Boom(Exception):
    pass


async def _run_continuation_failure(
    middleware_module, middleware_source, index, error, save_to_chat=True
):
    """Drive the shipped `except` arm with an upstream failure raised into it."""
    upsert = AsyncMock(return_value=None)
    emitted = []

    async def event_emitter(event):
        emitted.append(event)

    handler = _continuation_error_handlers(middleware_source)[index]
    helpers = _message_error_helpers(middleware_source)
    namespace = _namespace(
        middleware_module,
        [*helpers, *handler.body],
        Chats=types.SimpleNamespace(upsert_message_to_chat_by_id_and_message_id=upsert),
        metadata={"chat_id": CHAT_ID, "message_id": MESSAGE_ID},
        event_emitter=event_emitter,
        save_to_chat=save_to_chat,
        _boom=error,
    )
    _exec(helpers, namespace)

    guarded = ast.Try(
        body=[ast.Raise(exc=ast.Name(id="_boom", ctx=ast.Load()), cause=None)],
        handlers=[handler],
        orelse=[],
        finalbody=[],
    )
    await _exec_async_loop_body([guarded], namespace)
    return upsert, emitted


@pytest.mark.asyncio
@pytest.mark.parametrize(("index", "what"), [(0, "tool call"), (1, "code interpreter run")])
async def test_continuation_failure_is_reported(
    middleware_module, middleware_source, index, what
):
    upsert, emitted = await _run_continuation_failure(
        middleware_module, middleware_source, index, _Boom("upstream refused")
    )

    assert [event["type"] for event in emitted] == ["chat:message:error"], (
        f"a reply that raised while continuing after a {what} logged at debug and broke out, "
        f"so the answer stopped mid-sentence with nothing said (#27411)"
    )
    assert "upstream refused" in str(emitted[0]["data"]["error"]["content"])
    assert upsert.await_count == 1, (
        "the failure has to be written to the message too, or it disappears on reload (#27411)"
    )


@pytest.mark.asyncio
async def test_continuation_failure_reports_the_http_detail(
    middleware_module, middleware_source
):
    error = middleware_module.HTTPException(status_code=502, detail="provider unavailable")
    _, emitted = await _run_continuation_failure(middleware_module, middleware_source, 0, error)

    assert emitted[0]["data"]["error"]["content"] == "provider unavailable", (
        "an HTTPException must surface its detail, not its repr (#27411)"
    )


@pytest.mark.asyncio
async def test_continuation_failure_on_an_unsaved_chat_still_emits(
    middleware_module, middleware_source
):
    upsert, emitted = await _run_continuation_failure(
        middleware_module, middleware_source, 0, _Boom("nope"), save_to_chat=False
    )

    assert upsert.await_count == 0, "an unsaved chat has no message row to write to"
    assert [event["type"] for event in emitted] == ["chat:message:error"]


# =============================================================================
# #27365: a plain-JSON error line must actually reach the chat
# =============================================================================


async def _run_raw_json_error(middleware_module, middleware_source, line, save_to_chat=True):
    upsert = AsyncMock(return_value=None)
    emitted = []

    async def event_emitter(event):
        emitted.append(event)

    statements = _raw_json_error_branch(middleware_source)
    namespace = _namespace(
        middleware_module,
        statements,
        Chats=types.SimpleNamespace(upsert_message_to_chat_by_id_and_message_id=upsert),
        metadata={"chat_id": CHAT_ID, "message_id": MESSAGE_ID},
        event_emitter=event_emitter,
        save_to_chat=save_to_chat,
        data=line,
    )
    await _exec_async_loop_body(statements, namespace)
    return upsert, emitted


@pytest.mark.asyncio
async def test_raw_json_error_line_is_persisted(middleware_module, middleware_source):
    upsert, emitted = await _run_raw_json_error(
        middleware_module, middleware_source, '{"error": "rate limit exceeded"}'
    )

    assert upsert.await_count == 1, (
        "the upsert was called without await, so the coroutine never ran and the upstream "
        "error left no trace in the chat after a reload (#27074)"
    )
    assert emitted[0]["data"]["error"] == "rate limit exceeded"


@pytest.mark.asyncio
async def test_raw_json_error_line_is_written_to_the_right_message(
    middleware_module, middleware_source
):
    upsert, _ = await _run_raw_json_error(
        middleware_module, middleware_source, '{"error": {"message": "boom"}}'
    )

    chat_id, message_id, update = upsert.await_args.args
    assert (chat_id, message_id) == (CHAT_ID, MESSAGE_ID)
    assert update["error"]["content"] == {"message": "boom"}


# --- Nearby: lines that are not errors, and unsaved chats -------------------


@pytest.mark.asyncio
async def test_raw_json_without_an_error_key_is_ignored(middleware_module, middleware_source):
    upsert, emitted = await _run_raw_json_error(
        middleware_module, middleware_source, '{"choices": []}'
    )

    assert upsert.call_count == 0
    assert emitted == []


@pytest.mark.asyncio
async def test_non_json_line_is_ignored(middleware_module, middleware_source):
    upsert, emitted = await _run_raw_json_error(middleware_module, middleware_source, "not json")

    assert upsert.call_count == 0
    assert emitted == []


@pytest.mark.asyncio
async def test_raw_json_error_on_an_unsaved_chat_still_emits(
    middleware_module, middleware_source
):
    # Only the event is asserted: the `save_to_chat` guard around the upsert is a separate
    # 0.11.0 change, so the persistence half differs between refs by design.
    _, emitted = await _run_raw_json_error(
        middleware_module, middleware_source, '{"error": "nope"}', save_to_chat=False
    )

    assert emitted[0]["data"]["error"] == "nope"


# =============================================================================
# 0.11.2: the two streaming repairs in `streaming_chat_response_handler`
# =============================================================================


def _tag_output_handler(middleware_module, source):
    """The handler's own nested tag detector, lifted so the content branch can call it."""
    hits = [
        node
        for node in ast.walk(_function(source, "streaming_chat_response_handler"))
        if isinstance(node, ast.FunctionDef) and node.name == "tag_output_handler"
    ]
    assert len(hits) == 1, f"expected one nested `tag_output_handler`, found {len(hits)}"
    # the two per-stream scan tables it closes over in the handler
    namespace = _namespace(
        middleware_module, hits, tag_scan_positions={}, tag_boundary_positions={}
    )
    return _exec(hits, namespace)["tag_output_handler"]


def _stream_namespace(middleware_module, statements, output, chunk, content_parts, tag_handler):
    return _namespace(
        middleware_module,
        statements,
        delta=chunk,
        data=chunk,
        output=output,
        full_output=lambda: output,
        content_parts=content_parts,
        save_current_response_stream=AsyncMock(return_value=None),
        request=_request(),
        metadata={"chat_id": CHAT_ID, "message_id": MESSAGE_ID},
        user=None,
        DETECT_REASONING_TAGS=True,
        DETECT_CODE_INTERPRETER=False,
        tag_output_handler=tag_handler,
        reasoning_tags=middleware_module.DEFAULT_REASONING_TAGS,
        # base64 image rewriting is the I/O boundary of this branch
        ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION=False,
    )


async def _stream_deltas(middleware_module, middleware_source, chunks: list[dict], output=None):
    """Drive the shipped delta branch over a finite chunk list; returns output and events."""
    statements = _delta_block(middleware_source)
    tag_handler = _tag_output_handler(middleware_module, middleware_source)
    output = [] if output is None else output
    content_parts: list[str] = []
    events = []
    for chunk in chunks:
        namespace = _stream_namespace(
            middleware_module, statements, output, chunk, content_parts, tag_handler
        )
        await _exec_async_loop_body(statements, namespace)
        events.append(namespace["data"])
    return output, events


def _finish_tool_round(middleware_module, middleware_source, output, call_id, name="search_web"):
    """Run the shipped end-of-tool-round bookkeeping against one in-progress call."""
    output.append(
        {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
    )
    statements = _tool_round_block(middleware_source)
    namespace = _namespace(
        middleware_module,
        statements,
        output=output,
        results=[{"tool_call_id": call_id, "content": "tool said hello"}],
        response_tool_calls=[{"id": call_id, "function": {"name": name, "arguments": "{}"}}],
    )
    _exec(statements, namespace)
    return output


def _reasoning_items(output: list) -> list[dict]:
    return [item for item in output if item["type"] == "reasoning"]


def _message_text(output: list) -> str:
    return "".join(
        part.get("text", "")
        for item in output
        if item["type"] == "message"
        for part in item.get("content", [])
    )


def _reasoning_text(item: dict) -> str:
    return "".join(part.get("text", "") for part in item.get("content", []))


# --- Narrow: #29035, a reasoning delta carrying details must still be emitted -


REASONING_WITH_DETAILS = {
    "reasoning": "weighing the options",
    "reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "weighing the options"}],
}


@pytest.mark.asyncio
async def test_reasoning_delta_with_details_is_still_emitted(middleware_module, middleware_source):
    _, events = await _stream_deltas(
        middleware_module, middleware_source, [REASONING_WITH_DETAILS]
    )

    assert events[0] is not None, (
        "a reasoning chunk that also carried reasoning_details was dropped, so a reasoning "
        "model streamed nothing until generation finished (#29035)"
    )
    assert events[0]["type"] == "response.reasoning_text.delta"
    assert events[0]["delta"] == "weighing the options"


@pytest.mark.asyncio
async def test_every_reasoning_chunk_with_details_reaches_the_client(
    middleware_module, middleware_source
):
    chunks = [
        {
            "reasoning": word,
            "reasoning_details": [{"type": "reasoning.text", "index": 0, "text": word}],
        }
        for word in ("first ", "second ", "third")
    ]
    output, events = await _stream_deltas(middleware_module, middleware_source, chunks)

    assert [event and event["delta"] for event in events] == ["first ", "second ", "third"], (
        "the stream stalled: every reasoning delta accompanied by details was swallowed (#29035)"
    )
    assert _reasoning_text(_reasoning_items(output)[0]) == "first second third"


@pytest.mark.asyncio
async def test_reasoning_details_arriving_alone_emit_nothing(middleware_module, middleware_source):
    _, events = await _stream_deltas(
        middleware_module,
        middleware_source,
        [{"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "detail only"}]}],
    )

    assert events == [None], "a details-only chunk has no text delta to send and must stay dropped"


@pytest.mark.asyncio
async def test_reasoning_without_details_is_emitted(middleware_module, middleware_source):
    _, events = await _stream_deltas(
        middleware_module, middleware_source, [{"reasoning_content": "plain thought"}]
    )

    assert events[0]["type"] == "response.reasoning_text.delta"
    assert events[0]["delta"] == "plain thought"


# --- Narrow: #29040, thinking after a tool call belongs in Thoughts ----------


@pytest.mark.asyncio
async def test_post_tool_call_reasoning_opens_a_live_thoughts_block(
    middleware_module, middleware_source
):
    output = _finish_tool_round(middleware_module, middleware_source, [], "call_1")
    await _stream_deltas(
        middleware_module, middleware_source, [{"reasoning": "now I know"}], output
    )

    assert [item["type"] for item in output] == [
        "function_call",
        "function_call_output",
        "reasoning",
    ], (
        "a finished tool round left an empty message item behind, so the next round's thinking "
        "was routed relative to it instead of opening a Thoughts block of its own (#29040)"
    )
    reasoning = _reasoning_items(output)
    assert _reasoning_text(reasoning[0]) == "now I know"
    assert reasoning[0]["status"] == "in_progress", (
        "the empty message item pre-appended after a tool call made the next round's thinking "
        "land in a Thoughts block stamped completed on the spot instead of a live one (#29040)"
    )


@pytest.mark.asyncio
async def test_post_tool_call_reasoning_is_not_merged_into_the_earlier_thoughts(
    middleware_module, middleware_source
):
    output, _ = await _stream_deltas(
        middleware_module, middleware_source, [{"reasoning": "I should search"}]
    )
    _finish_tool_round(middleware_module, middleware_source, output, "call_1")
    await _stream_deltas(
        middleware_module, middleware_source, [{"reasoning": "the result says 42"}], output
    )

    reasoning = _reasoning_items(output)
    assert len(reasoning) == 2, (
        "reasoning for the step after a tool call was folded back into the previous, already "
        "completed thinking block (#29040)"
    )
    assert _reasoning_text(reasoning[0]) == "I should search"
    assert _reasoning_text(reasoning[1]) == "the result says 42"
    tool_result_index = next(
        index for index, item in enumerate(output) if item["type"] == "function_call_output"
    )
    assert output.index(reasoning[1]) > tool_result_index


# --- Broad: reasoning routes to a live Thoughts block wherever it arrives ----


@pytest.mark.asyncio
async def test_reasoning_routes_to_a_live_block_before_after_and_without_tools(
    middleware_module, middleware_source
):
    before: list = []
    await _stream_deltas(middleware_module, middleware_source, [{"reasoning": "why"}], before)
    _finish_tool_round(middleware_module, middleware_source, before, "call_1")

    after = _finish_tool_round(middleware_module, middleware_source, [], "call_1")
    await _stream_deltas(middleware_module, middleware_source, [{"reasoning": "why"}], after)

    without: list = []
    await _stream_deltas(middleware_module, middleware_source, [{"reasoning": "why"}], without)

    for label, output in (("before", before), ("after", after), ("none", without)):
        assert [item["type"] for item in output if item["type"] in ("reasoning", "message")] == [
            "reasoning"
        ], f"{label}: thinking did not land in a Thoughts block of its own"
        item = next(item for item in output if item["type"] == "reasoning")
        assert item["status"] == "in_progress", f"{label}: Thoughts block was already closed"
        assert _reasoning_text(item) == "why"


@pytest.mark.asyncio
async def test_consecutive_tool_calls_keep_routing_reasoning_correctly(
    middleware_module, middleware_source
):
    output: list = []
    for round_index, call_id in enumerate(("call_1", "call_2", "call_3")):
        await _stream_deltas(
            middleware_module, middleware_source, [{"reasoning": f"step {round_index}"}], output
        )
        _finish_tool_round(middleware_module, middleware_source, output, call_id)

    await _stream_deltas(
        middleware_module, middleware_source, [{"reasoning": "done"}, {"content": "42"}], output
    )

    reasoning = _reasoning_items(output)
    assert [_reasoning_text(item) for item in reasoning] == [
        "step 0",
        "step 1",
        "step 2",
        "done",
    ], "reasoning across consecutive tool calls was merged or misplaced (#29040)"
    assert _message_text(output) == "42"


# --- Nearby: streams without reasoning or without tools are unchanged -------


@pytest.mark.asyncio
async def test_plain_content_stream_assembles_one_message(middleware_module, middleware_source):
    output, events = await _stream_deltas(
        middleware_module,
        middleware_source,
        [{"content": "Hello"}, {"content": ", "}, {"content": "world"}],
    )

    assert [item["type"] for item in output] == ["message"]
    assert _message_text(output) == "Hello, world"
    assert [event["type"] for event in events] == ["response.output_text.delta"] * 3


@pytest.mark.asyncio
async def test_empty_stream_assembles_nothing(middleware_module, middleware_source):
    output, events = await _stream_deltas(middleware_module, middleware_source, [])

    assert output == []
    assert events == []


@pytest.mark.asyncio
async def test_reasoning_then_answer_without_tools_is_unchanged(
    middleware_module, middleware_source
):
    output, _ = await _stream_deltas(
        middleware_module,
        middleware_source,
        [{"reasoning": "hmm"}, {"content": "The answer is 42"}],
    )

    assert [item["type"] for item in output] == ["reasoning", "message"]
    assert output[0]["status"] == "completed"
    assert _message_text(output) == "The answer is 42"
