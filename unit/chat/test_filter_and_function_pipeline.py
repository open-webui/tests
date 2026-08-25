"""Regression tests for the 0.11.1 filter/function plumbing fixes.

Five independent regressions in the filter and pipe-function path:

* Disabled functions (commit a3a81fee03, `routers/functions.py`). The two
  user-valves endpoints never looked at `function.is_active`, so a switched-off
  function still had its module loaded (arbitrary plugin code executed) and
  per-user valves written that could never take effect. The spec endpoint now
  returns None and the update endpoint raises 400 'Function is not active'.
* System prompt duplication (PR #28739, commit ebd4d9c6c, `functions.py`).
  `generate_function_chat_completion` re-ran `apply_system_prompt_to_body` on
  every tool-call continuation, and `add_or_update_system_message` prepends,
  so the system message grew one extra copy per round. The fix honours
  `request.state.bypass_system_prompt`, set server-side by `utils/chat.py`.
* Dropped reply text (PR #28840, commit ac091273b, `utils/middleware.py`).
  `delta.content` and the reasoning delta are raw JSON, so a provider (or a
  stream filter) can make them any type. The coercion used to happen only at
  the `content_parts.append` call, after the branches that concatenate the
  value onto the message item, so a non-string chunk blew up mid-stream and
  the rest of the reply vanished. Both values are now coerced on read.
* Stream filters on direct API calls (commit 684111715f, `utils/middleware.py`).
  The no-event-emitter fallback handed the raw SSE line (`data: {...}`, often
  bytes) to `process_filter_functions(filter_type='stream')` verbatim, so any
  filter indexing the event as a dict raised and ended the reply partway. The
  payload is now decoded, `[DONE]` skipped, and the result re-wrapped.
* Silent filter failures (commit 35fbde0a3f, `utils/filter.py`). Every filter
  exception was `log.debug`-ed and re-raised, invisible at the default log
  level. Outlet and stream failures now use `log.exception`, valve
  construction moved inside the try, and user-valve failures name the filter.

Discriminates: passes on v0.11.1, fails on v0.11.0 (no is_active guard, the
system prompt gets re-applied, a non-string delta truncates the reply, the
stream filter is fed a raw SSE line, and outlet/stream failures log nothing
above DEBUG).
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

pytestmark = pytest.mark.regression


# -----------------------------------------------------------------------------
# Module fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    """`open_webui.utils.filter` (process_filter_function)."""
    return owui_module("open_webui.utils.filter")


@pytest.fixture(scope="session")
def functions_router(owui_module):
    """`open_webui.routers.functions` (user-valves endpoints)."""
    return owui_module("open_webui.routers.functions")


@pytest.fixture(scope="session")
def functions_module(owui_module):
    """`open_webui.functions` (generate_function_chat_completion)."""
    return owui_module("open_webui.functions")


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    """`open_webui.utils.middleware` (process_chat_response)."""
    return owui_module("open_webui.utils.middleware")


# -----------------------------------------------------------------------------
# Shared doubles
# -----------------------------------------------------------------------------


class FakeFunction:
    """The `Functions` row a filter/function endpoint works from."""

    def __init__(self, function_id: str, is_active: bool = True):
        self.id = function_id
        self.name = function_id
        self.type = "filter"
        self.is_active = is_active
        self.is_global = False


class FakeUser:
    def __init__(self, user_id: str = "user-1", role: str = "user"):
        self.id = user_id
        self.role = role


def make_request(**state) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(**state),
        cookies={},
        app=SimpleNamespace(state=SimpleNamespace(MODELS={}, redis=None)),
    )


def filter_logs(caplog, min_level: int = logging.ERROR) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == "open_webui.utils.filter" and r.levelno >= min_level
    ]


# =============================================================================
# 197 -- silent filter failures (utils/filter.py)
# =============================================================================


class RaisingOutletModule:
    async def outlet(self, body):
        raise RuntimeError("outlet exploded")


class RaisingStreamModule:
    def stream(self, event):
        raise RuntimeError("stream exploded")


class RaisingInletModule:
    async def inlet(self, body):
        raise RuntimeError("inlet exploded")


class BadValvesModule:
    class Valves(BaseModel):
        priority: int = 0

        def __init__(self, **kwargs):
            raise RuntimeError("valves exploded")

    valves = None

    async def outlet(self, body):
        return body


class BadUserValvesModule:
    class UserValves(BaseModel):
        def __init__(self, **kwargs):
            raise RuntimeError("user valves exploded")

    async def outlet(self, body, __user__):
        return {**body, "ran": True}


@pytest.fixture
def patch_module_loader(filter_utils, monkeypatch):
    """Swap the plugin loader, the only I/O boundary process_filter_function has."""

    def _install(module):
        async def fake_loader(request, function_id, function=None, load_from_db=True):
            return module, "filter", {}

        monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
        return module

    return _install


async def run_filter(
    filter_utils, filter_type, form_data, filter_id="acme_filter", filter_context=None, valves=None
):
    return await filter_utils.process_filter_function(
        request=make_request(),
        function=FakeFunction(filter_id),
        filter_type=filter_type,
        form_data=form_data,
        extra_params={"__user__": {"id": "user-1"}},
        filter_context=filter_context,
        valves_by_id=valves,
        filter_ids=[filter_id],
    )


@pytest.mark.asyncio
async def test_outlet_failure_is_logged_with_traceback(filter_utils, patch_module_loader, caplog):
    patch_module_loader(RaisingOutletModule())

    with pytest.raises(RuntimeError):
        await run_filter(filter_utils, "outlet", {"messages": []})

    records = filter_logs(caplog)
    assert records, "an outlet filter that raised produced no log record above DEBUG"
    assert "acme_filter" in records[0].getMessage()
    assert records[0].exc_info is not None


@pytest.mark.asyncio
async def test_stream_failure_is_logged_with_traceback(filter_utils, patch_module_loader, caplog):
    patch_module_loader(RaisingStreamModule())

    with pytest.raises(RuntimeError):
        await run_filter(filter_utils, "stream", {"choices": []})

    records = filter_logs(caplog)
    assert records, "a stream filter that raised produced no log record above DEBUG"
    assert "acme_filter" in records[0].getMessage()
    assert records[0].exc_info is not None


@pytest.mark.asyncio
async def test_valve_construction_failure_is_logged(filter_utils, patch_module_loader, caplog):
    """Valve building moved inside the try, so a bad Valves class is no longer silent."""
    patch_module_loader(BadValvesModule())

    with pytest.raises(RuntimeError):
        await run_filter(
            filter_utils, "outlet", {"messages": []}, valves={"acme_filter": {"priority": 1}}
        )

    records = filter_logs(caplog)
    assert records, "a filter whose Valves raised produced no log record above DEBUG"
    assert "acme_filter" in records[0].getMessage()


@pytest.mark.asyncio
async def test_user_valve_failure_names_the_filter(
    filter_utils, patch_module_loader, monkeypatch, caplog
):
    patch_module_loader(BadUserValvesModule())

    class FakeFunctions:
        @staticmethod
        async def get_user_valves_by_id_and_user_id(filter_id, user_id):
            return {}

    monkeypatch.setattr(filter_utils, "Functions", FakeFunctions)

    form_data, _, _ = await run_filter(filter_utils, "outlet", {"messages": []})

    # A user-valve failure is not fatal: the handler still runs.
    assert form_data["ran"] is True

    records = filter_logs(caplog)
    assert records, "a failing UserValves produced no log record above DEBUG"
    assert "acme_filter" in records[0].getMessage(), "the log line does not say which filter failed"


@pytest.mark.asyncio
async def test_inlet_failure_stays_quiet(filter_utils, patch_module_loader, caplog):
    """Inlet failures are already surfaced to the caller, so they stay at DEBUG."""
    patch_module_loader(RaisingInletModule())

    with pytest.raises(RuntimeError):
        await run_filter(filter_utils, "inlet", {"messages": []})

    assert not filter_logs(caplog)


@pytest.mark.asyncio
async def test_successful_filter_logs_nothing(filter_utils, patch_module_loader, caplog):
    class GoodModule:
        async def outlet(self, body):
            return {**body, "touched": True}

    patch_module_loader(GoodModule())

    form_data, _, _ = await run_filter(filter_utils, "outlet", {"messages": []})

    assert form_data["touched"] is True
    assert not filter_logs(caplog, logging.DEBUG)


@pytest.mark.asyncio
async def test_missing_handler_is_a_noop(filter_utils, patch_module_loader):
    class NoOutletModule:
        async def inlet(self, body):
            return body

    patch_module_loader(NoOutletModule())

    form_data, valves, skip_files = await run_filter(filter_utils, "outlet", {"messages": []})

    assert form_data == {"messages": []}
    assert valves is None
    assert skip_files is None


# =============================================================================
# 131 -- user valves on a switched-off function (routers/functions.py)
# =============================================================================


class UserValvesSpecModule:
    class UserValves(BaseModel):
        greeting: str = "hi"


@pytest.fixture
def valves_endpoint_env(functions_router, monkeypatch):
    """Wire both user-valves endpoints to in-memory doubles.

    Records whether the plugin module was loaded and whether valves were
    written, which is what the is_active guard is meant to prevent.
    """
    state = SimpleNamespace(function=None, module_loads=[], stored=[])

    class FakeFunctions:
        @staticmethod
        async def get_function_by_id(function_id, db=None):
            return state.function

        @staticmethod
        async def update_user_valves_by_id_and_user_id(function_id, user_id, valves, db=None):
            state.stored.append((function_id, user_id, valves))
            return valves

    async def fake_loader(request, function_id, **kwargs):
        state.module_loads.append(function_id)
        return UserValvesSpecModule(), "filter", {}

    async def fake_publish_event(*args, **kwargs):
        return None

    monkeypatch.setattr(functions_router, "Functions", FakeFunctions)
    monkeypatch.setattr(functions_router, "get_function_module_from_cache", fake_loader)
    monkeypatch.setattr(functions_router, "publish_event", fake_publish_event)
    return state


@pytest.mark.asyncio
async def test_user_valves_spec_hidden_for_inactive_function(functions_router, valves_endpoint_env):
    valves_endpoint_env.function = FakeFunction("acme_filter", is_active=False)

    spec = await functions_router.get_function_user_valves_spec_by_id(
        request=make_request(), id="acme_filter", user=FakeUser(), db=None
    )

    assert spec is None
    assert valves_endpoint_env.module_loads == [], "a disabled function's module was still loaded"


@pytest.mark.asyncio
async def test_user_valves_update_rejected_for_inactive_function(
    functions_router, valves_endpoint_env
):
    valves_endpoint_env.function = FakeFunction("acme_filter", is_active=False)

    with pytest.raises(HTTPException) as excinfo:
        await functions_router.update_function_user_valves_by_id(
            request=make_request(),
            id="acme_filter",
            form_data={"greeting": "hello"},
            user=FakeUser(),
            db=None,
        )

    assert excinfo.value.status_code == 400
    assert "not active" in str(excinfo.value.detail).lower()
    assert valves_endpoint_env.stored == [], "valves were written for a disabled function"
    assert valves_endpoint_env.module_loads == [], "a disabled function's module was still loaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["spec", "update"])
async def test_no_user_valves_endpoint_loads_an_inactive_module(
    functions_router, valves_endpoint_env, endpoint
):
    """The invariant both endpoints gained: disabled means the plugin never runs."""
    valves_endpoint_env.function = FakeFunction("acme_filter", is_active=False)

    try:
        if endpoint == "spec":
            await functions_router.get_function_user_valves_spec_by_id(
                request=make_request(), id="acme_filter", user=FakeUser(), db=None
            )
        else:
            await functions_router.update_function_user_valves_by_id(
                request=make_request(),
                id="acme_filter",
                form_data={"greeting": "hello"},
                user=FakeUser(),
                db=None,
            )
    except HTTPException:
        pass

    assert valves_endpoint_env.module_loads == []
    assert valves_endpoint_env.stored == []


@pytest.mark.asyncio
async def test_user_valves_spec_returned_for_active_function(functions_router, valves_endpoint_env):
    valves_endpoint_env.function = FakeFunction("acme_filter", is_active=True)

    spec = await functions_router.get_function_user_valves_spec_by_id(
        request=make_request(), id="acme_filter", user=FakeUser(), db=None
    )

    assert "greeting" in spec["properties"]
    assert valves_endpoint_env.module_loads == ["acme_filter"]


@pytest.mark.asyncio
async def test_user_valves_update_stored_for_active_function(functions_router, valves_endpoint_env):
    valves_endpoint_env.function = FakeFunction("acme_filter", is_active=True)

    result = await functions_router.update_function_user_valves_by_id(
        request=make_request(),
        id="acme_filter",
        form_data={"greeting": "hello"},
        user=FakeUser(),
        db=None,
    )

    assert result == {"greeting": "hello"}
    assert valves_endpoint_env.stored == [("acme_filter", "user-1", {"greeting": "hello"})]


@pytest.mark.asyncio
async def test_user_valves_spec_unknown_function_is_401(functions_router, valves_endpoint_env):
    valves_endpoint_env.function = None

    with pytest.raises(HTTPException) as excinfo:
        await functions_router.get_function_user_valves_spec_by_id(
            request=make_request(), id="nope", user=FakeUser(), db=None
        )

    assert excinfo.value.status_code == 401


# =============================================================================
# 133 -- system prompt re-applied on every tool-call continuation (functions.py)
# =============================================================================


class FakeParams:
    def __init__(self, values: dict):
        self._values = values

    def model_dump(self):
        return dict(self._values)


class FakeModelInfo:
    def __init__(self, params: dict):
        self.base_model_id = None
        self.params = FakeParams(params)


@pytest.fixture(scope="session")
def pipe_user(owui_module):
    """A real UserModel: generate_function_chat_completion re-wraps the caller's user."""
    users = owui_module("open_webui.models.users")
    return users.UserModel(
        id="user-1",
        email="user@example.com",
        name="User",
        role="user",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


@pytest.fixture
def pipe_env(functions_module, monkeypatch):
    """Drive generate_function_chat_completion against an in-memory pipe."""
    state = SimpleNamespace(model_info=FakeModelInfo({"system": "SYS"}), bodies=[])

    class FakeModels:
        @staticmethod
        async def get_model_by_id(model_id):
            return state.model_info

    class PipeModule:
        @staticmethod
        def pipe(body):
            state.bodies.append(json.loads(json.dumps(body)))
            return "ok"

    async def fake_get_function_module_by_id(request, pipe_id):
        return PipeModule

    async def fake_check_model_access(*args, **kwargs):
        return None

    monkeypatch.setattr(functions_module, "Models", FakeModels)
    monkeypatch.setattr(
        functions_module, "get_function_module_by_id", fake_get_function_module_by_id
    )
    monkeypatch.setattr(functions_module, "check_model_access", fake_check_model_access)
    return state


async def run_pipe(functions_module, pipe_user, messages, bypass=None):
    state_kwargs = {} if bypass is None else {"bypass_system_prompt": bypass}
    request = make_request(**state_kwargs)
    request.app.state.oauth_manager = None
    await functions_module.generate_function_chat_completion(
        request,
        {"model": "acme_pipe", "messages": messages, "stream": False},
        pipe_user,
    )


@pytest.mark.asyncio
async def test_tool_call_continuation_does_not_restack_system_prompt(
    functions_module, pipe_env, pipe_user
):
    """The continuation payload already carries the system message from round one."""
    await run_pipe(
        functions_module,
        pipe_user,
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
        bypass=True,
    )

    system_message = pipe_env.bodies[0]["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == "SYS"


@pytest.mark.asyncio
async def test_bypass_survives_repeated_continuations(functions_module, pipe_env, pipe_user):
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    for _ in range(3):
        await run_pipe(functions_module, pipe_user, messages, bypass=True)

    assert [body["messages"][0]["content"] for body in pipe_env.bodies] == ["SYS", "SYS", "SYS"]


@pytest.mark.asyncio
async def test_first_turn_still_applies_the_system_prompt(functions_module, pipe_env, pipe_user):
    await run_pipe(functions_module, pipe_user, [{"role": "user", "content": "hi"}], bypass=None)

    assert pipe_env.bodies[0]["messages"][0] == {"role": "system", "content": "SYS"}


@pytest.mark.asyncio
async def test_model_without_params_leaves_messages_alone(functions_module, pipe_env, pipe_user):
    pipe_env.model_info = FakeModelInfo({})

    await run_pipe(functions_module, pipe_user, [{"role": "user", "content": "hi"}], bypass=None)

    assert pipe_env.bodies[0]["messages"] == [{"role": "user", "content": "hi"}]


# =============================================================================
# Streaming harness (shared by 158 and 196)
# =============================================================================


def sse_response(payloads) -> StreamingResponse:
    async def body():
        for payload in payloads:
            if isinstance(payload, (bytes, str)):
                line = payload
            else:
                line = f"data: {json.dumps(payload)}"
            yield line.encode("utf-8") if isinstance(line, str) else line

    return StreamingResponse(body(), media_type="text/event-stream")


def content_chunk(value) -> dict:
    return {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "model": "acme",
        "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
    }


def make_stream_ctx(event_emitter=None, chat_id="") -> dict:
    request = make_request()
    return {
        "request": request,
        "form_data": {
            "model": "acme",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        "user": FakeUser(),
        "model": {"id": "acme", "name": "acme"},
        "metadata": {"chat_id": chat_id, "message_id": "message-1"},
        "tasks": {},
        "events": [],
        "event_emitter": event_emitter,
        "event_caller": None,
    }


@pytest.fixture
def stream_env(middleware_module, filter_utils, monkeypatch):
    """Neutralise the I/O around the streaming handler; leave the handler real."""
    state = SimpleNamespace(filter_functions=[], module=None)

    async def fake_get_filter_functions(request, model, filter_ids=None):
        return list(state.filter_functions)

    async def fake_get_system_oauth_token(request, user):
        return None

    async def fake_loader(request, function_id, function=None, load_from_db=True):
        return state.module, "filter", {}

    monkeypatch.setattr(middleware_module, "get_filter_functions", fake_get_filter_functions)
    monkeypatch.setattr(middleware_module, "get_system_oauth_token", fake_get_system_oauth_token)
    monkeypatch.setattr(middleware_module, "get_sorted_filters", lambda *a, **k: [])
    monkeypatch.setattr(middleware_module, "ENABLE_API_OUTLET_FILTERS", False)
    monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
    return state


# =============================================================================
# 196 -- stream filters on direct API calls (utils/middleware.py fallback path)
# =============================================================================


class RecordingStreamFilter:
    """A filter whose stream hook treats the event as a decoded dict."""

    def __init__(self):
        self.seen = []

    def stream(self, event):
        self.seen.append(event)
        if isinstance(event, dict):
            event["choices"][0]["delta"]["content"] = "FILTERED"
        return event


async def drain(response) -> list[str]:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return chunks


@pytest.mark.asyncio
async def test_direct_api_stream_filter_receives_decoded_events(middleware_module, stream_env):
    plugin = RecordingStreamFilter()
    stream_env.module = plugin
    stream_env.filter_functions = [FakeFunction("acme_filter")]

    response = sse_response([content_chunk("hello"), content_chunk(" there"), "data: [DONE]"])
    wrapped = await middleware_module.process_chat_response(response, make_stream_ctx())
    chunks = await drain(wrapped)

    assert plugin.seen, "the stream filter was never called"
    assert all(isinstance(event, dict) for event in plugin.seen), (
        "the stream filter was handed a raw SSE line instead of a decoded event"
    )
    assert len(plugin.seen) == 2, "the [DONE] sentinel was handed to the filter"
    assert sum("FILTERED" in chunk for chunk in chunks) == 2
    assert any("[DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_direct_api_stream_filter_edit_reaches_the_client(middleware_module, stream_env):
    """A dict-indexing stream hook must not truncate the reply."""

    class RewritingFilter:
        def stream(self, event):
            event["choices"][0]["delta"]["content"] = event["choices"][0]["delta"][
                "content"
            ].upper()
            return event

    stream_env.module = RewritingFilter()
    stream_env.filter_functions = [FakeFunction("acme_filter")]

    response = sse_response(
        [content_chunk("a"), content_chunk("b"), content_chunk("c"), "data: [DONE]"]
    )
    wrapped = await middleware_module.process_chat_response(response, make_stream_ctx())
    chunks = await drain(wrapped)

    texts = [
        json.loads(c.removeprefix("data:").strip())["choices"][0]["delta"]["content"]
        for c in chunks[:3]
    ]
    assert texts == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_direct_api_stream_without_filters_passes_through(middleware_module, stream_env):
    stream_env.filter_functions = []

    response = sse_response([content_chunk("hello"), "data: [DONE]"])
    wrapped = await middleware_module.process_chat_response(response, make_stream_ctx())
    chunks = await drain(wrapped)

    assert (
        json.loads(chunks[0].removeprefix("data:").strip())["choices"][0]["delta"]["content"]
        == "hello"
    )
    assert chunks[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_direct_api_stream_keeps_non_sse_lines(middleware_module, stream_env):
    """Keep-alive comments and blank lines are not events; they pass through."""
    stream_env.module = RecordingStreamFilter()
    stream_env.filter_functions = [FakeFunction("acme_filter")]

    response = sse_response([": ping", content_chunk("hello"), "data: [DONE]"])
    wrapped = await middleware_module.process_chat_response(response, make_stream_ctx())
    chunks = await drain(wrapped)

    assert chunks[0].strip() == ": ping"


# =============================================================================
# 158 -- non-string delta content dropped from the reply (utils/middleware.py)
# =============================================================================


async def collect_completion_output(middleware_module, payloads) -> dict:
    events = []

    async def event_emitter(event):
        events.append(event)

    ctx = make_stream_ctx(event_emitter=event_emitter)
    await middleware_module.process_chat_response(sse_response(payloads), ctx)

    completion = [
        e for e in events if e.get("type") == "chat:completion" and e.get("data", {}).get("done")
    ]
    assert completion, "the stream never completed"
    return completion[-1]["data"]


def output_text(data: dict) -> str:
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
    return "".join(parts)


@pytest.mark.asyncio
async def test_non_string_delta_content_is_kept(middleware_module, stream_env):
    """A provider (or a stream filter) can put a non-string in delta.content."""
    data = await collect_completion_output(
        middleware_module,
        [content_chunk("Hello "), content_chunk(123), content_chunk(" World"), "data: [DONE]"],
    )

    assert output_text(data) == "Hello 123 World"


@pytest.mark.asyncio
async def test_structured_delta_content_is_kept(middleware_module, stream_env):
    """A dict-valued chunk is stringified rather than silently dropped."""
    structured = {"text": "x"}
    data = await collect_completion_output(
        middleware_module,
        [content_chunk("head "), content_chunk(structured), content_chunk(" tail"), "data: [DONE]"],
    )

    assert output_text(data) == f"head {structured} tail"


def reasoning_text(data: dict) -> str:
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "reasoning":
            for part in item.get("content", []):
                parts.append(part.get("text", ""))
    return "".join(str(part) for part in parts)


@pytest.mark.asyncio
async def test_non_string_reasoning_delta_is_kept(middleware_module, stream_env):
    """The first non-string reasoning delta poisons the item every later delta appends to."""

    def reasoning_chunk(value):
        return {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "model": "acme",
            "choices": [{"index": 0, "delta": {"reasoning_content": value}, "finish_reason": None}],
        }

    data = await collect_completion_output(
        middleware_module,
        [reasoning_chunk(42), reasoning_chunk(" more"), content_chunk("done"), "data: [DONE]"],
    )

    assert reasoning_text(data) == "42 more"
    assert output_text(data) == "done"


@pytest.mark.asyncio
async def test_plain_string_stream_is_unchanged(middleware_module, stream_env):
    data = await collect_completion_output(
        middleware_module,
        [content_chunk("Hello "), content_chunk("World"), "data: [DONE]"],
    )

    assert output_text(data) == "Hello World"


@pytest.mark.asyncio
async def test_empty_and_none_deltas_are_ignored(middleware_module, stream_env):
    data = await collect_completion_output(
        middleware_module,
        [content_chunk(None), content_chunk(""), content_chunk("only"), "data: [DONE]"],
    )

    assert output_text(data) == "only"
