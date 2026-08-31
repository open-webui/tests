"""Regression tests for the 'request' filter stage (commit 2daa610cb, `utils/middleware.py`).

Filters could only see the payload at `inlet`, before retrieval, tool definitions, system
prompt assembly and message normalisation had run, so a plugin had no way to inspect or edit
what the model was actually asked. A new `request` stage now runs immediately before every
model call: at the end of `process_chat_payload`, and again in `drain_approved_tool_calls`
before the follow-up call that carries tool results. `inlet`, `stream` and `outlet` are
untouched.

Discriminates: passes on v0.11.3, fails on v0.11.1 (a filter defining `request` is never
invoked, on the first model call or on the post-tool-call one).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    return owui_module("open_webui.utils.filter")


@pytest.fixture(scope="session")
def user_model(owui_module):
    return owui_module("open_webui.models.users").UserModel


@pytest.fixture(scope="session")
def groups_model(owui_module):
    return owui_module("open_webui.models.groups").Groups


@pytest.fixture(scope="session")
def skills_model(owui_module):
    return owui_module("open_webui.models.skills").Skills


class FakeFunction:
    def __init__(self, function_id: str = "recorder"):
        self.id = function_id
        self.name = function_id
        self.type = "filter"
        self.is_active = True
        self.is_global = True


class RecordingFilter:
    """A filter module that logs which stages ran and stamps the payload it was handed."""

    def __init__(self, stages=("inlet", "request")):
        self.calls = []
        for stage in stages:
            setattr(self, stage, self._make_handler(stage))

    def _make_handler(self, stage):
        async def handler(body):
            self.calls.append((stage, [m.get("role") for m in body.get("messages", [])]))
            return {**body, f"{stage}_marker": len(self.calls)}

        return handler


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


SKILL_ACCESSORS = ("get_skills_by_user_id", "get_skills")


@pytest.fixture
def install_filter(middleware_module, filter_utils, monkeypatch):
    """Wire one filter module into the pipeline, patching only the plugin/db boundaries."""

    def _install(module):
        async def fake_get_filter_functions(request, model, enabled_filter_ids=None):
            return [FakeFunction()]

        async def fake_loader(request, function_id, function=None, load_from_db=True):
            return module, "filter", {}

        monkeypatch.setattr(middleware_module, "get_filter_functions", fake_get_filter_functions)
        monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
        return module

    return _install


@pytest.fixture
def no_user_lookups(groups_model, skills_model):
    """Groups and skills are pure db reads on the process_chat_payload path."""
    present = [name for name in SKILL_ACCESSORS if hasattr(skills_model, name)]
    assert present, f"none of {SKILL_ACCESSORS} on this checkout's Skills table"
    patches = [patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[]))]
    patches += [patch.object(skills_model, name, AsyncMock(return_value=[])) for name in present]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


async def run_process_chat_payload(middleware_module, user, messages=None):
    form_data = {
        "model": "m",
        "params": {"function_calling": "native"},
        "messages": messages or [{"role": "user", "content": "hi"}],
    }
    metadata = {
        "chat_id": "",
        "session_id": "session-1",
        "message_id": "assistant-1",
        "params": {"function_calling": "native"},
        "features": {},
    }
    form_data, _metadata, _events = await middleware_module.process_chat_payload(
        _request(), form_data, user, metadata, {"id": "m"}
    )
    return form_data


# =============================================================================
# Narrow -- the request stage runs before the first model call
# =============================================================================


@pytest.mark.asyncio
async def test_request_filter_runs_before_the_model_call(
    middleware_module, install_filter, no_user_lookups, user_model
):
    recorder = install_filter(RecordingFilter())

    form_data = await run_process_chat_payload(middleware_module, _user(user_model))

    assert [stage for stage, _ in recorder.calls] == ["inlet", "request"], (
        "the 'request' filter stage did not run before the model call (2daa610cb)"
    )
    assert form_data["request_marker"] == 2, (
        "the payload returned by the 'request' filter was discarded"
    )


@pytest.mark.asyncio
async def test_request_filter_can_change_the_payload(
    middleware_module, install_filter, no_user_lookups, user_model
):
    class Rewriter:
        async def request(self, body):
            return {**body, "messages": [*body["messages"], {"role": "user", "content": "extra"}]}

    install_filter(Rewriter())

    form_data = await run_process_chat_payload(middleware_module, _user(user_model))

    assert form_data["messages"][-1]["content"] == "extra", (
        "an edit made by the 'request' filter never reached the model payload (2daa610cb)"
    )


@pytest.mark.asyncio
async def test_request_filter_sees_the_assembled_payload(
    middleware_module, install_filter, no_user_lookups, user_model
):
    """inlet sees the raw payload, request sees the assembled one, tool definitions included."""
    seen = {}

    class Probe:
        async def inlet(self, body):
            seen["inlet"] = dict(body)
            return body

        async def request(self, body):
            seen["request"] = dict(body)
            return body

    install_filter(Probe())

    await run_process_chat_payload(
        middleware_module,
        _user(user_model),
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
    )

    assert "request" in seen, "the 'request' filter stage never ran (2daa610cb)"
    assert [m["role"] for m in seen["request"]["messages"]] == ["system", "user"]
    assert "tools" in seen["request"] and "tools" not in seen["inlet"], (
        "the 'request' filter was not handed the assembled payload (2daa610cb)"
    )
    assert "metadata" in seen["request"] and "metadata" not in seen["inlet"]


# =============================================================================
# Narrow -- the request stage runs again for the post-tool-call model call
# =============================================================================


@pytest.fixture
def drain_boundaries(middleware_module, monkeypatch):
    """Patch the chat store, tool execution and event plumbing drain_approved_tool_calls uses."""
    stored = {
        "output": [
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "c1",
                "name": "get_time",
                "arguments": "{}",
                "status": "queued",
                "approved": True,
            }
        ]
    }

    class FakeChats:
        @staticmethod
        async def get_message_by_id_and_message_id(chat_id, message_id):
            return stored

        @staticmethod
        async def upsert_message_to_chat_by_id_and_message_id(*args, **kwargs):
            return None

    async def fake_execute(request, form_data, user, metadata, event_caller, event_emitter, call):
        return {"tool_call_id": "c1", "content": "12:00"}

    async def fake_emitter_and_caller(metadata):
        return None, None

    async def fake_load_messages(chat_id, message_id):
        return []

    async def fake_oauth_token(request, user):
        return None

    monkeypatch.setattr(middleware_module, "Chats", FakeChats)
    monkeypatch.setattr(middleware_module, "execute_tool_call_for_output", fake_execute)
    monkeypatch.setattr(middleware_module, "get_event_emitter_and_caller", fake_emitter_and_caller)
    monkeypatch.setattr(middleware_module, "load_messages_from_db", fake_load_messages)
    monkeypatch.setattr(middleware_module, "get_system_oauth_token", fake_oauth_token)
    return stored


async def run_drain(middleware_module, user, form_data):
    return await middleware_module.drain_approved_tool_calls(
        _request(),
        form_data,
        user,
        {"id": "m"},
        {"chat_id": "chat-1", "assistant_message_id": "a1", "message_id": "a1"},
    )


@pytest.mark.asyncio
async def test_request_filter_runs_for_the_post_tool_call(
    middleware_module, install_filter, drain_boundaries, user_model
):
    recorder = install_filter(RecordingFilter(stages=("request",)))
    form_data = {"model": "m", "messages": [{"role": "user", "content": "what time is it"}]}

    paused = await run_drain(middleware_module, _user(user_model), form_data)

    assert paused is False
    assert [stage for stage, _ in recorder.calls] == ["request"], (
        "the 'request' filter stage did not run before the follow-up call that carries the tool "
        "results (2daa610cb)"
    )


@pytest.mark.asyncio
async def test_post_tool_call_request_filter_edit_reaches_the_caller(
    middleware_module, install_filter, drain_boundaries, user_model
):
    """drain_approved_tool_calls edits form_data in place, so a new dict must be copied back."""

    class Rewriter:
        async def request(self, body):
            return {**body, "temperature": 0.25}

    install_filter(Rewriter())
    form_data = {"model": "m", "messages": [{"role": "user", "content": "what time is it"}]}

    await run_drain(middleware_module, _user(user_model), form_data)

    assert form_data.get("temperature") == 0.25, (
        "the post-tool-call 'request' filter's payload never reached the caller's form_data "
        "(2daa610cb)"
    )


# =============================================================================
# Broad and nearby -- inlet, stream and outlet are unchanged
# =============================================================================


@pytest.mark.asyncio
async def test_inlet_only_filter_still_shapes_the_payload(
    middleware_module, install_filter, no_user_lookups, user_model
):
    recorder = install_filter(RecordingFilter(stages=("inlet",)))

    form_data = await run_process_chat_payload(middleware_module, _user(user_model))

    assert [stage for stage, _ in recorder.calls] == ["inlet"]
    assert form_data["inlet_marker"] == 1


@pytest.mark.asyncio
async def test_stream_and_outlet_handlers_are_not_run_by_the_payload_pass(
    middleware_module, install_filter, no_user_lookups, user_model
):
    recorder = install_filter(RecordingFilter(stages=("inlet", "stream", "outlet")))

    await run_process_chat_payload(middleware_module, _user(user_model))

    assert [stage for stage, _ in recorder.calls] == ["inlet"], (
        "a stage other than inlet ran while building the model payload"
    )


@pytest.mark.asyncio
async def test_inlet_failure_still_aborts_the_payload_pass(
    middleware_module, install_filter, no_user_lookups, user_model
):
    class RaisingInlet:
        async def inlet(self, body):
            raise RuntimeError("inlet exploded")

    install_filter(RaisingInlet())

    with pytest.raises(Exception, match="inlet exploded"):
        await run_process_chat_payload(middleware_module, _user(user_model))


@pytest.mark.asyncio
async def test_filter_with_no_handlers_leaves_the_payload_alone(
    middleware_module, install_filter, no_user_lookups, user_model
):
    install_filter(types.SimpleNamespace())

    form_data = await run_process_chat_payload(middleware_module, _user(user_model))

    assert form_data["messages"][-1]["content"] == "hi"


@pytest.mark.asyncio
async def test_drain_without_filters_still_drains(
    middleware_module, install_filter, drain_boundaries, user_model
):
    install_filter(types.SimpleNamespace())
    form_data = {"model": "m", "messages": [{"role": "user", "content": "what time is it"}]}

    paused = await run_drain(middleware_module, _user(user_model), form_data)

    assert paused is False
    assert drain_boundaries["output"][0]["status"] == "completed"
    assert any(item["type"] == "function_call_output" for item in drain_boundaries["output"])
