"""Five 0.11.1 fixes in the filter and function plumbing.

* Disabled functions (commit a3a81fee03): the user-valves endpoints ignored `is_active`, so a
  switched-off function still loaded its code and accepted valves that could never apply. The
  spec now answers null and the update answers 400 "Function is not active".
* System prompt duplication (PR #28739, commit ebd4d9c6c): a pipe model's system prompt was
  applied again on every tool-call continuation, and since it is prepended into the existing
  system message, each round added one more copy.
* Dropped reply text (PR #28840, commit ac091273b): a non-string `delta.content` (or reasoning
  delta) was concatenated before it was coerced, so it broke the stream and the rest of the reply
  vanished. Both are coerced on read now.
* Stream filters on direct API calls (commit 684111715f): without a socket session the stream
  filter was handed the raw `data: {...}` line, so a filter reading the event as a dict raised
  and ended the reply. The event is decoded first, `[DONE]` is skipped and other lines pass.
* Silent filter failures (commit 35fbde0a3f): every filter failure was logged at DEBUG. Outlet
  and stream failures are now logged with their traceback and the filter id, a Valves class that
  fails to build is caught by the same handler, and a user-valves failure names the filter.

Twin of unit/chat/test_filter_and_function_pipeline.py.

Discriminates: reverting each fix in its own copy fails that fix's tests and no others: the
disabled-function spec and update, the pipe's repeated system prompt, the number, object and
reasoning deltas, the direct stream filter (the keep-alive line too), and the outlet, stream,
valves and user-valves log lines. The nearby tests (active function, unknown id, a pipe without
params, plain and empty deltas, no filter, a quiet inlet, a working filter) pass on all five.
"""

from __future__ import annotations

import json
import textwrap
import time

import httpx
import pytest

from harness import raw_provider
from harness import upstream as reply
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.plugins import installed_function
from harness.raw_provider import RAW_MODEL_ID, chunk, sse
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def source(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def wait_for_log(instance: LaunchedInstance, offset: int, needle: str) -> str:
    """The log written since `offset`, once it contains `needle` (outlets run after the reply)."""
    deadline = time.monotonic() + 15
    logged = instance.log_since(offset)
    while needle not in logged and time.monotonic() < deadline:
        time.sleep(0.2)
        logged = instance.log_since(offset)
    return logged


def direct_stream(client: httpx.Client, model: str = MOCK_MODEL_ID) -> list[str]:
    """A streamed chat without a socket session; returns the raw SSE lines."""
    request = {"model": model, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/api/chat/completions", json=request)
    assert response.status_code == 200, response.text
    return [line for line in response.text.splitlines() if line]


def streamed_text(lines: list[str]) -> str:
    events = [
        json.loads(line.removeprefix("data:"))
        for line in lines
        if line.startswith("data:") and line != "data: [DONE]"
    ]
    return "".join(
        choice["delta"].get("content") or ""
        for event in events
        for choice in event.get("choices", [])
    )


# --- disabled functions (a3a81fee03) -----------------------------------------------------------

USER_VALVES_FILTER = source(
    """
    from pydantic import BaseModel

    class Filter:
        class UserValves(BaseModel):
            greeting: str = "hi"

        async def inlet(self, body):
            return body
    """
)


def test_a_disabled_function_offers_no_user_valves(admin, user):
    with installed_function(admin, USER_VALVES_FILTER, active=False) as function_id:
        with user.client() as client:
            base = f"/api/v1/functions/id/{function_id}/valves/user"
            spec = client.get(f"{base}/spec")
            update = client.post(f"{base}/update", json={"greeting": "hello"})
            stored = client.get(base)

    assert (spec.status_code, spec.json()) == (200, None)
    assert update.status_code == 400
    assert "not active" in update.json()["detail"].lower()
    assert not stored.json(), "valves were stored for a disabled function"


def test_an_active_function_offers_and_stores_user_valves(admin, user):
    with installed_function(admin, USER_VALVES_FILTER) as function_id:
        with user.client() as client:
            base = f"/api/v1/functions/id/{function_id}/valves/user"
            spec = client.get(f"{base}/spec")
            update = client.post(f"{base}/update", json={"greeting": "hello"})
            stored = client.get(base)

    assert "greeting" in spec.json()["properties"]
    assert update.json() == {"greeting": "hello"}
    assert stored.json() == {"greeting": "hello"}


def test_user_valves_of_an_unknown_function_are_not_found(user):
    with user.client() as client:
        spec = client.get("/api/v1/functions/id/no_such_function/valves/user/spec")
    assert spec.status_code == 401


# --- pipe system prompt on tool-call continuations (ebd4d9c6c) ---------------------------------

TOOL_CALLING_PIPE = source(
    """
    import json

    class Pipe:
        def pipe(self, body):
            messages = body["messages"]
            rounds = sum(1 for message in messages if message["role"] == "tool")
            if rounds < 2:
                call = {"index": 0, "id": f"call_{rounds}", "type": "function"}
                call["function"] = {"name": "get_current_timestamp", "arguments": "{}"}
                yield {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
                yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                return
            yield json.dumps([m["content"] for m in messages if m["role"] == "system"])
    """
)


@pytest.mark.parametrize(
    "system_prompt,expected",
    [
        pytest.param("Answer as a pirate.", ["Answer as a pirate."], id="with-system-prompt"),
        pytest.param(None, [], id="without-model-params"),
    ],
)
def test_a_pipe_keeps_one_system_prompt_across_tool_calls(admin, system_prompt, expected):
    with installed_function(admin, TOOL_CALLING_PIPE) as pipe_id, admin.client() as client:
        client.get("/api/models").raise_for_status()
        if system_prompt:
            model = {
                "id": pipe_id,
                "name": pipe_id,
                "meta": {},
                "params": {"system": system_prompt},
            }
            client.post("/api/v1/models/create", json=model).raise_for_status()
        try:
            _, message = ask(client, "what time is it?", model=pipe_id)
        finally:
            if system_prompt:
                client.post("/api/v1/models/model/delete", json={"id": pipe_id})

    calls = [item for item in message["output"] if item["type"] == "function_call"]
    assert len(calls) == 2, "the pipe was not called back after its tool calls"
    assert json.loads(message["content"]) == expected


# --- non-string deltas (ac091273b) -------------------------------------------------------------


@pytest.mark.parametrize(
    "pieces,stored",
    [
        pytest.param(["Hello ", 123, " World"], "Hello 123 World", id="number"),
        pytest.param(["head ", {"text": "x"}, " tail"], "head {'text': 'x'} tail", id="object"),
        pytest.param(["Hello ", "World"], "Hello World", id="strings"),
    ],
)
def test_a_non_string_content_delta_keeps_the_whole_reply(user, upstream, pieces, stored):
    upstream.queue(reply.text(pieces))
    with user.client() as client:
        _, message = ask(client, "say hello")
    assert message["content"] == stored


@pytest.fixture
def raw(admin, preserve, listener) -> raw_provider.RawProvider:
    preserve(OPENAI_CONFIG)
    return raw_provider.connect(admin, listener)


def stored_reasoning(message: dict) -> str:
    return "".join(
        part["text"]
        for item in message["output"]
        if item["type"] == "reasoning"
        for part in item.get("content") or []
    )


def test_a_non_string_reasoning_delta_keeps_the_reasoning_and_the_reply(admin, raw):
    raw.stream(
        sse(
            chunk({"role": "assistant", "reasoning_content": 42}),
            chunk({"reasoning_content": " more"}),
            chunk({"content": "done"}),
            chunk({}, "stop"),
        )
    )
    with admin.client() as client:
        _, message = ask(client, "think first", model=RAW_MODEL_ID)
    assert stored_reasoning(message) == "42 more"
    assert message["content"] == "done"


def test_empty_and_null_deltas_are_skipped(admin, raw):
    deltas = [{"role": "assistant", "content": None}, {"content": ""}, {"content": "only"}]
    raw.stream(sse(*[chunk(delta) for delta in deltas], chunk({}, "stop")))
    with admin.client() as client:
        _, message = ask(client, "say it", model=RAW_MODEL_ID)
    assert message["content"] == "only"


# --- stream filters on direct API calls (684111715f) -------------------------------------------

UPPERCASE_STREAM_FILTER = source(
    """
    class Filter:
        def stream(self, event):
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str):
                    choice["delta"]["content"] = content.upper()
            return event
    """
)


def test_a_stream_filter_edits_every_chunk_of_a_direct_call(admin, user, upstream):
    upstream.queue(reply.text(["hello ", "there ", "friend"]))
    with installed_function(admin, UPPERCASE_STREAM_FILTER, is_global=True):
        with user.client() as client:
            lines = direct_stream(client)

    assert streamed_text(lines) == "HELLO THERE FRIEND"
    assert lines[-1] == "data: [DONE]"


def test_a_direct_call_without_filters_passes_through(user, upstream):
    upstream.queue(reply.text(["hello ", "there"]))
    with user.client() as client:
        lines = direct_stream(client)

    assert streamed_text(lines) == "hello there"
    assert lines[-1] == "data: [DONE]"


def test_a_stream_filter_leaves_keep_alive_lines_alone(admin, raw):
    raw.stream(sse(": keep-alive", chunk({"content": "hello"}), chunk({}, "stop")))
    with installed_function(admin, UPPERCASE_STREAM_FILTER, is_global=True):
        with admin.client() as client:
            lines = direct_stream(client, RAW_MODEL_ID)

    assert ": keep-alive" in lines
    assert streamed_text(lines) == "HELLO"


# --- filter failures in the log (35fbde0a3f) ---------------------------------------------------

RAISING_OUTLET = source(
    """
    class Filter:
        async def outlet(self, body):
            raise RuntimeError("outlet exploded")
    """
)

RAISING_STREAM = source(
    """
    class Filter:
        def stream(self, event):
            raise RuntimeError("stream exploded")
    """
)

UNBUILDABLE_VALVES = source(
    """
    from pydantic import BaseModel

    class Filter:
        class Valves(BaseModel):
            def __init__(self, **values):
                raise RuntimeError("valves exploded")

        valves = None

        async def outlet(self, body):
            return body
    """
)


@pytest.mark.parametrize(
    "filter_source,stage,error",
    [
        pytest.param(RAISING_OUTLET, "outlet", "outlet exploded", id="outlet"),
        pytest.param(UNBUILDABLE_VALVES, "outlet", "valves exploded", id="valves"),
    ],
)
def test_a_failing_outlet_is_logged_with_the_filter_and_traceback(
    instance, admin, user, upstream, filter_source, stage, error
):
    offset = instance.log_size()
    with installed_function(admin, filter_source, is_global=True) as filter_id:
        with user.client() as client:
            _, message = ask(client, "hello")
        logged = wait_for_log(instance, offset, f"Error in {stage} filter {filter_id}")

    assert f"Error in {stage} filter {filter_id}" in logged
    assert f"RuntimeError: {error}" in logged
    assert message["done"]


def test_a_failing_stream_filter_is_logged_with_the_filter_and_traceback(
    instance, admin, user, upstream
):
    offset = instance.log_size()
    with installed_function(admin, RAISING_STREAM, is_global=True) as filter_id:
        with user.client() as client:
            direct_stream(client)
        logged = wait_for_log(instance, offset, f"Error in stream filter {filter_id}")

    assert f"Error in stream filter {filter_id}" in logged
    assert "RuntimeError: stream exploded" in logged


BROKEN_USER_VALVES = source(
    """
    from pydantic import BaseModel

    class Filter:
        class UserValves(BaseModel):
            def __init__(self, **values):
                raise RuntimeError("user valves exploded")

        async def inlet(self, body, __user__):
            body["messages"][-1]["content"] += " [inlet ran]"
            return body
    """
)


def test_a_failing_user_valves_names_the_filter_and_the_handler_still_runs(
    instance, admin, user, upstream
):
    offset = instance.log_size()
    with installed_function(admin, BROKEN_USER_VALVES, is_global=True) as filter_id:
        with user.client() as client:
            ask(client, "hello")
        logged = wait_for_log(instance, offset, f"Failed to get user valves for filter {filter_id}")

    assert f"Failed to get user valves for filter {filter_id}" in logged
    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == "hello [inlet ran]"


RAISING_INLET = source(
    """
    class Filter:
        async def inlet(self, body):
            raise RuntimeError("inlet exploded")
    """
)

QUIET_OUTLET = source(
    """
    class Filter:
        async def outlet(self, body):
            return body
    """
)


def test_a_failing_inlet_is_not_logged_as_an_error(instance, admin, user, upstream):
    offset = instance.log_size()
    with installed_function(admin, RAISING_INLET, is_global=True) as filter_id:
        with user.client() as client:
            request = {"model": MOCK_MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}
            refused = client.post("/api/chat/completions", json=request)

    assert refused.status_code != 200
    assert "inlet exploded" in refused.text
    assert f"filter {filter_id}" not in instance.log_since(offset)


def test_a_working_filter_logs_nothing(instance, admin, user, upstream):
    offset = instance.log_size()
    with installed_function(admin, QUIET_OUTLET, is_global=True) as filter_id:
        with user.client() as client:
            _, message = ask(client, "hello")

    assert message["content"] == "ok"
    assert f"filter {filter_id}" not in instance.log_since(offset)
