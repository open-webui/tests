"""Regression: a direct-connection streaming request left a listener behind on every bad exit.

open-webui 0.11.4 PR #29509 (commit `aa48106fc`): a direct-connection
streaming request registered a per-request socket handler before asking the
browser to start the completion, and that handler was only removed once the
response had been fully streamed. Every other way the request could end left
it behind for the life of the process: the call to the browser raising, a
non-success status, cancellation while waiting, a response body closed or
dropped before it finished. A client that repeatedly hit a failing direct
connection grew the server's handler table and the closures it holds without
bound, and nothing ever cleaned it up.

The fix moves the removal to one helper and calls it on every exit path.
`0180efecf` later replaced the per-request socket handler with an entry in the
shared `EVENT_QUEUES` dict (routed by the catch-all socket handler and, behind
Redis, a pubsub listener), so the unbounded-growth failure now lives there: each
entry holds a queue and the closure over the request's channel. The tests below
drive the real `generate_direct_chat_completion` and pin whichever per-request
container the checkout registers: `EVENT_QUEUES` on dev, `sio.handlers['/']` on
older refs. Both are unbounded, keyed per request, and only cleared on a clean
finish, which is exactly the footprint shape `unit/footprint/` pins.

Discriminates: passes on 344ea5306 (0.11.4), fails on aa48106fc^ (every
non-clean exit leaves the per-request listener registered).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

DIRECT_METADATA = {
    "user_id": "u-1",
    "session_id": "sess-1",
    "chat_id": "chat-1",
    "message_id": "msg-1",
}


def _form(stream: bool = True) -> dict:
    return {
        "model": "m",
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": dict(DIRECT_METADATA),
    }


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


@pytest.fixture(scope="module")
def chat_module(owui_module):
    return owui_module("open_webui.utils.chat")


def _per_request_registry(chat_module):
    """The per-request listener table this checkout registers into.

    dev routes stream events through `EVENT_QUEUES`; the shape the fix landed
    against registered a per-channel handler in `sio.handlers['/']`. Both are
    the thing a failing request leaks."""
    if hasattr(chat_module, "EVENT_QUEUES"):
        return chat_module.EVENT_QUEUES
    return chat_module.sio.handlers["/"]


def _registry_keys(registry) -> set:
    return set(registry)


async def _run_direct(chat_module, event_caller, form=None):
    """Drive the real completion entry point with the socket session stubbed.

    Returns the per-request registry entries the call left behind."""
    registry = _per_request_registry(chat_module)
    before = _registry_keys(registry)
    with patch.object(chat_module, "get_event_call", AsyncMock(return_value=event_caller)):
        try:
            result = await asyncio.wait_for(
                chat_module.generate_direct_chat_completion(
                    _request(), dict(form or _form()), user=None, models={"m": {"id": "m"}}
                ),
                timeout=10,
            )
        except BaseException:
            result = None
    await asyncio.sleep(0)
    return _registry_keys(registry) - before, result


# --- narrow: every way a request can end badly ----------------------------------


@pytest.mark.asyncio
async def test_a_call_that_raises_leaves_no_listener_behind(chat_module):
    async def failing_caller(event):
        raise RuntimeError("browser call failed")

    leftover, _ = await _run_direct(chat_module, failing_caller)

    assert not leftover, (
        "a direct-connection request whose call to the browser raised left its "
        "stream listener registered, so a failing connection grew the server's "
        "handler table without bound (#29509)"
    )


@pytest.mark.asyncio
async def test_a_non_success_status_leaves_no_listener_behind(chat_module):
    async def refusing_caller(event):
        return {"status": False, "detail": "refused"}

    leftover, _ = await _run_direct(chat_module, refusing_caller)

    assert not leftover, (
        "a direct-connection request the browser refused left its stream listener "
        "registered for the life of the process (#29509)"
    )


@pytest.mark.asyncio
async def test_an_ack_without_a_status_leaves_no_listener_behind(chat_module):
    async def empty_caller(event):
        return {}

    leftover, _ = await _run_direct(chat_module, empty_caller)

    assert not leftover, "an ack carrying no status leaked the stream listener (#29509)"


@pytest.mark.asyncio
async def test_a_cancelled_call_leaves_no_listener_behind(chat_module):
    async def cancelled_caller(event):
        raise asyncio.CancelledError()

    leftover, _ = await _run_direct(chat_module, cancelled_caller)

    assert not leftover, (
        "cancelling the request while it waited for the browser leaked the stream "
        "listener; the guard must catch BaseException, and cancellation is not an "
        "Exception (#29509)"
    )


@pytest.mark.asyncio
async def test_many_failing_requests_grow_nothing(chat_module):
    """The footprint invariant: repeated failures must not accumulate entries."""
    async def failing_caller(event):
        raise RuntimeError("browser call failed")

    registry = _per_request_registry(chat_module)
    for attempt in range(25):
        leftover, _ = await _run_direct(chat_module, failing_caller)
        assert not leftover, f"attempt {attempt} left a listener behind"

    assert len(registry) == 0, (
        "repeated failing direct-connection requests accumulated listeners; a client "
        "that keeps hitting a failing connection grows the table without bound (#29509)"
    )


# --- nearby: the clean paths still clean up, and streaming still works ------------


@pytest.mark.asyncio
async def test_a_live_stream_holds_its_entry_until_consumed(chat_module):
    async def ok_caller(event):
        return {"status": True}

    registry = _per_request_registry(chat_module)
    before = _registry_keys(registry)
    with patch.object(chat_module, "get_event_call", AsyncMock(return_value=ok_caller)):
        response = await asyncio.wait_for(
            chat_module.generate_direct_chat_completion(
                _request(), dict(_form()), user=None, models={"m": {"id": "m"}}
            ),
            timeout=10,
        )

    registered = _registry_keys(registry) - before
    assert registered, "a live stream keeps its listener entry while the response is open"

    # Feed the stream's queue: the closure the handler fills, reached by shape so the
    # test works on both the EVENT_QUEUES ref and the sio-handler ref.
    stream_queue = None
    entry = registry[next(iter(registered))]
    if isinstance(entry, asyncio.Queue):
        stream_queue = entry
    else:
        for cell in entry.__closure__ or ():
            if isinstance(cell.cell_contents, asyncio.Queue):
                stream_queue = cell.cell_contents
    assert stream_queue is not None
    await stream_queue.put({"done": True})

    async for _chunk in response.body_iterator:
        pass

    assert not (_registry_keys(registry) & registered), (
        "a fully streamed response left its listener behind after the stream ended (#29509)"
    )


@pytest.mark.asyncio
async def test_a_non_streaming_request_never_registers_a_listener(chat_module):
    async def ok_caller(event):
        return {"content": "the answer"}

    registry = _per_request_registry(chat_module)
    before = _registry_keys(registry)
    with patch.object(chat_module, "get_event_call", AsyncMock(return_value=ok_caller)):
        await asyncio.wait_for(
            chat_module.generate_direct_chat_completion(
                _request(), dict(_form(stream=False)), user=None, models={"m": {"id": "m"}}
            ),
            timeout=10,
        )

    assert _registry_keys(registry) == before, (
        "a non-streaming direct request registered a stream listener it never uses (#29509)"
    )
