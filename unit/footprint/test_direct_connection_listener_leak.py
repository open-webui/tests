"""Regression: a direct-connection streaming request left a listener behind on every bad exit.

open-webui 0.11.4 PR #29509 (commit `aa48106fc`): a direct-connection streaming request
registered a per-request listener before asking the browser to start the completion, and that
listener was only removed once the response had been fully streamed. Every other way the
request could end left it behind for the life of the process: the call to the browser raising,
a non-success status, cancellation while waiting. A client that kept hitting a failing direct
connection grew the server's listener table without bound. The fix removes the listener on every
exit path. The listener table is found by what grows: every module-level container of the
socket and chat modules is measured around the real `generate_direct_chat_completion`, so the
tests do not depend on where upstream keeps it (a socket handler table, later `EVENT_QUEUES`).
The call to the browser is the one I/O boundary, stubbed at `get_event_call`.

Stays a unit test: the leak is a few hundred bytes per request, invisible in the server's size.
Discriminates: passes on dev bbfa876af, fails with the fix's `EVENT_QUEUES.pop` calls removed
from the raising and refused paths (every non-clean exit leaves its queue registered).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request

pytestmark = pytest.mark.regression

METADATA = {"user_id": "u-1", "session_id": "sess-1", "chat_id": "chat-1", "message_id": "msg-1"}


@pytest.fixture(scope="module")
def chat_module(owui_module):
    return owui_module("open_webui.utils.chat")


@pytest.fixture(scope="module")
def socket_module(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture
def module_containers(chat_module, socket_module):
    """Every module-level dict, list and set of the chat and socket modules, by name."""
    containers, seen = {}, set()
    for module in (socket_module, chat_module):
        for name, value in vars(module).items():
            if isinstance(value, (dict, list, set)) and id(value) not in seen:
                seen.add(id(value))
                containers[f"{module.__name__}.{name}"] = value
    return containers


def _sizes(containers: dict) -> dict[str, int]:
    return {name: len(value) for name, value in containers.items()}


def _grown(containers: dict, before: dict[str, int]) -> dict[str, int]:
    return {name: size for name, size in _sizes(containers).items() if size > before[name]}


def _form(stream: bool = True) -> dict:
    return {
        "model": "m",
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": dict(METADATA),
    }


async def _complete(chat_module, event_caller, stream: bool = True):
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "app": FastAPI()}
    request = Request(scope)
    with patch.object(chat_module, "get_event_call", AsyncMock(return_value=event_caller)):
        return await asyncio.wait_for(
            chat_module.generate_direct_chat_completion(
                request=request, form_data=_form(stream), user=None, models={"m": {"id": "m"}}
            ),
            timeout=10,
        )


async def _drain(body_iterator) -> None:
    async for _chunk in body_iterator:
        pass


async def _raising_call(event):
    raise RuntimeError("browser call failed")


async def _refused_call(event):
    return {"status": False, "detail": "refused"}


async def _empty_ack(event):
    return {}


async def _cancelled_call(event):
    raise asyncio.CancelledError()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_caller",
    [_raising_call, _refused_call, _empty_ack, _cancelled_call],
    ids=lambda caller: caller.__name__,
)
async def test_a_failed_stream_request_leaves_no_listener_behind(
    chat_module, module_containers, event_caller
):
    before = _sizes(module_containers)

    for _ in range(25):
        with pytest.raises(BaseException):
            await _complete(chat_module, event_caller)

    assert not _grown(module_containers, before), (
        "failed direct-connection requests left their stream listener registered, so a client "
        "that keeps hitting a failing connection grows the table without bound (#29509)"
    )


@pytest.mark.asyncio
async def test_a_live_stream_holds_its_listener_until_it_is_consumed(
    chat_module, module_containers
):
    async def accepting_call(event):
        return {"status": True}

    before = _sizes(module_containers)
    keys_before = {name: set(table) for name, table in module_containers.items()}
    response = await _complete(chat_module, accepting_call)

    grown = _grown(module_containers, before)
    assert len(grown) == 1, f"a live stream registers exactly one listener, grew: {grown}"
    table_name = next(iter(grown))
    (channel,) = set(module_containers[table_name]) - keys_before[table_name]
    await module_containers[table_name][channel].put({"done": True})
    await asyncio.wait_for(_drain(response.body_iterator), timeout=10)

    assert not _grown(module_containers, before), "a finished stream left its listener behind"


@pytest.mark.asyncio
async def test_a_non_streaming_request_never_registers_a_listener(chat_module, module_containers):
    async def answering_call(event):
        return {"content": "the answer"}

    before = _sizes(module_containers)
    answer = await _complete(chat_module, answering_call, stream=False)

    assert answer == {"content": "the answer"}
    assert not _grown(module_containers, before), "a non-streaming request registered a listener"
