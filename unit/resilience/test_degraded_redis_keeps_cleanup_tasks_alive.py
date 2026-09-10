"""Guard: a Redis error inside a periodic cleanup task is logged and retried.

`periodic_session_pool_cleanup` and `periodic_usage_pool_cleanup` are started once per process
and own the reaping of orphaned socket sessions and stale usage entries. The usage task wraps
its whole loop in `except Exception` and sleeps before retrying. The session task does not: its
lock acquire sits outside the `try`, and every Redis call inside it is unguarded, so the first
timeout or connection reset during a Redis blip ends the coroutine. From then on the pool only
grows, and the only trace is one "Task exception was never retrieved" at shutdown.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where the session task dies on the first
error; strict `xfail`. The usage task survives and is the control. Unmarked: no issue filed
yet.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(scope="session")
def socket_main_module(owui_module):
    return owui_module("open_webui.socket.main")


def _redis_blip():
    raise ConnectionError("Error while reading from socket")


async def _run_briefly(coroutine_function) -> asyncio.Task:
    task = asyncio.create_task(coroutine_function())
    await asyncio.sleep(0.2)
    return task


async def _finish(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="the acquire is outside the try")
@pytest.mark.asyncio
async def test_session_cleanup_survives_a_redis_error_on_acquire(socket_main_module, monkeypatch):
    monkeypatch.setattr(socket_main_module, "session_aquire_func", _redis_blip)

    task = await _run_briefly(socket_main_module.periodic_session_pool_cleanup)
    try:
        assert not task.done(), f"session cleanup task died: {task.exception()!r}"
    finally:
        await _finish(task)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="no except around the sweep")
@pytest.mark.asyncio
async def test_session_cleanup_survives_a_redis_error_while_holding_the_lock(
    socket_main_module, monkeypatch
):
    monkeypatch.setattr(socket_main_module, "session_aquire_func", lambda: True)
    monkeypatch.setattr(socket_main_module, "session_release_func", lambda: True)
    monkeypatch.setattr(socket_main_module, "session_renew_func", _redis_blip)

    task = await _run_briefly(socket_main_module.periodic_session_pool_cleanup)
    try:
        assert not task.done(), f"session cleanup task died: {task.exception()!r}"
    finally:
        await _finish(task)


@pytest.mark.asyncio
async def test_usage_cleanup_survives_a_redis_error_on_acquire(socket_main_module, monkeypatch):
    """Control: the sibling task logs the error and retries after its delay."""
    monkeypatch.setattr(socket_main_module, "aquire_func", _redis_blip)

    task = await _run_briefly(socket_main_module.periodic_usage_pool_cleanup)
    try:
        assert not task.done(), f"usage cleanup task died: {task.exception()!r}"
    finally:
        await _finish(task)
