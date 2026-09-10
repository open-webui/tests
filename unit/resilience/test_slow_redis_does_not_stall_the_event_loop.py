"""Guard: a slow Redis on the sign-in path slows sign-in only.

`signin_rate_limiter` holds a synchronous redis-py client and `signin` calls `is_limited` inline,
so every sign-in attempt does two or three blocking round trips on the event-loop thread. With
the default `REDIS_SOCKET_TIMEOUT` of None a slow Redis holds the loop for as long as it takes,
and nothing else on the worker (health checks, chats, sockets) is served meanwhile. A brief
latency spike on a shared cache must cost the caller alone.

The test replaces the limiter's client with one whose `incr` blocks for a second and watches a
ticking coroutine while `signin` runs: if the loop was free, no tick is late.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where the tick is a second late; strict
`xfail`. Unmarked: no issue filed yet.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import HTTPException, Request, Response

REDIS_STALL = 1.0
TOLERATED_GAP = 0.3


@pytest.fixture(scope="session")
def auths_module(owui_module):
    return owui_module("open_webui.routers.auths")


class SlowRedis:
    """Every attempt reads as over the limit, so sign-in stops right after the stall."""

    def incr(self, key):
        time.sleep(REDIS_STALL)
        return 100

    def mget(self, keys):
        return ["100"] * len(keys)


async def _widest_tick_gap(stop: asyncio.Event) -> float:
    widest, last = 0.0, time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        widest, last = max(widest, now - last), now
    return widest


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="sync redis client on the loop")
@pytest.mark.asyncio
async def test_signin_waiting_on_redis_leaves_the_loop_free(auths_module, monkeypatch):
    monkeypatch.setattr(auths_module.signin_rate_limiter, "r", SlowRedis())
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    form = auths_module.SigninForm(email="someone@example.com", password="x")
    stop = asyncio.Event()
    ticker = asyncio.create_task(_widest_tick_gap(stop))
    await asyncio.sleep(0)  # let the ticker take its first timestamp

    try:
        with pytest.raises(HTTPException) as refused:
            await auths_module.signin(request, Response(), form, db=None)
    finally:
        stop.set()
        widest_gap = await ticker

    if refused.value.status_code != 429:
        pytest.fail("sign-in never reached the rate limiter")
    assert widest_gap < TOLERATED_GAP, "the event loop stalled while sign-in waited on Redis"
