"""Guard: a slow Redis on the sign-in path slows sign-in only.

`signin` calls `is_limited` inline, so every sign-in attempt does two or three Redis round trips
before it answers. Held on a synchronous client those trips run on the event-loop thread, and
with the default `REDIS_SOCKET_TIMEOUT` of None a slow Redis holds the loop for as long as it
takes, so nothing else on the worker (health checks, chats, sockets) is served meanwhile. A
brief latency spike on a shared cache must cost the caller alone.

The test hands sign-in a Redis whose `incr` answers a second late and watches a ticking
coroutine while `signin` runs: if the loop was free, no tick is late. The fake only answers to
`await`, so a limiter that went back to a synchronous client fails on the unawaited call.

Read on upstream dev at 4948842be (2026-09-09), where the tick is a second late; #29977 moved
the limiter to the async client and takes it from `request.app.state.redis`.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request, Response

REDIS_STALL = 1.0
TOLERATED_GAP = 0.3


@pytest.fixture(scope="session")
def auths_module(owui_module):
    return owui_module("open_webui.routers.auths")


class SlowRedis:
    """Every attempt reads as over the limit, so sign-in stops right after the stall."""

    def __init__(self):
        self.reached = False

    async def incr(self, key):
        self.reached = True
        await asyncio.sleep(REDIS_STALL)
        return 100

    async def mget(self, keys):
        return ["100"] * len(keys)


async def _widest_tick_gap(stop: asyncio.Event) -> float:
    widest, last = 0.0, time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        widest, last = max(widest, now - last), now
    return widest


@pytest.mark.asyncio
async def test_signin_waiting_on_redis_leaves_the_loop_free(auths_module):
    slow_redis = SlowRedis()
    app = SimpleNamespace(state=SimpleNamespace(redis=slow_redis))
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "app": app})
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

    if refused.value.status_code != 429 or not slow_redis.reached:
        pytest.fail("sign-in never reached the rate limiter")
    assert widest_gap < TOLERATED_GAP, "the event loop stalled while sign-in waited on Redis"
