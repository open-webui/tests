"""Guard: the instance keeps answering while one request waits on a slow Redis.

The sign-in rate limiter used to talk to Redis through a synchronous client on the event-loop
thread, so while one sign-in waited on a slow `INCRBY` nothing else on the worker was served.
Here the fake Redis answers that one command two seconds late; a health check sent once the
sign-in has reached that command has to come back inside a second. The control is the same
health check with nothing slow, which shows the bound itself is easy to meet.

Twin of unit/resilience/test_slow_redis_does_not_stall_the_event_loop.py.

Read on upstream dev at 4948842be (2026-09-09), where the health check waits out the stall;
#29977 moved the limiter to the async client, so the stalled round now asserts the bound too.
Discriminates: passes on dev bbfa876af, fails with d2e62db69 (#29977) reverted in a copy of it
(health waits out the two-second stall).
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

REDIS_STALL = 2.0
HEALTH_TIMEOUT = 1.0


def _health_during_signin(instance, fake_redis) -> float:
    fake_redis.seen.clear()
    signin = threading.Thread(
        target=httpx.post,
        args=(f"{instance.base_url}/api/v1/auths/signin",),
        kwargs={"json": {"email": "nobody@example.com", "password": "x"}, "timeout": 30.0},
    )
    signin.start()
    deadline = time.monotonic() + 10
    while "INCRBY" not in fake_redis.seen:
        if time.monotonic() > deadline:
            pytest.fail("sign-in never reached Redis")
        time.sleep(0.01)
    started = time.perf_counter()
    try:
        httpx.get(f"{instance.base_url}/health", timeout=30.0).raise_for_status()
        return time.perf_counter() - started
    finally:
        signin.join()


def test_health_answers_while_a_signin_waits_on_a_fast_redis(degraded_instance, fake_redis):
    """Control: nothing slow, so health answers well inside the bound."""
    fake_redis.delays.clear()

    elapsed = _health_during_signin(degraded_instance, fake_redis)
    assert elapsed < HEALTH_TIMEOUT, f"health took {elapsed:.2f}s with nothing slow"


def test_health_answers_while_a_signin_waits_on_a_slow_redis(degraded_instance, fake_redis):
    fake_redis.delays["INCRBY"] = REDIS_STALL
    try:
        elapsed = _health_during_signin(degraded_instance, fake_redis)
        assert elapsed < HEALTH_TIMEOUT, f"health took {elapsed:.2f}s beside a stalled sign-in"
    finally:
        fake_redis.delays.clear()
