"""Guard: a Redis error inside the session cleanup task is logged and retried.

With `WEBSOCKET_MANAGER=redis`, `periodic_session_pool_cleanup` takes a Redis lock (a `SET NX`)
and renews it while it reaps orphaned socket sessions, once per process. It used to take and
renew the lock outside any `except`, so the first Redis error ended the coroutine: from then on
the session pool only grew, and the only trace was one "Task exception was never retrieved" at
shutdown. #29976 (263e56e27) logs the error and retries. Here the Redis fails the first take
of the session cleanup lock and the first renew once it is held: a living task comes back for
the lock after each, a dead one never asks again. The usage cleanup task, always guarded, fails
its first take too and is the control.

Twin of unit/resilience/test_degraded_redis_keeps_cleanup_tasks_alive.py.
Discriminates: passes on dev bbfa876af, fails with 263e56e27 reverted in a copy of it (the
session lock is asked for once and never again).
"""

from __future__ import annotations

import time
from typing import Generator

import pytest

from harness.instance import LaunchedInstance, launch
from integration.flaky_redis import FlakyRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

SESSION_LOCK = "session_cleanup_lock"
USAGE_LOCK = "usage_cleanup_lock"
LOCK_TIMEOUT = 2  # seconds; the tasks retry after half to all of it


def _takes(lock: str):
    return lambda name, args: name == "SET" and args[0].endswith(lock)


def _renews(lock: str):
    return lambda name, args: name == "EVAL" and "expire" in args[0] and args[2].endswith(lock)


@pytest.fixture(scope="module")
def flaky_redis() -> Generator[FlakyRedis, None, None]:
    redis = FlakyRedis()
    redis.fail_once(_takes(SESSION_LOCK))
    redis.fail_once(_renews(SESSION_LOCK))
    redis.fail_once(_takes(USAGE_LOCK))
    yield redis
    redis.close()


@pytest.fixture(scope="module")
def redis_instance(mock_upstream, flaky_redis) -> Generator[LaunchedInstance, None, None]:
    yield from launch(
        mock_upstream,
        {
            "WEBSOCKET_MANAGER": "redis",
            "WEBSOCKET_REDIS_URL": flaky_redis.url,
            "WEBSOCKET_REDIS_LOCK_TIMEOUT": str(LOCK_TIMEOUT),
        },
    )


def _lock_takes(redis: FlakyRedis, lock: str, at_least: int, timeout: float = 15.0) -> int:
    """How often the lock was asked for, once that reaches `at_least` or the time is up."""
    deadline = time.monotonic() + timeout
    while True:
        takes = sum(args[0].endswith(lock) for args in redis.sent("SET"))
        if takes >= at_least or time.monotonic() > deadline:
            return takes
        time.sleep(0.2)


def test_usage_cleanup_asks_for_its_lock_again_after_a_redis_error(redis_instance, flaky_redis):
    """Control: the sibling task has always logged the error and retried."""
    assert _lock_takes(flaky_redis, USAGE_LOCK, at_least=2) >= 2


def test_session_cleanup_survives_a_redis_error_on_take_and_on_renew(redis_instance, flaky_redis):
    takes = _lock_takes(flaky_redis, SESSION_LOCK, at_least=3)

    assert takes >= 2, "the session cleanup task died on a Redis error taking its lock (#29976)"
    assert takes >= 3, "the session cleanup task died on a Redis error renewing its lock (#29976)"
