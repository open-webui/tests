"""Dependency contract: async-timeout (import name ``async_timeout``).

``async-timeout`` provides the ``async with timeout(seconds): ...``
cancellation primitive — a context manager that schedules a task
cancellation after a deadline and re-raises it as ``asyncio.TimeoutError``.
The Open WebUI backend does not import it directly; it is a *transitive*
dependency of ``aiohttp`` (and, on Python < 3.11, of other async libs),
which the backend uses for nearly all outbound async HTTP (Ollama/OpenAI
proxying, web retrieval, webhooks, audio/image providers). A breaking
bump would surface as request timeouts mis-firing or never firing, deep
inside aiohttp.

This module pins the small public surface aiohttp binds to — the
``timeout`` / ``timeout_at`` factories and the ``Timeout`` context-manager
object — plus offline behavioural contracts that prove the deadline
semantics (completes under budget, raises ``asyncio.TimeoutError`` when
exceeded, ``None`` disables). Everything runs on a local event loop with
``asyncio.sleep`` — no network.

Pattern mirrors test_requests.py: symbol-existence + signature checks plus
offline behavioural contracts. Uses ``depcheck`` from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "async_timeout"
DIST_NAME = "async-timeout"

TOP_LEVEL_SYMBOLS = [
    "timeout",  # timeout(delay) -> Timeout context manager
    "timeout_at",  # timeout_at(deadline) -> Timeout context manager
    "Timeout",  # the context-manager class itself
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`async_timeout` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "async_timeout"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks (API surface aiohttp binds to)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """timeout / timeout_at / Timeout must all exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_factories_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "timeout")
    depcheck.assert_callable(mod, "timeout_at")


def test_timeout_signature(depcheck):
    """timeout(delay) — a single optional float delay. aiohttp calls
    timeout(total) / ceil_timeout. The first parameter must remain."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.timeout)
    params = list(sig.parameters)
    assert params, "timeout() lost its delay parameter"
    assert params[0] in ("delay", "timeout"), f"unexpected first param: {params}"


def test_timeout_at_signature(depcheck):
    """timeout_at(deadline) — loop-clock absolute deadline."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.timeout_at)
    params = list(sig.parameters)
    assert params, "timeout_at() lost its deadline parameter"
    assert params[0] in ("deadline", "when"), f"unexpected first param: {params}"


def test_timeout_object_is_async_context_manager(depcheck):
    """`async with timeout(...)` requires __aenter__/__aexit__. Build the
    object inside a loop (it grabs the running loop at construction)."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        cm = mod.timeout(5)
        assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__")
        for name in ("expired", "deadline", "reschedule", "shift"):
            assert name in dir(cm), f"Timeout.{name} missing"
        return True

    assert asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — deadline semantics on a local loop.
# ---------------------------------------------------------------------------


def test_behaviour_completes_under_budget(depcheck):
    """A block that finishes before the deadline must NOT raise, and the
    Timeout must report not-expired afterwards."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        async with mod.timeout(1.0) as cm:
            await asyncio.sleep(0.001)
        return cm.expired

    # .expired is a custom truthy/falsy object, not a bare bool; test via bool().
    expired = asyncio.run(scenario())
    assert not expired


def test_behaviour_raises_timeout_error_when_exceeded(depcheck):
    """A block exceeding the deadline must raise asyncio.TimeoutError — the
    exact exception aiohttp surfaces as a client timeout."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        try:
            async with mod.timeout(0.01):
                await asyncio.sleep(5)
        except asyncio.TimeoutError:
            return "timed_out"
        return "no_timeout"

    assert asyncio.run(scenario()) == "timed_out"


def test_behaviour_expired_flag_set_on_timeout(depcheck):
    """After a timeout fires, the Timeout object's .expired must be True (some
    consumers inspect it to distinguish timeout from other cancellations)."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        cm = mod.timeout(0.01)
        try:
            async with cm:
                await asyncio.sleep(5)
        except asyncio.TimeoutError:
            pass
        return cm.expired

    assert bool(asyncio.run(scenario())) is True


def test_behaviour_none_disables_timeout(depcheck):
    """timeout(None) must act as a no-op guard (no deadline). aiohttp passes
    None when a timeout is disabled — it must not raise spuriously."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        async with mod.timeout(None):
            await asyncio.sleep(0.001)
        return "ok"

    assert asyncio.run(scenario()) == "ok"


def test_behaviour_timeout_at_with_loop_clock(depcheck):
    """timeout_at uses the running loop's monotonic clock. A deadline already
    in the (near) past must fire promptly."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 0.01
        try:
            async with mod.timeout_at(deadline):
                await asyncio.sleep(5)
        except asyncio.TimeoutError:
            return "timed_out"
        return "no_timeout"

    assert asyncio.run(scenario()) == "timed_out"


def test_behaviour_nested_inner_timeout_fires_first(depcheck):
    """Nested timeouts (aiohttp nests per-request inside per-session) must each
    be independent: the tighter inner deadline fires without the outer one
    masking it."""
    mod = depcheck.load(IMPORT_NAME)

    async def scenario():
        async with mod.timeout(10):  # generous outer
            try:
                async with mod.timeout(0.01):  # tight inner
                    await asyncio.sleep(5)
            except asyncio.TimeoutError:
                return "inner_fired"
        return "outer_swallowed"

    assert asyncio.run(scenario()) == "inner_fired"
