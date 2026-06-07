"""Dependency contract: aiocache.

Open WebUI uses aiocache purely through its ``@cached`` decorator to memoise
the expensive model-list aggregations: ``routers/openai.py`` and
``routers/ollama.py`` both wrap their async ``get_all_models(request, user)``
in ``@cached(ttl=MODELS_CACHE_TTL, key=lambda _, user: ...)``. The decorator
is imported in six modules (main.py, utils/{chat,middleware,models}.py and
the two routers) — always as ``from aiocache import cached``.

The contract the backend leans on is narrow but load-bearing:
  - ``cached`` is a decorator that accepts ``ttl=`` and a ``key=`` *callable*
    receiving the wrapped function's positional args (the lambdas build the
    cache key from the ``user`` arg, namespacing per-user vs anonymous);
  - it wraps an ``async def`` and the wrapper is itself awaitable;
  - within the TTL a repeated call with the same key returns the cached
    value WITHOUT re-invoking the body (the whole point — avoid re-fanning
    every model backend on each request);
  - distinct keys are cached independently;
  - the default backend is in-process memory (no Redis/network needed for
    the decorator to function).

A bump that renamed ``cached``, changed the ``key`` callable contract, or
made the wrapper non-awaitable would break model listing app-wide. This
module pins the API surface plus those offline behavioural guarantees.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiocache"
DIST_NAME = "aiocache"

# Top-level symbols the backend (and the cached decorator machinery) relies on.
USED_SYMBOLS = [
    "cached",  # the only symbol the backend imports
    "Cache",
    "SimpleMemoryCache",
    "BaseCache",
    "caches",
    # submodules the package exposes
    "backends",
    "decorators",
    "serializers",
]


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "aiocache"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every aiocache symbol the codebase relies on must still exist; chiefly
    `cached`, the only one actually imported."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_cached_is_usable_as_decorator(depcheck):
    """`cached` must be callable as a decorator factory: cached(ttl=, key=)
    returns something that wraps a function. (In 0.12 it's a class whose
    instances are decorators; the backend uses it as @cached(...).)"""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.cached)
    deco = mod.cached(ttl=60)
    assert callable(deco)


def test_cached_accepts_ttl_and_key_params(depcheck):
    """The backend calls cached(ttl=..., key=lambda ...). Those parameter
    names must remain accepted by the decorator's constructor/factory."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.cached.__init__ if inspect.isclass(mod.cached) else mod.cached)
    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    for name in ("ttl", "key"):
        assert name in params or has_var_kw, f"cached no longer accepts {name!r} (sig: {sig})"


def test_default_cache_is_memory(depcheck):
    """The backend never configures a backend, so the decorator must default
    to the in-process SimpleMemoryCache (no Redis/network required to work)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.cached.__init__ if inspect.isclass(mod.cached) else mod.cached)
    cache_param = sig.parameters.get("cache")
    assert cache_param is not None, "cached lost its `cache` parameter"
    # Default should be the memory cache class (not a network backend).
    assert cache_param.default is mod.SimpleMemoryCache


def test_simplememorycache_is_basecache(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.SimpleMemoryCache, mod.BaseCache)


# ---------------------------------------------------------------------------
# Behavioural: the @cached decorator memoises an async function offline.
# ---------------------------------------------------------------------------


def test_cached_wraps_async_and_returns_value(depcheck):
    """@cached over an async def must yield an awaitable wrapper that returns
    the body's value on first call."""
    mod = depcheck.load(IMPORT_NAME)

    @mod.cached(ttl=60)
    async def fn(x):
        return x * 2

    result = asyncio.run(fn(21))
    assert result == 42


def test_cached_memoises_within_ttl(depcheck):
    """The core guarantee: within the TTL, a second call with the same args
    returns the cached value and does NOT re-execute the body. (openai.py /
    ollama.py rely on this to avoid re-fanning every model backend.)"""
    mod = depcheck.load(IMPORT_NAME)
    calls = {"n": 0}

    @mod.cached(ttl=300)
    async def fn(x):
        calls["n"] += 1
        return x * 10

    async def scenario():
        a = await fn(5)
        b = await fn(5)  # same key -> served from cache
        return a, b

    a, b = asyncio.run(scenario())
    assert a == 50 and b == 50
    assert calls["n"] == 1, "cached body re-executed despite a cache hit"


def test_cached_key_param_is_static_not_callable(depcheck):
    """CONTRACT PIN + latent-bug guard. In aiocache 0.12 the ``key=`` argument
    is a STATIC cache key, not a per-call callable: passing a callable makes
    that callable object itself the (constant) key, so EVERY call collides to
    one entry regardless of arguments. The per-call hook is ``key_builder=``.

    routers/openai.py:488 and routers/ollama.py:302 currently do
    ``@cached(ttl=..., key=lambda _, user: f'..._{user.id}')`` — under this
    semantics the lambda is a constant key, the per-user namespacing never
    happens, and one user's (permission-filtered) model list is served to
    every other user (and to anonymous callers) within the TTL. The intended
    keyword is ``key_builder=``. This test pins the real dependency behaviour
    so the divergence is documented; see test_cached_key_builder_namespaces
    for the correct call shape."""
    mod = depcheck.load(IMPORT_NAME)
    bodies = {"n": 0}

    @mod.cached(ttl=300, key=lambda r, u: f"models_{u}")
    async def get_all_models(request, user):
        bodies["n"] += 1
        return {"for": user}

    async def scenario():
        a = await get_all_models("req", "u1")
        b = await get_all_models("req", "u2")  # different arg, SAME static key
        return a, b

    a, b = asyncio.run(scenario())
    # The bug surfaces: u2's call returns u1's cached payload (one shared entry).
    assert a == {"for": "u1"}
    assert b == {"for": "u1"}, (
        "aiocache `key=` is no longer a static key — if a callable is now "
        "invoked per-call, the openai/ollama @cached(key=lambda...) usage "
        "would suddenly start namespacing; revisit those call sites."
    )
    assert bodies["n"] == 1, "key= stopped collapsing all calls to one entry"


def test_cached_key_builder_namespaces(depcheck):
    """The CORRECT per-call hook is ``key_builder=callable``: it receives
    ``(func, *args, **kwargs)`` and its return value is the cache key. This is
    what the openai/ollama call sites *intend* — distinct users must cache
    independently. Pin the contract the fix would rely on."""
    mod = depcheck.load(IMPORT_NAME)
    bodies = {"n": 0}

    class User:
        def __init__(self, uid):
            self.id = uid

    def build_key(_func, request, user):
        return f"all_models_{user.id}" if user else "all_models"

    @mod.cached(ttl=300, key_builder=build_key)
    async def get_all_models(request, user):
        bodies["n"] += 1
        return {"for": user.id}

    async def scenario():
        u1, u2 = User("u1"), User("u2")
        r1 = await get_all_models("req", u1)
        r1b = await get_all_models("req2", u1)  # same user.id -> cache hit
        r2 = await get_all_models("req", u2)  # different user.id -> miss
        return r1, r1b, r2

    r1, r1b, r2 = asyncio.run(scenario())
    assert r1 == {"for": "u1"}
    assert r1b == {"for": "u1"}, "key_builder did not reuse the key for the same user.id"
    assert r2 == {"for": "u2"}, "key_builder did not namespace distinct users"
    assert bodies["n"] == 2, f"expected 2 body executions (u1 then u2), got {bodies['n']}"


def test_cached_key_builder_param_exists(depcheck):
    """`key_builder` must remain a recognised parameter (it is the supported
    callable hook the openai/ollama caches should be using)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.cached.__init__ if inspect.isclass(mod.cached) else mod.cached)
    assert "key_builder" in sig.parameters


def test_cached_distinct_args_distinct_cache(depcheck):
    """Without a custom key, distinct args must produce distinct cache entries
    (the default key-builder keys on the call arguments)."""
    mod = depcheck.load(IMPORT_NAME)
    bodies = {"n": 0}

    @mod.cached(ttl=300)
    async def fn(x):
        bodies["n"] += 1
        return x

    async def scenario():
        return await fn(1), await fn(2), await fn(1)

    a, b, c = asyncio.run(scenario())
    assert (a, b, c) == (1, 2, 1)
    assert bodies["n"] == 2, "distinct args were not cached independently"


def test_cached_wrapper_is_awaitable(depcheck):
    """Calling the wrapped function must return an awaitable (it is awaited
    everywhere in the backend), and only run the body when awaited."""
    mod = depcheck.load(IMPORT_NAME)
    ran = {"v": False}

    @mod.cached(ttl=60)
    async def fn():
        ran["v"] = True
        return "ok"

    async def scenario():
        coro = fn()
        assert inspect.isawaitable(coro)
        assert ran["v"] is False, "body ran before the coroutine was awaited"
        return await coro

    assert asyncio.run(scenario()) == "ok"
    assert ran["v"] is True


# ---------------------------------------------------------------------------
# SimpleMemoryCache direct contract (the default backend) — offline.
# ---------------------------------------------------------------------------


def test_memory_cache_set_get_delete(depcheck):
    """The memory backend must support the basic async set/get/delete/exists
    cycle the decorator builds on."""
    mod = depcheck.load(IMPORT_NAME)
    cache = mod.SimpleMemoryCache()

    async def scenario():
        await cache.set("k", "v")
        got = await cache.get("k")
        exists = await cache.exists("k")
        deleted = await cache.delete("k")
        gone = await cache.get("k")
        return got, exists, deleted, gone

    got, exists, deleted, gone = asyncio.run(scenario())
    assert got == "v"
    assert exists is True
    assert deleted == 1
    assert gone is None


def test_memory_cache_methods_are_coroutines(depcheck):
    """get/set/delete on the memory backend must be coroutine functions
    (awaited throughout aiocache's internals)."""
    mod = depcheck.load(IMPORT_NAME)
    cache = mod.SimpleMemoryCache()
    for name in ("get", "set", "delete", "exists", "clear"):
        meth = getattr(cache, name)
        assert asyncio.iscoroutinefunction(meth), f"SimpleMemoryCache.{name} is not async"
