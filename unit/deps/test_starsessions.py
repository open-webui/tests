"""Dependency contract: starsessions.

``starsessions`` provides Open WebUI's server-side session machinery for
OAuth/social-login flows. main.py wires it as ASGI middleware: when a
Redis URL is configured it uses ``SessionAutoloadMiddleware`` +
``SessionMiddleware`` (imported aliased as ``StarSessionsMiddleware``)
backed by ``starsessions.stores.redis.RedisStore``; otherwise it falls
back to Starlette's own cookie ``SessionMiddleware`` (a *different* class —
not pinned here). The exact call shapes the backend depends on:

    RedisStore(url=REDIS_URL, prefix="...:session:")
    app.add_middleware(SessionAutoloadMiddleware)
    app.add_middleware(
        StarSessionsMiddleware,
        store=...,
        cookie_name="owui-session",
        cookie_same_site=...,
        cookie_https_only=...,
    )

A breaking bump (renamed middleware kwargs, moved RedisStore, changed
store coroutine surface) would break session persistence and OAuth login
silently. This module pins those symbols + the keyword arguments main.py
passes, and exercises the store read/write/remove contract offline via
the in-memory store (and constructs RedisStore without connecting).

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "starsessions"
DIST_NAME = "starsessions"

# Top-level symbols main.py imports (directly or aliased).
TOP_LEVEL_SYMBOLS = [
    "SessionMiddleware",  # imported `as StarSessionsMiddleware`
    "SessionAutoloadMiddleware",  # added bare to autoload sessions per-request
    "SessionStore",  # base store type
    "InMemoryStore",  # used here for the offline behavioural contract
    "load_session",  # session lifecycle helpers (public API surface)
    "get_session_id",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`starsessions` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "starsessions"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_redis_store_submodule_imports(depcheck):
    """main.py does `from starsessions.stores.redis import RedisStore`; the
    submodule must remain importable and expose RedisStore."""
    mod = depcheck.load("starsessions.stores.redis")
    assert hasattr(mod, "RedisStore")


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level starsessions symbol main.py resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_middlewares_are_classes(depcheck):
    """Both middlewares are passed to app.add_middleware(); they must be
    constructable classes (ASGI middleware)."""
    mod = depcheck.load(IMPORT_NAME)
    import inspect

    assert inspect.isclass(mod.SessionMiddleware)
    assert inspect.isclass(mod.SessionAutoloadMiddleware)


# ---------------------------------------------------------------------------
# Signature contracts — main.py passes specific kwargs.
# ---------------------------------------------------------------------------


def test_session_middleware_accepts_our_kwargs(depcheck):
    """StarSessionsMiddleware is added with store=, cookie_name=,
    cookie_same_site=, cookie_https_only=. All four kwargs must remain on
    SessionMiddleware.__init__ (the starsessions one, NOT Starlette's)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.SessionMiddleware.__init__,
        ["store", "cookie_name", "cookie_same_site", "cookie_https_only"],
    )


def test_session_autoload_middleware_signature(depcheck):
    """SessionAutoloadMiddleware is added bare (just the app). Its __init__
    must accept the ASGI app positionally with everything else optional."""
    mod = depcheck.load(IMPORT_NAME)
    import inspect

    sig = inspect.signature(mod.SessionAutoloadMiddleware.__init__)
    # `app` is required; `paths` (and any others) must be optional so the
    # no-extra-kwargs add_middleware call works.
    required = [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert required == ["app"], f"unexpected required params: {required}"


def test_redis_store_accepts_url_and_prefix(depcheck):
    """RedisStore(url=REDIS_URL, prefix='...:session:') — both kwargs must
    remain accepted."""
    mod = depcheck.load("starsessions.stores.redis")
    depcheck.assert_params(mod.RedisStore.__init__, ["url", "prefix"])


# ---------------------------------------------------------------------------
# RedisStore construction contract (OFFLINE — lazy, never connects).
# ---------------------------------------------------------------------------


def test_redis_store_constructs_without_connecting(depcheck):
    """main.py builds RedisStore(url=..., prefix=...) at import time, before any
    request. Construction must be lazy (no socket) and yield the store's
    read/write/remove coroutine surface."""
    mod = depcheck.load("starsessions.stores.redis")
    store = mod.RedisStore(url="redis://localhost:6379/0", prefix="owui:session:")
    assert store is not None
    for name in ("read", "write", "remove"):
        assert hasattr(store, name), f"RedisStore.{name} missing"
        assert callable(getattr(store, name))


def test_redis_store_is_a_session_store(depcheck):
    """RedisStore must remain a SessionStore subclass so the middleware accepts
    it as its `store`."""
    base = depcheck.load(IMPORT_NAME).SessionStore
    redis_store = depcheck.load("starsessions.stores.redis").RedisStore
    assert issubclass(redis_store, base)


def test_redis_store_prefix_callable_form_accepted(depcheck):
    """The prefix kwarg also accepts a callable (prefix_factory); main.py uses
    the str form but the type must stay flexible to avoid a future break."""
    mod = depcheck.load("starsessions.stores.redis")
    store = mod.RedisStore(url="redis://localhost:6379/0", prefix=lambda sid: f"p:{sid}")
    assert store is not None


# ---------------------------------------------------------------------------
# Store read/write/remove behavioural contract (OFFLINE via InMemoryStore).
# The middleware drives any SessionStore through these three coroutines; we
# pin the round-trip semantics without a Redis server.
# ---------------------------------------------------------------------------


def test_behaviour_inmemory_store_read_write_remove(depcheck):
    """Pin the SessionStore coroutine contract the middleware relies on:
    write(session_id, data, lifetime, ttl) persists bytes, read returns them,
    remove deletes (subsequent read yields empty bytes)."""
    mod = depcheck.load(IMPORT_NAME)
    store = mod.InMemoryStore()

    async def scenario():
        new_id = await store.write("sid-1", b"payload", lifetime=3600, ttl=3600)
        got = await store.read("sid-1", lifetime=3600)
        await store.remove("sid-1")
        after = await store.read("sid-1", lifetime=3600)
        return new_id, got, after

    new_id, got, after = asyncio.run(scenario())
    assert new_id == "sid-1"
    assert got == b"payload"
    # After removal the store must report no data (empty bytes, not the old).
    assert after in (b"", None) or after != b"payload"


def test_behaviour_store_overwrite_updates_value(depcheck):
    """A second write to the same session id must replace the stored payload
    (session updates on each request)."""
    mod = depcheck.load(IMPORT_NAME)
    store = mod.InMemoryStore()

    async def scenario():
        await store.write("sid-2", b"first", lifetime=3600, ttl=3600)
        await store.write("sid-2", b"second", lifetime=3600, ttl=3600)
        return await store.read("sid-2", lifetime=3600)

    assert asyncio.run(scenario()) == b"second"


def test_behaviour_read_unknown_session_is_empty(depcheck):
    """Reading an id that was never written must yield empty/None, not raise —
    the middleware treats this as a fresh session."""
    mod = depcheck.load(IMPORT_NAME)
    store = mod.InMemoryStore()

    async def scenario():
        return await store.read("never-written", lifetime=3600)

    result = asyncio.run(scenario())
    assert result in (b"", None) or not result
