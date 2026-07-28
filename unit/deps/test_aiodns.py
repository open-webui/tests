"""Dependency contract: aiodns.

`aiodns==4.0.4` was added to `backend/requirements.txt` and `pyproject.toml`
by commit 2196b4e1f (PR #27440). Despite the "makes aiohttp resolve DNS on
the event loop instead of the threadpool" comment on the pin, it is a HARD
runtime requirement, not an optional speedup: the Mistral OCR loader
(`retrieval/loaders/mistral.py`) builds
`aiohttp.TCPConnector(resolver=aiohttp.AsyncResolver())`, and aiohttp's
`AsyncResolver.__init__` raises `RuntimeError("Resolver requires aiodns
library")` when aiodns is not importable. Before this pin existed, Mistral
OCR failed on a stock install.

The backend never imports `aiodns` itself; it reaches it only through
aiohttp. So the contract pinned here is the *reachability* one: aiodns
imports, aiohttp sees it (`aiohttp.resolver.aiodns_default`), and
`aiohttp.AsyncResolver()` actually CONSTRUCTS inside a running loop and is
accepted as `TCPConnector(resolver=...)`. That construction test is what a
future aiohttp or aiodns bump would break (aiohttp probes
`aiodns.DNSResolver.getaddrinfo` at import and falls back when absent).

All checks are OFFLINE: resolvers and connectors are constructed and closed,
never used to look anything up. Uses the `depcheck` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiodns"
DIST_NAME = "aiodns"

# aiodns symbols aiohttp reaches for in `aiohttp/resolver.py`.
USED_SYMBOLS = [
    "error",
    "DNSResolver",
    # aiohttp's import-time probe is literally
    # `aiodns_default = hasattr(aiodns.DNSResolver, "getaddrinfo")`; without it
    # aiohttp silently demotes DefaultResolver back to the threadpool resolver.
    "DNSResolver.getaddrinfo",
    "DNSResolver.gethostbyname",
    "DNSResolver.query",
    "DNSResolver.close",
]


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "aiodns"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_pycares_backend_present(depcheck):
    """aiodns is a thin async wrapper over pycares; without the C-ares
    binding installed, importing aiodns fails and AsyncResolver goes with it."""
    depcheck.load("pycares")


# --------------------------------------------------------------------------- #
# Symbol existence (API surface aiohttp uses)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    """aiohttp's AsyncResolver calls getaddrinfo/gethostbyname/query and closes
    the resolver on teardown."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_dns_resolver_accepts_loop_kwarg(depcheck):
    """aiohttp constructs `aiodns.DNSResolver(*args, loop=..., **kwargs)`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.DNSResolver.__init__, ["loop"])


# --------------------------------------------------------------------------- #
# The real contract: aiohttp sees aiodns and AsyncResolver constructs
# --------------------------------------------------------------------------- #


def test_aiohttp_detects_aiodns(depcheck):
    """`aiohttp.resolver.aiodns_default` is True only when aiodns imported
    cleanly at aiohttp import time; False means AsyncResolver is unusable."""
    aiohttp_resolver = depcheck.resolve(depcheck.load("aiohttp"), "resolver")
    assert aiohttp_resolver.aiodns is not None, (
        "aiohttp could not import aiodns; aiohttp.AsyncResolver() raises "
        "RuntimeError and the Mistral OCR loader fails at connector setup."
    )
    assert aiohttp_resolver.aiodns_default is True


def test_default_resolver_is_async_resolver(depcheck):
    """With aiodns installed, aiohttp aliases DefaultResolver to AsyncResolver,
    so every connector that does not pass a resolver also resolves on the loop."""
    aiohttp_resolver = depcheck.resolve(depcheck.load("aiohttp"), "resolver")
    assert aiohttp_resolver.DefaultResolver is aiohttp_resolver.AsyncResolver


def test_async_resolver_constructs_offline(depcheck):
    """mistral.py's exact call: `aiohttp.AsyncResolver()`. It needs a running
    loop and raises RuntimeError without aiodns, so constructing it inside
    asyncio.run() is the one assertion that proves the dependency works. No
    DNS query is issued by construction."""
    aiohttp = depcheck.load("aiohttp")
    depcheck.load(IMPORT_NAME)

    async def build_and_close():
        resolver = aiohttp.AsyncResolver()
        await resolver.close()

    asyncio.run(build_and_close())


def test_async_resolver_accepted_by_tcp_connector_offline(depcheck):
    """The full mistral.py expression:
    `aiohttp.TCPConnector(resolver=aiohttp.AsyncResolver(), ...)`. Build it
    and close it without connecting anywhere."""
    aiohttp = depcheck.load("aiohttp")
    depcheck.load(IMPORT_NAME)

    async def build_and_close():
        resolver = aiohttp.AsyncResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=300)
        await connector.close()
        await resolver.close()

    asyncio.run(build_and_close())


def test_async_resolver_exposes_resolve_and_close(depcheck):
    """The connector drives the resolver through `resolve()` and shuts it down
    through `close()`; both must stay on the class."""
    aiohttp = depcheck.load("aiohttp")
    for name in ("resolve", "close"):
        assert callable(getattr(aiohttp.AsyncResolver, name, None)), (
            f"aiohttp.AsyncResolver.{name} missing/not callable"
        )
