"""Dependency contract: aiodns.

`aiodns` was added to `backend/requirements.txt` and `pyproject.toml` by
commit 2196b4e1f (PR #27440) so that aiohttp resolves DNS on the event loop
(c-ares) instead of the threadpool. 0.11.1 walked that back in commit
c5ec01b1f (PR #28242, "make the aiodns resolver opt-in and pin aiodns to
3.6.1"): c-ares broke name resolution on some hosts (#28013, #28215), so
`env.py` now forces aiohttp's `DefaultResolver` back to `ThreadedResolver`
unless `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER=true`, and the Mistral OCR loader
no longer builds its own `AsyncResolver` connector.

aiodns stays a hard requirement because the opt-in has to work the moment it
is switched on: aiohttp's `AsyncResolver.__init__` raises
`RuntimeError("Resolver requires aiodns library")` when aiodns is not
importable. The backend never imports `aiodns` itself, it reaches it only
through aiohttp, so the contract pinned here is the *reachability* one:
aiodns imports, aiohttp sees it (`aiohttp.resolver.aiodns_default`), and
`aiohttp.AsyncResolver()` actually CONSTRUCTS inside a running loop and is
accepted as `TCPConnector(resolver=...)`. That construction test is what a
future aiohttp or aiodns bump would break (aiohttp probes
`aiodns.DNSResolver.getaddrinfo` at import and falls back when absent).

The opt-in gate is checked by what it does: env.py is imported in a fresh
interpreter with and without the flag and the resolver aliases it leaves behind
are read back, however env.py happens to spell the gate.

All checks are OFFLINE: resolvers and connectors are constructed and closed,
never used to look anything up. Uses the `depcheck` fixture from
unit/deps/conftest.py.

Discriminates: aiodns missing or its aiohttp-facing API drifting fails the
construction tests; in a backend copy, a gate defaulting to on, a gate removed
and one alias no longer rewritten each fail the opt-in test.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiodns"
DIST_NAME = "aiodns"
OPT_IN = "AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER"

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
        "RuntimeError, so AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER=true breaks every "
        "outbound request at connector setup."
    )
    assert aiohttp_resolver.aiodns_default is True


# The three aliases env.py rewrites: connectors read `aiohttp.connector.DefaultResolver`,
# plugin code the top-level one, so an alias left out keeps that path on c-ares.
RESOLVER_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import aiohttp
import open_webui.env  # noqa: F401
modules = (aiohttp, aiohttp.resolver, aiohttp.connector)
print("RESOLVERS", " ".join(module.DefaultResolver.__name__ for module in modules))
"""


def _resolvers_once_env_ran(backend, opt_in: str | None) -> list[str]:
    """aiohttp's DefaultResolver aliases after importing env.py in a fresh interpreter."""
    env = {name: value for name, value in os.environ.items() if name != OPT_IN}
    if opt_in is not None:
        env[OPT_IN] = opt_in
    probe = [sys.executable, "-c", RESOLVER_PROBE, str(backend)]
    probed = subprocess.run(probe, capture_output=True, text=True, timeout=120, env=env)
    assert probed.returncode == 0, f"importing env.py failed\n{probed.stderr[-3000:]}"
    printed = [line for line in probed.stdout.splitlines() if line.startswith("RESOLVERS ")]
    assert printed, f"the probe printed no resolvers\n{probed.stdout[-3000:]}"
    return printed[-1].split()[1:]


def test_async_dns_is_opt_in_and_off_by_default(depcheck, open_webui_backend):
    """0.11.1 (c5ec01b1f) demoted c-ares to opt-in: importing env.py leaves every
    DefaultResolver alias on the threaded resolver unless the flag is set, and the
    flag hands them back to aiohttp's own aiodns default."""
    depcheck.load(IMPORT_NAME)

    assert _resolvers_once_env_ran(open_webui_backend, None) == ["ThreadedResolver"] * 3, (
        "c-ares resolution is back on by default (#28013, #28215)"
    )
    assert _resolvers_once_env_ran(open_webui_backend, "true") == ["AsyncResolver"] * 3


def test_threaded_resolver_override_target_exists(depcheck):
    """env.py assigns `aiohttp.resolver.ThreadedResolver`; if aiohttp ever drops
    or renames it, importing env.py dies at startup."""
    aiohttp_resolver = depcheck.resolve(depcheck.load("aiohttp"), "resolver")
    assert isinstance(aiohttp_resolver.ThreadedResolver, type)


def test_async_resolver_constructs_offline(depcheck):
    """What the opt-in resolves to: `aiohttp.AsyncResolver()`. It needs a
    running loop and raises RuntimeError without aiodns, so constructing it
    inside asyncio.run() is the one assertion that proves the dependency works.
    No DNS query is issued by construction."""
    aiohttp = depcheck.load("aiohttp")
    depcheck.load(IMPORT_NAME)

    async def build_and_close():
        resolver = aiohttp.AsyncResolver()
        await resolver.close()

    asyncio.run(build_and_close())


def test_async_resolver_accepted_by_tcp_connector_offline(depcheck):
    """The shape the opt-in produces:
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
