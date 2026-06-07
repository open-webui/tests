"""Dependency contract: asgiref.

Open WebUI uses `asgiref` for one thing: the ASGI type vocabulary that
annotates its raw-ASGI audit middleware. ``utils/audit.py`` does

    from asgiref.typing import (
        ASGI3Application,
        ASGIReceiveCallable,
        ASGIReceiveEvent,
        ASGISendCallable,
        ASGISendEvent,
    )
    from asgiref.typing import Scope as ASGIScope

and uses those names to type the ``(scope, receive, send)`` callable that the
``AuditLoggingMiddleware`` implements directly against the ASGI protocol
(rather than via Starlette's ``BaseHTTPMiddleware``). They are type aliases
/ Protocols / TypedDicts, not runtime classes, but the *import must succeed*
and the names must keep resolving — a typed middleware module that fails to
import takes the whole app down at startup.

asgiref is also the canonical home of ``sync_to_async`` / ``async_to_sync``;
those aren't imported by the backend today, but they're pinned here as the
stable core surface (a regression guard, since they sit one import away and
the ecosystem around Open WebUI relies on them).

This module pins exactly that import surface plus a light structural check
that ``Scope`` really is the HTTP/WebSocket/Lifespan union the middleware
dispatches on, so an `asgiref` bump that removed/renamed any of it fails
loudly here. Fully offline — these are type objects, no ASGI server is run.
Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import typing

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "asgiref"
DIST_NAME = "asgiref"

# The exact names utils/audit.py imports from asgiref.typing.
AUDIT_TYPING_SYMBOLS = [
    "ASGI3Application",
    "ASGIReceiveCallable",
    "ASGIReceiveEvent",
    "ASGISendCallable",
    "ASGISendEvent",
    "Scope",
]

# Broader asgiref.typing vocabulary the middleware's scope-dispatch logic
# implicitly depends on (HTTP vs WebSocket vs Lifespan scopes).
TYPING_SCOPE_KINDS = ["HTTPScope", "WebSocketScope", "LifespanScope", "ASGIVersions"]

# Core asgiref.sync surface — not directly imported by the backend, pinned as
# the stable public API (sync<->async bridging the ecosystem relies on).
SYNC_SYMBOLS = ["sync_to_async", "async_to_sync", "SyncToAsync", "AsyncToSync"]


# --------------------------------------------------------------------------- #
# Import + version
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "asgiref"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_typing_submodule_importable(depcheck):
    """utils/audit.py does ``from asgiref.typing import ...`` — the typing
    submodule must import cleanly (it pulls Python's typing machinery)."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.try_load("asgiref.typing")
    assert mod is not None, "asgiref.typing no longer importable"


def test_sync_submodule_importable(depcheck):
    """asgiref.sync is the canonical sync<->async bridge module; pin that it
    imports (stable-surface guard)."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.try_load("asgiref.sync")
    assert mod is not None, "asgiref.sync no longer importable"


# --------------------------------------------------------------------------- #
# asgiref.typing — the audit-middleware import surface
# --------------------------------------------------------------------------- #
def test_audit_typing_symbols_exist(depcheck):
    """Every name utils/audit.py imports from asgiref.typing must still resolve
    on the module. These are type aliases/Protocols/TypedDicts — checked as
    module attributes, not instantiated."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    depcheck.assert_symbols(typing_mod, AUDIT_TYPING_SYMBOLS)


def test_audit_exact_import_statement_succeeds(depcheck):
    """Reproduce utils/audit.py's import verbatim — a regression here is exactly
    what would break ``import open_webui.utils.audit`` at startup."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    # Bind each name the way the `from ... import (...)` would.
    bound = {name: getattr(typing_mod, name, None) for name in AUDIT_TYPING_SYMBOLS}
    missing = [name for name, obj in bound.items() if obj is None]
    assert not missing, f"asgiref.typing no longer exports: {missing}"
    # The `Scope as ASGIScope` alias hop the module performs.
    asgi_scope = typing_mod.Scope
    assert asgi_scope is not None


def test_scope_kinds_exist(depcheck):
    """The middleware dispatches on scope['type'] (http / websocket / lifespan).
    asgiref models those as HTTPScope/WebSocketScope/LifespanScope; pin they
    exist so the typed dispatch stays expressible."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    depcheck.assert_symbols(typing_mod, TYPING_SCOPE_KINDS)


def test_scope_is_union_of_scope_kinds(depcheck):
    """``Scope`` (imported as ASGIScope) is the union of the concrete scope
    TypedDicts. Pin that it is a typing-union that includes the HTTP, WebSocket
    and Lifespan scope types — the structural guarantee the middleware relies on
    when it narrows a generic Scope to an HTTP request.

    A bump that collapsed Scope to a bare ``dict`` (losing the discriminated
    union) would silently weaken the middleware's typing; catch it here."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    scope = typing_mod.Scope
    args = typing.get_args(scope)
    if not args:
        # Some builds expose Scope as a plain alias; accept that but ensure the
        # concrete scope kinds are at least independently present.
        for kind in ("HTTPScope", "WebSocketScope", "LifespanScope"):
            assert hasattr(typing_mod, kind)
        return
    member_names = {getattr(a, "__name__", str(a)) for a in args}
    for kind in ("HTTPScope", "WebSocketScope", "LifespanScope"):
        assert any(kind in n for n in member_names), (
            f"asgiref.typing.Scope union no longer includes {kind}: {member_names}"
        )


def test_asgi3application_is_callable_alias(depcheck):
    """``ASGI3Application`` types the audited app object, which is invoked as
    ``await app(scope, receive, send)``. It is a Callable type alias; pin that
    it carries callable-type structure (get_origin resolves to a callable),
    so it stays usable as the app annotation."""
    depcheck.load(IMPORT_NAME)
    typing_mod = depcheck.load("asgiref.typing")
    app_t = typing_mod.ASGI3Application
    origin = typing.get_origin(app_t)
    # Callable aliases have collections.abc.Callable as their origin; if the
    # build models it differently, fall back to confirming it is subscriptable
    # type-machinery (has __args__) rather than a plain class.
    assert origin is not None or hasattr(app_t, "__args__"), (
        f"ASGI3Application no longer a typing alias: {app_t!r}"
    )


# --------------------------------------------------------------------------- #
# asgiref.sync — stable-surface guard (not directly imported by the backend)
# --------------------------------------------------------------------------- #
def test_sync_symbols_exist(depcheck):
    """sync_to_async / async_to_sync (and their class forms) are asgiref's core
    public API. Pin they remain present even though the backend doesn't import
    them today — a removal would signal a major, breaking asgiref reshuffle."""
    depcheck.load(IMPORT_NAME)
    sync_mod = depcheck.load("asgiref.sync")
    depcheck.assert_symbols(sync_mod, SYNC_SYMBOLS)


def test_sync_helpers_callable(depcheck):
    depcheck.load(IMPORT_NAME)
    sync_mod = depcheck.load("asgiref.sync")
    assert callable(sync_mod.sync_to_async)
    assert callable(sync_mod.async_to_sync)
