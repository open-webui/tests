"""Dependency contract: requests.

Open WebUI uses `requests` pervasively for outbound HTTP (retrieval web
loaders, image fetches, OAuth, provider calls, etc.). This module pins
the slice of the requests API the codebase actually relies on, so a
`requests` bump that removed/renamed any of it fails loudly instead of
surfacing as a runtime AttributeError deep in a request path.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API
surface) + offline behavioural contracts (no network). Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "requests"
DIST_NAME = "requests"

# Symbols the Open WebUI backend references on `requests`.
USED_SYMBOLS = [
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "request",
    "Session",
    "Response",
    "RequestException",
    "exceptions.RequestException",
    "exceptions.ConnectionError",
    "exceptions.Timeout",
    "exceptions.HTTPError",
    "adapters.HTTPAdapter",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "requests"


def test_used_symbols_exist(depcheck):
    """Every requests symbol the codebase calls must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_http_verbs_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for verb in ("get", "post", "put", "patch", "delete", "head", "request"):
        depcheck.assert_callable(mod, verb)


def test_get_signature_supports_our_kwargs(depcheck):
    """retrieval/utils.py calls requests.get(url, stream=, timeout=,
    allow_redirects=, headers=, ...). Those kwargs must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    # requests routes verbs through Session.request(**kwargs); the public
    # functions accept **kwargs, so assert the underlying request supports them.
    depcheck.assert_params(
        mod.Session.request,
        ["method", "url", "headers", "timeout", "stream", "allow_redirects"],
    )


def test_session_context_manager_and_methods(depcheck):
    """`with requests.Session() as s: s.get(...)` is used; Session must be a
    context manager exposing the verb methods and a mountable adapter API."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.Session()
    try:
        assert hasattr(s, "__enter__") and hasattr(s, "__exit__")
        for m in ("get", "post", "put", "delete", "head", "request", "mount"):
            assert callable(getattr(s, m)), f"Session.{m} missing/not callable"
        assert hasattr(s, "headers")
    finally:
        s.close()


def test_response_contract(depcheck):
    """Response objects are consumed via .status_code/.headers/.text/.json()/
    .content/.raise_for_status()/.iter_content(); pin that shape offline."""
    mod = depcheck.load(IMPORT_NAME)
    # dir(instance) lists instance + class names WITHOUT executing descriptors,
    # so property getters like `ok` (which raise on a blank Response) aren't
    # triggered, while instance attrs set in __init__ (status_code) are seen.
    names = set(dir(mod.Response()))
    for attr in (
        "status_code",
        "headers",
        "text",
        "content",
        "json",
        "raise_for_status",
        "iter_content",
        "url",
        "ok",
    ):
        assert attr in names, f"Response.{attr} missing"
    assert callable(mod.Response.json)
    assert callable(mod.Response.raise_for_status)


def test_exception_hierarchy(depcheck):
    """Code catches requests.exceptions.{ConnectionError,Timeout,HTTPError};
    all must subclass RequestException so broad `except RequestException`
    handlers keep working."""
    mod = depcheck.load(IMPORT_NAME)
    base = mod.exceptions.RequestException
    for name in ("ConnectionError", "Timeout", "HTTPError"):
        exc = getattr(mod.exceptions, name)
        assert issubclass(exc, base), f"{name} no longer subclasses RequestException"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None
