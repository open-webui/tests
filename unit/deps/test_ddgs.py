"""Dependency contract: ddgs (import name ``ddgs``).

ddgs (the maintained successor to ``duckduckgo_search``) backs Open WebUI's
DuckDuckGo web-search loader, ``retrieval/web/duckduckgo.py``:

    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
    ...
    with DDGS(proxy=proxy) as ddgs:
        if concurrent_requests:
            ddgs.threads = concurrent_requests
        try:
            kwargs = {'safesearch': 'moderate', 'max_results': count}
            if backend and backend != 'auto':
                kwargs['backend'] = backend
            results = ddgs.text(query, **kwargs)
            ...
        except RatelimitException as e:
            ...

So the contract is: ``DDGS`` is a context manager constructed with a ``proxy``
keyword; on the entered object the loader sets a ``threads`` attribute and calls
``text(query, safesearch=, max_results=, backend=)``, iterating the returned
result dicts for ``href`` / ``title`` / ``body`` keys; and
``ddgs.exceptions.RatelimitException`` is the catchable rate-limit error.

A real ``text()`` call performs a LIVE web search, which this offline suite must
NOT do. So this module pins the *surface only*: the ``DDGS`` class + context-
manager protocol, that it constructs with the ``proxy`` keyword and accepts a
``threads`` attribute, that ``text`` exists with a ``(query, **kwargs)``
signature, and the ``RatelimitException`` hierarchy — all without issuing a
search. A ddgs bump that removed/renamed ``DDGS`` / ``text`` / the exception, or
dropped the context-manager protocol or the proxy keyword, fails loudly here
instead of breaking web search at query time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks +
offline construction (NO network, NO real search). Uses the `depcheck` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "ddgs"
DIST_NAME = "ddgs"


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "ddgs"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_ddgs_class_exists_and_callable(depcheck):
    """``from ddgs import DDGS`` then ``DDGS(...)`` — pin the class is present
    and callable."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "DDGS")
    assert callable(mod.DDGS)


def test_exceptions_submodule_importable(depcheck):
    """``from ddgs.exceptions import RatelimitException`` — the exceptions
    submodule must import cleanly."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.try_load("ddgs.exceptions")
    assert exc_mod is not None, "ddgs.exceptions no longer importable"


# --------------------------------------------------------------------------- #
# DDGS.text — the search method surface (NO real search issued)
# --------------------------------------------------------------------------- #
def test_text_method_exists_and_callable(depcheck):
    """The loader calls ``ddgs.text(query, ...)``. Pin that ``text`` exists on
    the class and is callable (we never invoke it — that hits the network)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(getattr(mod.DDGS, "text", None)), "DDGS.text missing/not callable"


def test_text_signature_accepts_query_and_kwargs(depcheck):
    """``text(query, **kwargs)`` — the loader passes the query positionally and
    safesearch=/max_results=/backend= as keywords. Pin that the first parameter
    is ``query`` and the method accepts **kwargs (so those keywords are valid).
    """
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.DDGS.text)
    params = sig.parameters
    # First non-self parameter must be the query.
    non_self = [p for p in params if p != "self"]
    assert non_self and non_self[0] == "query", f"DDGS.text first param changed: {list(params)}"
    # **kwargs must be accepted so safesearch/max_results/backend pass through.
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        f"DDGS.text no longer accepts **kwargs: {sig}"
    )


# --------------------------------------------------------------------------- #
# DDGS construction + context-manager protocol (offline — no search)
# --------------------------------------------------------------------------- #
def test_ddgs_is_context_manager(depcheck):
    """The loader uses ``with DDGS(proxy=proxy) as ddgs:`` — DDGS must implement
    the context-manager protocol (__enter__/__exit__)."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod.DDGS, "__enter__"), "DDGS missing __enter__"
    assert hasattr(mod.DDGS, "__exit__"), "DDGS missing __exit__"


def test_ddgs_constructs_with_proxy_kwarg(depcheck):
    """``DDGS(proxy=proxy)`` — construction with the ``proxy`` keyword must not
    raise. (The constructor signature is ``(*args, **kwargs)`` so the keyword is
    verified behaviourally, not via introspection.) No search is performed, so
    no network contact."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.DDGS(proxy=None)
    assert client is not None


def test_ddgs_context_manager_enter_exit_offline(depcheck):
    """Enter and exit the ``with DDGS(...)`` block without calling text() — the
    entered object must expose the ``text`` method. Constructing/entering must
    not touch the network (a search is only issued by text())."""
    mod = depcheck.load(IMPORT_NAME)
    with mod.DDGS(proxy=None) as ddgs:
        assert callable(getattr(ddgs, "text", None)), "entered DDGS has no text()"


def test_ddgs_threads_attribute_settable(depcheck):
    """The loader sets ``ddgs.threads = concurrent_requests``. Pin that the
    attribute accepts assignment and round-trips the value (offline)."""
    mod = depcheck.load(IMPORT_NAME)
    with mod.DDGS(proxy=None) as ddgs:
        ddgs.threads = 5
        assert ddgs.threads == 5


# --------------------------------------------------------------------------- #
# ddgs.exceptions.RatelimitException — the catchable rate-limit error
# --------------------------------------------------------------------------- #
def test_ratelimit_exception_exists(depcheck):
    """The loader catches ``RatelimitException``; it must exist in
    ddgs.exceptions."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert hasattr(exc_mod, "RatelimitException"), "ddgs.exceptions.RatelimitException missing"


def test_ratelimit_exception_is_exception_subclass(depcheck):
    """RatelimitException must subclass Exception so the loader's
    ``except RatelimitException`` handler is valid and a broad ``except
    Exception`` would also catch it."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert issubclass(exc_mod.RatelimitException, Exception)


def test_ddgs_exception_hierarchy(depcheck):
    """ddgs groups its errors under a base ``DDGSException``; RatelimitException
    (and TimeoutException) subclass it. Pin that base + the rate-limit member so
    a future move of the rate-limit type stays catchable via the base too."""
    depcheck.load(IMPORT_NAME)
    exc_mod = depcheck.load("ddgs.exceptions")
    assert hasattr(exc_mod, "DDGSException"), "ddgs.exceptions.DDGSException missing"
    base = exc_mod.DDGSException
    assert issubclass(exc_mod.RatelimitException, base), (
        "RatelimitException no longer subclasses DDGSException"
    )
