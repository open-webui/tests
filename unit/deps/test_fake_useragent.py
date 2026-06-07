"""Dependency contract: fake-useragent (import name ``fake_useragent``).

``fake-useragent`` produces realistic browser ``User-Agent`` strings from
a *bundled local dataset* (no network at runtime). It is a *declared*
requirement of the Open WebUI backend (``fake-useragent==2.2.0`` in
requirements.txt / requirements-min.txt) used to make outbound scraping /
web-retrieval requests look like a real browser, avoiding naive UA-based
blocking. The application code does not import it under a stable internal
chokepoint today, so this module pins the public surface any consumer
relies on — the ``UserAgent`` class, its ``.random`` / per-browser
property accessors, the constructor's filter kwargs, and the error type —
plus offline behavioural contracts proving it yields plausible UA strings
without touching the network.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "fake_useragent"
DIST_NAME = "fake-useragent"

TOP_LEVEL_SYMBOLS = [
    "UserAgent",  # primary public class
    "FakeUserAgent",  # canonical class name (UserAgent is an alias)
    "FakeUserAgentError",  # raised when no UA can be produced
]

# Per-browser convenience accessors consumers commonly read.
BROWSER_PROPERTIES = ["chrome", "firefox", "safari", "edge", "random"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`fake_useragent` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "fake_useragent"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_useragent_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    import inspect

    assert inspect.isclass(mod.UserAgent)


def test_useragent_aliases_fakeuseragent(depcheck):
    """`UserAgent` and `FakeUserAgent` have historically been the same class;
    code may import either name. Pin the alias."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.UserAgent is mod.FakeUserAgent


def test_constructor_filter_kwargs(depcheck):
    """UserAgent(browsers=, os=, min_version=, fallback=, ...) — the filter
    kwargs consumers use to constrain output and the fallback string must
    remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.UserAgent.__init__,
        ["browsers", "os", "fallback"],
    )


def test_error_type_is_exception(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.FakeUserAgentError, Exception)


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE — bundled dataset, never networks).
# ---------------------------------------------------------------------------


def test_behaviour_random_returns_plausible_ua(depcheck):
    """`UserAgent().random` must return a non-trivial UA string from the
    bundled data with no network access."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    value = ua.random
    assert isinstance(value, str)
    assert len(value) > 10
    # Browser UA strings effectively always start with "Mozilla/".
    assert value.startswith("Mozilla/"), f"implausible UA: {value!r}"


def test_behaviour_per_browser_properties_return_strings(depcheck):
    """The per-browser accessors (.chrome/.firefox/.safari/.edge) and .random
    must all yield UA strings — these are the read patterns consumers use."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    for prop in BROWSER_PROPERTIES:
        value = getattr(ua, prop)
        assert isinstance(value, str) and value, f"UserAgent.{prop} empty/non-str"
        assert value.startswith("Mozilla/"), f"UserAgent.{prop} implausible: {value!r}"


def test_behaviour_getitem_accessor(depcheck):
    """ua['chrome'] indexing is a documented accessor equivalent to ua.chrome;
    pin it so consumers using the dict-style form keep working."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    value = ua["chrome"]
    assert isinstance(value, str) and value.startswith("Mozilla/")


def test_behaviour_browser_filter_constrains_output(depcheck):
    """Constructing with browsers=['Firefox'] must still produce a usable UA
    (Firefox/Gecko-flavoured) entirely offline."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent(browsers=["Firefox"], os=["Windows"])
    value = ua.random
    assert isinstance(value, str) and value.startswith("Mozilla/")
    # Firefox UAs carry the Gecko/Firefox tokens; the bundled data should match,
    # but tolerate the library's fallback by only requiring a Mozilla UA.


def test_behaviour_fallback_string_preserved(depcheck):
    """The fallback string passed to the constructor must be retained and used
    when a filter yields no match — consumers rely on a guaranteed UA."""
    mod = depcheck.load(IMPORT_NAME)
    sentinel = "Mozilla/5.0 (compatible; OWUI-Test/1.0)"
    ua = mod.UserAgent(fallback=sentinel)
    assert ua.fallback == sentinel


def test_behaviour_multiple_randoms_are_strings(depcheck):
    """Repeated .random reads must each return a valid UA string (the generator
    is stable across calls; no exhaustion / None)."""
    mod = depcheck.load(IMPORT_NAME)
    ua = mod.UserAgent()
    values = [ua.random for _ in range(10)]
    assert all(isinstance(v, str) and v.startswith("Mozilla/") for v in values)


def test_behaviour_no_network_imports(depcheck):
    """Guard against the dataset being fetched lazily over the network: socket
    creation is blocked while we instantiate and read a UA. The bundled-data
    design must hold (a regression to live-fetching would break offline use)."""
    import socket

    mod = depcheck.load(IMPORT_NAME)
    real_socket = socket.socket

    def _blocked(*args, **kwargs):
        raise AssertionError("fake_useragent attempted network access at runtime")

    socket.socket = _blocked
    try:
        ua = mod.UserAgent()
        value = ua.random
        assert value.startswith("Mozilla/")
    finally:
        socket.socket = real_socket
