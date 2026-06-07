"""Dependency contract: google-auth-httplib2 (import ``google_auth_httplib2``).

google-auth-httplib2 is a **pinned direct requirement** of the Open WebUI
backend (``google-auth-httplib2==0.4.0``) but is *not* imported anywhere in the
``open_webui`` package source directly. It is the adapter that lets the
``google-api-python-client`` stack authenticate ``httplib2`` HTTP calls with
``google.auth`` credentials — the transport layer underneath Open WebUI's
Google Drive integration (file picker / loader). The two public objects are:

  * ``AuthorizedHttp(credentials, http=None, ...)`` — an httplib2-compatible
    ``Http`` wrapper that injects/refreshes the bearer token; the API client
    calls ``.request(...)`` on it.
  * ``Request(http)`` — a ``google.auth.transport.Request`` implementation that
    credential refresh uses to perform the token-endpoint round-trip.

Because there is no in-tree call site and any real ``.request(...)`` would hit
the network, the meaningful contract is the *module surface + offline
construction*: both classes exist, ``AuthorizedHttp`` constructs around a
credentials object and exposes httplib2's ``request`` method, and ``Request`` is
a genuine ``google.auth.transport.Request`` (so credential refresh accepts it).
A bump that removed/renamed either class, changed ``AuthorizedHttp``'s
constructor shape, or broke the transport-Request inheritance fails loudly here
instead of breaking Google Drive auth at request time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks +
offline construction with a stand-in credentials object (NO network, NO real
HTTP). Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "google_auth_httplib2"
DIST_NAME = "google-auth-httplib2"

# The two public classes the google-api-python-client stack uses.
USED_SYMBOLS = ["AuthorizedHttp", "Request"]


class _StubCredentials:
    """Minimal stand-in for a google.auth Credentials object — enough to
    construct AuthorizedHttp without real Google credentials or any network.
    AuthorizedHttp only stores it and calls these on an actual request, which
    this suite never issues."""

    expired = False
    valid = True

    def before_request(self, request, method, url, headers):  # pragma: no cover - not called
        pass

    def refresh(self, request):  # pragma: no cover - not called
        pass


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "google_auth_httplib2"


def test_version_reported(depcheck):
    """The installed distribution version (PyPI ``google-auth-httplib2``) must be
    resolvable so bump tooling and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """``AuthorizedHttp`` and ``Request`` — both public classes must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_classes_are_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.AuthorizedHttp)
    assert callable(mod.Request)


# --------------------------------------------------------------------------- #
# AuthorizedHttp — constructor shape + httplib2 request surface
# --------------------------------------------------------------------------- #
def test_authorized_http_constructor_signature(depcheck):
    """The API client builds ``AuthorizedHttp(credentials, http=...)`` — first
    positional is the credentials, with ``http`` / ``refresh_status_codes`` /
    ``max_refresh_attempts`` as the tunables. Pin those parameter names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.AuthorizedHttp.__init__,
        ["credentials", "http", "refresh_status_codes", "max_refresh_attempts"],
    )


def test_authorized_http_constructs_with_credentials(depcheck):
    """``AuthorizedHttp(credentials)`` must construct around a credentials object
    without opening any connection (no http passed -> it builds a default
    httplib2.Http lazily; no request is issued)."""
    mod = depcheck.load(IMPORT_NAME)
    authed = mod.AuthorizedHttp(_StubCredentials())
    assert authed is not None


def test_authorized_http_exposes_httplib2_request(depcheck):
    """The API client drives the wrapper via ``.request(...)`` (the httplib2
    Http interface). Pin that a constructed AuthorizedHttp exposes a callable
    ``request`` (we never call it — that would hit the network)."""
    mod = depcheck.load(IMPORT_NAME)
    authed = mod.AuthorizedHttp(_StubCredentials())
    assert callable(getattr(authed, "request", None)), "AuthorizedHttp.request missing"


def test_authorized_http_has_close(depcheck):
    """httplib2 consumers may ``.close()`` the Http; pin AuthorizedHttp forwards
    a close method (resource cleanup)."""
    mod = depcheck.load(IMPORT_NAME)
    authed = mod.AuthorizedHttp(_StubCredentials())
    assert callable(getattr(authed, "close", None)), "AuthorizedHttp.close missing"


# --------------------------------------------------------------------------- #
# Request — must be a google.auth.transport.Request (refresh accepts it)
# --------------------------------------------------------------------------- #
def test_request_constructs_with_http(depcheck):
    """``Request(http)`` wraps an httplib2.Http for credential refresh. Construct
    it around a real (but unused) httplib2.Http — no request is performed."""
    mod = depcheck.load(IMPORT_NAME)
    httplib2 = depcheck.load("httplib2")
    req = mod.Request(httplib2.Http())
    assert req is not None
    # The transport Request is itself callable (called as request(url, ...)).
    assert callable(req)


def test_request_is_transport_request_subclass(depcheck):
    """``Request`` must subclass ``google.auth.transport.Request`` — that is the
    contract credential ``.refresh(request)`` relies on when refreshing the
    Google Drive access token. A break here would make token refresh reject the
    request object."""
    mod = depcheck.load(IMPORT_NAME)
    transport = depcheck.load("google.auth.transport")
    assert issubclass(mod.Request, transport.Request), (
        "google_auth_httplib2.Request no longer subclasses google.auth.transport.Request"
    )


def test_request_first_param_is_http(depcheck):
    """``Request(http)`` — the single constructor parameter is the httplib2 Http
    instance. Pin the parameter name."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Request.__init__, ["http"])
