"""Dependency contract: google-auth-oauthlib (import ``google_auth_oauthlib``).

google-auth-oauthlib provides the OAuth 2.0 *authorization-flow* helpers
that sit on top of ``google-auth`` (it bridges ``requests-oauthlib`` and
google credentials). It's pinned in ``backend/requirements.txt``
(``google-auth-oauthlib==1.4.0``) alongside the rest of the Google API
client stack (``google-api-python-client``, ``google-auth-httplib2``) that
backs Open WebUI's Google integrations (e.g. Google Drive document
picking / OAuth).

IMPORTANT — usage note: the Open WebUI *application* code does NOT import
``google_auth_oauthlib`` directly anywhere (its Google OAuth login flow
uses authlib; the google-api stack is what consumes this package). It is a
declared dependency providing the 3-legged OAuth ``Flow`` used when
exchanging an authorization code for Google API credentials. There are no
first-party call sites, so this module pins the package's *core public
surface* — the ``flow.Flow`` / ``flow.InstalledAppFlow`` classes and the
``from_client_config`` / ``authorization_url`` / ``fetch_token`` /
``credentials`` API — and exercises the OFFLINE part of the flow (building
a Flow from an in-memory client config and generating the authorization
URL, which is pure local URL construction; we never call ``fetch_token``,
which would hit Google's token endpoint).

No network: only the authorization-URL builder runs, which contacts
nothing.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "google_auth_oauthlib"
DIST_NAME = "google-auth-oauthlib"

# Top-level package surface.
PACKAGE_SYMBOLS = [
    "flow",  # the OAuth Flow submodule
    "get_user_credentials",  # convenience end-to-end helper
    "helpers",
]

# Methods/attributes the Flow object exposes (the 3-legged OAuth contract).
FLOW_MEMBERS = [
    "from_client_config",
    "from_client_secrets_file",
    "authorization_url",
    "fetch_token",
    "credentials",
]

# A minimal, fake OAuth client config (no real secrets, never sent anywhere).
_FAKE_CLIENT_CONFIG = {
    "web": {
        "client_id": "fake-client-id.apps.googleusercontent.com",
        "client_secret": "fake-client-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["https://example.com/oauth2/callback"],
    }
}
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_REDIRECT = "https://example.com/oauth2/callback"


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "google_auth_oauthlib"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_package_symbols_exist(depcheck):
    """flow / get_user_credentials / helpers must remain on the package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, PACKAGE_SYMBOLS)


def test_flow_classes_importable(depcheck):
    """`from google_auth_oauthlib.flow import Flow, InstalledAppFlow` must work,
    and InstalledAppFlow must remain a Flow subclass (it specialises the local
    redirect handling)."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    assert hasattr(flow_mod, "Flow"), "google_auth_oauthlib.flow.Flow missing"
    assert hasattr(flow_mod, "InstalledAppFlow"), "InstalledAppFlow missing"
    assert inspect.isclass(flow_mod.Flow)
    assert issubclass(flow_mod.InstalledAppFlow, flow_mod.Flow)


def test_flow_has_core_members(depcheck):
    """The 3-legged OAuth surface (from_client_config / authorization_url /
    fetch_token / credentials) must remain on Flow."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    names = set(dir(flow_mod.Flow))
    missing = [m for m in FLOW_MEMBERS if m not in names]
    assert not missing, f"google_auth_oauthlib Flow missing member(s): {missing}"


def test_from_client_config_signature(depcheck):
    """Flow.from_client_config(client_config, scopes, **kwargs) is the primary
    constructor (no secrets file needed). Pin client_config + scopes."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    depcheck.assert_params(
        flow_mod.Flow.from_client_config,
        ["client_config", "scopes"],
    )


def test_build_flow_and_authorization_url_offline(depcheck):
    """End-to-end offline slice: build a Flow from an in-memory client config and
    generate the authorization URL. This is pure local URL construction (no
    token exchange / network). The URL must point at Google's auth endpoint and
    embed our client_id + scope, proving the flow wired the config through."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    flow = flow_mod.Flow.from_client_config(
        _FAKE_CLIENT_CONFIG,
        scopes=_SCOPES,
        redirect_uri=_REDIRECT,
    )
    result = flow.authorization_url(access_type="offline", prompt="consent")
    # authorization_url returns (url, state).
    assert isinstance(result, tuple) and len(result) == 2
    url, state = result
    assert url.startswith("https://accounts.google.com/o/oauth2/auth")
    assert "fake-client-id" in url
    assert "drive.readonly" in url  # scope is URL-encoded but the path survives
    assert state, "authorization_url did not return a CSRF state token"


def test_authorization_url_accepts_oauth_kwargs(depcheck):
    """Callers pass access_type=/prompt=/state= etc. through authorization_url.
    Those flow into the query string; pin that the call accepts arbitrary OAuth
    kwargs (it forwards **kwargs to the underlying oauth session)."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    flow = flow_mod.Flow.from_client_config(
        _FAKE_CLIENT_CONFIG,
        scopes=_SCOPES,
        redirect_uri=_REDIRECT,
    )
    # Passing a custom state must be honoured in the produced URL.
    url, state = flow.authorization_url(state="my-csrf-token", access_type="offline")
    assert state == "my-csrf-token"
    assert "my-csrf-token" in url


def test_installed_app_flow_has_run_local_server(depcheck):
    """InstalledAppFlow adds run_local_server (the desktop loopback flow). Pin
    the method exists (we never call it — it would spin up a local server and
    open a browser)."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    assert hasattr(flow_mod.InstalledAppFlow, "run_local_server")
    assert callable(flow_mod.InstalledAppFlow.run_local_server)


def test_fetch_token_present_but_not_called(depcheck):
    """fetch_token is the code->token exchange (network). Pin it's a callable on
    the flow, but NEVER invoke it here — that would contact Google's token
    endpoint. This documents the offline boundary of this suite."""
    depcheck.load(IMPORT_NAME)
    flow_mod = depcheck.load("google_auth_oauthlib.flow")
    flow = flow_mod.Flow.from_client_config(
        _FAKE_CLIENT_CONFIG,
        scopes=_SCOPES,
        redirect_uri=_REDIRECT,
    )
    assert callable(flow.fetch_token)


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): the backend's own OAuth login uses
    authlib; google-auth-oauthlib is the 3-legged flow helper for the
    google-api-python-client stack. The offline flow pins above guard the slice
    that stack depends on."""
    assert True
