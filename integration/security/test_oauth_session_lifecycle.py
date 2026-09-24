"""Regression: the OAuth sign-in lifecycle, from the switch that turns it off to the MCP flows.

- `ENABLE_OAUTH` (#26988, commit 71f8b6d5b): the admin panel's OAuth switch has to gate the
  sign-in itself, not just the button: while it is off `/oauth/{provider}/login` and the
  callback answer 404 and `/api/config` lists no providers.
- Key rotation (#27310, issue #26407, commit 4f823774a): the callback evicts the cached JWKS
  when the ID token fails its signature check, but caught authlib's `BadSignatureError` while
  authlib 1.7 raises joserfc's. After the provider rotated its key under the same `kid`, every
  sign-in failed against the stale keys until a restart.
- MCP OAuth authorize (#26654, issue #26647, commit 3fff80ad2): a client registered against an
  authorization server whose metadata has no authorize endpoint made authlib raise
  `RuntimeError('Missing "authorize_url" value')`, which escaped as a 500. It is now a 400 that
  says to re-register the server.
- MCP OAuth callback (c2107e5bb): the callback bound the new connection to whoever returned
  with the code instead of the account that started the flow. The initiator is now stamped into
  the state, and a different signed-in account returning to the callback gets nothing.

Twin of unit/security/test_oauth_session_lifecycle.py (and of the client-callback test in
unit/security/test_oauth_identity.py).

Discriminates: passes on dev bbfa876af; in a copy with the matching fix reverted, login and
callback redirect to the provider while OAuth is off, the sign-in after a key rotation keeps
failing, the MCP authorize answers 500 and the callback files the connection under the account
that returned to it.
"""

from __future__ import annotations

import secrets
import urllib.parse

import httpx
import pytest

from harness.listener import json_answer
from harness.oidc_provider import (
    browser_for,
    oauth_settings,
    shared_provider,
    sign_in,
    sso_env,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def error_of(response: httpx.Response) -> str | None:
    """The `error` a redirect hands the page it sends the browser to."""
    query = urllib.parse.urlsplit(response.headers["location"]).query
    return dict(urllib.parse.parse_qsl(query)).get("error")


# ── the ENABLE_OAUTH switch ────────────────────────────────────────────────


def test_with_oauth_turned_off_sign_in_never_reaches_the_provider(sso, idp):
    """Narrow: login and both callback paths answer 404 and the provider hears nothing."""
    idp.sign_in_as()
    with oauth_settings(sso, ENABLE_OAUTH=False), browser_for(sso) as browser:
        answers = [
            browser.get("/oauth/oidc/login"),
            browser.get("/oauth/oidc/callback", params={"code": "c", "state": "s"}),
            browser.get("/oauth/oidc/login/callback", params={"code": "c", "state": "s"}),
        ]
        providers = browser.get("/api/config").json()["oauth"]["providers"]
    assert [answer.status_code for answer in answers] == [404, 404, 404]
    assert idp.requests == [], "sign-in reached the provider while OAuth was off"
    assert providers == {}, "the sign-in page still offers the provider"


def test_with_oauth_on_sign_in_goes_to_the_provider_and_unknown_ones_404(sso, idp):
    """Nearby: the switch on redirects to the provider; an unconfigured provider still 404s."""
    with browser_for(sso) as browser:
        login = browser.get("/oauth/oidc/login")
        unknown = browser.get("/oauth/not-configured/login")
        providers = browser.get("/api/config").json()["oauth"]["providers"]
    assert providers == {"oidc": "SSO"}
    assert login.status_code == 302
    assert login.headers["location"].startswith(f"{idp.base_url}/authorize")
    assert unknown.status_code == 404


# ── signing key rotation (#27310) ──────────────────────────────────────────


def test_sign_in_recovers_after_the_provider_rotates_its_key(sso, idp):
    """Narrow: a rotation under the same `kid` drops the stale keys, so signing in works again."""
    idp.sign_in_as()
    assert sign_in(sso).token, "the sign-in before the rotation failed"
    idp.rotate_key()

    attempts = [sign_in(sso) for _ in range(2)]

    assert attempts[-1].token, f"still failing after the rotation: {attempts[-1].error}"


def test_a_failed_code_exchange_is_not_retried_and_keeps_the_keys(sso, idp):
    """Nearby: an ordinary token error fails once, and the cached keys stay cached."""
    idp.sign_in_as()
    assert sign_in(sso).token
    key_fetches = len(idp.requests_to("/jwks"))

    with browser_for(sso) as browser:
        start = browser.get("/oauth/oidc/login")
        approved = browser.get(start.headers["location"])
        callback = browser.get(approved.headers["location"].replace("code=", "code=spoiled-"))

    assert error_of(callback)
    assert len(idp.requests_to("/token")) == 2, "a failed code exchange was retried"
    assert sign_in(sso).token
    assert len(idp.requests_to("/jwks")) == key_fetches, "a token error evicted the keys"


# ── MCP OAuth: authorize and callback ──────────────────────────────────────


def connect_mcp_server(admin, server_url: str) -> str:
    """Add an MCP server with static OAuth credentials the way the admin panel does."""
    server_id = f"mcp-{secrets.token_hex(4)}"
    registered = admin.post(
        "/api/v1/configs/oauth/clients/register",
        params={"type": "mcp"},
        json={"url": server_url, "client_id": server_id, "client_secret": "mcp-secret"},
    )
    assert registered.status_code == 200, registered.text
    connection = {
        "url": server_url,
        "path": "",
        "type": "mcp",
        "auth_type": "oauth_2.1_static",
        "key": "",
        "config": {"enable": True},
        "info": {
            "id": server_id,
            "name": server_id,
            "oauth_client_id": "mcp-client",
            "oauth_client_secret": "mcp-secret",
            "oauth_client_info": registered.json()["oauth_client_info"],
        },
    }
    saved = admin.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    return f"mcp:{server_id}"


def has_connection(actor, client_key: str) -> bool:
    """Whether `actor` holds a stored OAuth connection, read by disconnecting it."""
    with actor.client() as client:
        return client.delete(f"/api/v1/auths/oauth/sessions/{client_key}").status_code == 200


def test_an_mcp_server_without_an_authorize_endpoint_gets_a_400(admin, preserve, listener):
    """Narrow (#26654): an unresolvable authorize endpoint is a clear 400, never a 500."""
    preserve(TOOL_SERVERS)
    metadata = {"issuer": listener.base_url, "token_endpoint": f"{listener.base_url}/token"}
    listener.route("GET", "/.well-known/oauth-authorization-server", json_answer(metadata))
    with admin.client() as client:
        client_key = connect_mcp_server(client, f"{listener.base_url}/mcp")
        authorize = client.get(f"/oauth/clients/{client_key}/authorize", follow_redirects=False)

    assert authorize.status_code == 400, authorize.text
    assert "Re-register" in authorize.json()["detail"]


def test_an_unknown_mcp_client_is_404(user):
    """Nearby: the unknown-client answer is not swallowed by the new handler."""
    with user.client() as client:
        answer = client.get("/oauth/clients/mcp:never-added/authorize", follow_redirects=False)
    assert answer.status_code == 404


def bearer(actor) -> dict[str, str]:
    return {"Authorization": f"Bearer {actor.token}"}


def mcp_callback_url(browser: httpx.Client, client_key: str, initiator) -> str:
    """`initiator` starts the authorization; returns where the provider sends the browser back."""
    authorize = browser.get(f"/oauth/clients/{client_key}/authorize", headers=bearer(initiator))
    assert authorize.status_code == 302, authorize.text
    approved = browser.get(authorize.headers["location"])
    return approved.headers["location"]


def test_the_mcp_connection_lands_on_the_account_that_started_it(
    instance, admin, preserve, make_user, idp
):
    """Nearby: the initiator coming back finds the connection filed under their account."""
    preserve(TOOL_SERVERS)
    with admin.client() as client:
        client_key = connect_mcp_server(client, f"{idp.base_url}/mcp")
    person = make_user()

    with browser_for(instance) as browser:
        callback = browser.get(
            mcp_callback_url(browser, client_key, person), headers=bearer(person)
        )

    assert error_of(callback) is None
    assert has_connection(person, client_key)


def test_an_mcp_callback_returned_by_another_account_stores_nothing(
    instance, admin, preserve, make_user, idp
):
    """Narrow (c2107e5bb): the code is never filed under whoever happened to come back, and
    the refused state cannot be replayed afterwards."""
    preserve(TOOL_SERVERS)
    with admin.client() as client:
        client_key = connect_mcp_server(client, f"{idp.base_url}/mcp")
    initiator, other = make_user(), make_user()

    with browser_for(instance) as browser:
        callback_url = mcp_callback_url(browser, client_key, initiator)
        refused = browser.get(callback_url, headers=bearer(other))
        replayed = browser.get(callback_url, headers=bearer(initiator))

    assert error_of(refused), "the callback accepted a different account"
    assert error_of(replayed), "the refused state stayed replayable"
    assert not has_connection(other, client_key), "the connection was filed under the returner"
    assert not has_connection(initiator, client_key)
