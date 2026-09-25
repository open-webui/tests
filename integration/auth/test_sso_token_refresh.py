"""Journey: an SSO sign-in's provider token, refreshed before Open WebUI forwards it.

After an OIDC sign-in Open WebUI keeps the provider's tokens as a server-side session and
forwards the access token to connections set to `system_oauth`. Within five minutes of its
expiry it first spends the refresh token at the provider's token endpoint and forwards the new
access token; the provider rotates the refresh token as it does. When the provider refuses the
refresh, the session is dropped and nothing stale is forwarded.

Discriminates: in a backend copy, returning the stored token without checking its expiry in the
SSO `get_oauth_token` fails the refresh and the refused-refresh tests (no refresh is tried).
"""

from __future__ import annotations

import pytest

from harness.listener import json_answer
from harness.oidc_provider import oauth_settings, shared_provider, sign_in, sso_env

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes
TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


@pytest.fixture
def tool_server(listener):
    listener.route("GET", "/openapi.json", json_answer(TOOL_SPEC))
    return listener


def sign_in_as_admin(sso, idp):
    """An SSO admin's browser, holding the session and the `oauth_session_id` cookie."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return result.browser


def forwarded_token(browser, tool_server) -> str | None:
    """Verify a `system_oauth` tool server; returns the bearer token it was sent."""
    browser.post(
        "/api/v1/configs/tool_servers/verify",
        json={
            "url": tool_server.base_url,
            "path": "openapi.json",
            "type": "openapi",
            "auth_type": "system_oauth",
            "key": "",
            "config": {},
        },
    )
    [fetch] = tool_server.requests_to("/openapi.json")
    authorization = fetch.headers.get("Authorization")
    return authorization.removeprefix("Bearer ") if authorization else None


def refresh_grants(idp) -> list[dict[str, str]]:
    forms = [entry.form for entry in idp.requests_to("/token")]
    return [form for form in forms if form.get("grant_type") == "refresh_token"]


def test_an_expiring_token_is_refreshed_and_the_new_one_forwarded(sso, idp, tool_server):
    idp.token_lifetime = EXPIRING_SOON
    browser = sign_in_as_admin(sso, idp)
    signed_in = idp.issued[-1]

    forwarded = forwarded_token(browser, tool_server)

    refreshes = refresh_grants(idp)
    assert refreshes, "the expiring token was forwarded without a refresh"
    assert refreshes[0]["refresh_token"] == signed_in["refresh_token"]
    refreshed = idp.issued[-1]
    assert refreshed["refresh_token"] != signed_in["refresh_token"], "the provider did not rotate"
    assert forwarded == refreshed["access_token"], "the expiring token was forwarded unrefreshed"


def test_a_token_far_from_expiry_is_forwarded_without_a_refresh(sso, idp, tool_server):
    browser = sign_in_as_admin(sso, idp)

    assert forwarded_token(browser, tool_server) == idp.issued[-1]["access_token"]
    assert refresh_grants(idp) == []


def test_a_refused_refresh_forwards_nothing_and_drops_the_session(sso, idp, tool_server):
    idp.token_lifetime = EXPIRING_SOON
    browser = sign_in_as_admin(sso, idp)
    idp.revoke_refresh_tokens()

    forwarded = forwarded_token(browser, tool_server)
    disconnected = browser.delete("/api/v1/auths/oauth/sessions/oidc")

    assert refresh_grants(idp), "the refresh was never tried"
    assert forwarded is None, "a token the provider would not refresh was forwarded"
    assert disconnected.status_code == 404, "the refused refresh left the SSO session behind"
