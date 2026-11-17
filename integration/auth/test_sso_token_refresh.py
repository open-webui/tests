"""Journey: an SSO sign-in's provider token, refreshed before Open WebUI forwards it.

After an OIDC sign-in Open WebUI keeps the provider's tokens as a server-side session and
forwards the access token to connections set to `system_oauth`. Within five minutes of its
expiry it first spends the refresh token at the provider's token endpoint and forwards the new
access token; the provider rotates the refresh token as it does. When the provider refuses the
refresh, the session is dropped and nothing stale is forwarded.

Regression #32820: a model connection set to `system_oauth` got no token when the same user
called the API with a bearer token and no browser cookie, while tool servers fall back to the
user's latest stored session.

Regression #34578: a refresh the provider answered with a 5xx or a 429, at the token endpoint or
for its discovery document, deleted the SSO session, so every tool using the user's provider
token got none until they signed in again. Such an answer now keeps the session and the next call
refreshes with the same refresh token, while a refused refresh still drops it.

Regression #34581: signing out kept the browser's stored session, and once a person reached
`OAUTH_MAX_SESSIONS_PER_USER` the next sign-in pruned the sessions created first, so a device
signed in early and still in use lost its session to signed-out ones. Signing out now drops the
browser's session when the person keeps another one for the provider (a person's only session
stays for automations and API keys), and the cap drops the sessions refreshed least recently.

Discriminates: in a backend copy, returning the stored token without checking its expiry in the
SSO `get_oauth_token` fails the refresh and the refused-refresh tests (no refresh is tried). The
API-call test passes with the #32820 fix and fails on dev ca8f451cc (no Authorization header).
The outage test (all three cases) is red on dev e1d272252 until the fix (branch
fix/sso-refresh-keeps-session-on-transient-error) merges: there nothing is forwarded once the
provider is back, as the session is gone. It passes on the fix branch.
The session-cap, refresh-order and sign-out tests pass with the #34581 fix and fail on dev
e1d272252 (the device still in use is the session pruned; the signed-out session stays stored).
The test that keeps a person's only session passes on both.
"""

from __future__ import annotations

import time

import pytest

from harness.listener import json_answer
from harness.oidc_provider import (
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)
from harness.second_provider import OPENAI_CONFIG, attach

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes
TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}
SSO_MODEL = "sso-protected-model"
MAX_SESSIONS = 10  # the OAUTH_MAX_SESSIONS_PER_USER default
COMPLETION = {
    "id": "sso",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}


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
    fetch = tool_server.requests_to("/openapi.json")[-1]
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


@pytest.mark.regression
@pytest.mark.parametrize(
    ("outage", "status"),
    [("refresh_outage", 503), ("refresh_outage", 429), ("discovery_outage", 503)],
    ids=["outage", "rate-limited", "discovery-outage"],
)
def test_a_provider_outage_during_the_refresh_keeps_the_session(
    sso, idp, tool_server, outage, status
):
    idp.token_lifetime = EXPIRING_SOON
    browser = sign_in_as_admin(sso, idp)
    signed_in = idp.issued[-1]
    setattr(idp, outage, status)

    during_outage = forwarded_token(browser, tool_server)
    setattr(idp, outage, None)
    after_outage = forwarded_token(browser, tool_server)

    assert refresh_grants(idp), "the refresh was never tried"
    assert during_outage is None, "a token was forwarded while the provider could not refresh it"
    assert after_outage == idp.issued[-1]["access_token"], (
        f"a {status} from the provider during the refresh deleted the user's SSO session, so "
        "nothing was forwarded once the provider was back (#34578)"
    )
    assert refresh_grants(idp)[-1]["refresh_token"] == signed_in["refresh_token"]


@pytest.mark.regression
def test_an_api_call_forwards_the_token_to_a_system_oauth_model_connection(
    sso, idp, listener, preserve
):
    preserve(OPENAI_CONFIG, on=sso)
    browser = sign_in_as_admin(sso, idp)
    access_token = idp.issued[-1]["access_token"]
    listener.route("POST", "/v1/chat/completions", json_answer(COMPLETION))
    attach(browser, listener, SSO_MODEL, auth_type="system_oauth")

    # the same user over the API: a bearer token, no `oauth_session_id` cookie
    with sso.client(token=browser.cookies.get("token")) as api:
        body = {
            "model": SSO_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        completed = api.post("/api/chat/completions", json=body)

    assert completed.status_code == 200, f"the API completion failed: {completed.text}"
    [sent] = listener.requests_to("/v1/chat/completions")
    assert sent.headers.get("Authorization") == f"Bearer {access_token}", (
        f"the connection got {sent.headers.get('Authorization')!r} instead of the user's token"
    )


def next_second() -> None:
    """Wait for the clock's next whole second, the resolution stored sessions keep time in."""
    second = int(time.time())
    while int(time.time()) == second:
        time.sleep(0.05)


def stored_session_ids(sso, account_id: str) -> set[str]:
    """The ids of the SSO sessions the admin sees stored for an account."""
    with sso.client() as admin:
        listed = admin.get(f"/api/v1/users/{account_id}/oauth/sessions")
    if listed.status_code == 400:  # the endpoint answers 400 when none are stored
        return set()
    assert listed.status_code == 200, listed.text
    return {session["id"] for session in listed.json()}


@pytest.mark.regression
def test_signing_in_and_out_elsewhere_keeps_the_first_devices_session(sso, idp):
    """Narrow (#34581), the issue's repro: sign-ins and sign-outs on another browser up to the
    cap leave the first device's session stored."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        first_device = sign_in(sso)
        next_second()
        for _ in range(MAX_SESSIONS):
            elsewhere = sign_in(sso)
            assert elsewhere.token, f"a sign-in failed: {elsewhere.error}"
            signed_out = elsewhere.browser.post("/api/v1/auths/signout")
            assert signed_out.status_code == 200, signed_out.text

    stored = stored_session_ids(sso, session_user(sso, first_device.token)["id"])
    assert first_device.browser.cookies.get("oauth_session_id") in stored, (
        "signing in and out elsewhere pruned the first device's session (#34581)"
    )
    assert len(stored) == 1, f"{len(stored)} sessions stored, the signed-out ones stayed"


@pytest.mark.regression
def test_the_session_cap_keeps_the_session_refreshed_last(sso, idp, tool_server):
    """Narrow (#34581): at the cap the sign-in prunes a session nobody refreshed, not the first
    device's, which refreshed its token after the others signed in."""
    idp.token_lifetime = EXPIRING_SOON
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        first_device = sign_in(sso)
        next_second()
        idle = [sign_in(sso) for _ in range(MAX_SESSIONS - 1)]
        next_second()
        assert forwarded_token(first_device.browser, tool_server), "the tool got no token"
        assert refresh_grants(idp), "using the first device did not refresh its session"
        newest = sign_in(sso)

    assert all(result.token for result in [first_device, *idle, newest]), "a sign-in failed"
    stored = stored_session_ids(sso, session_user(sso, newest.token)["id"])
    assert len(stored) == MAX_SESSIONS, f"{len(stored)} sessions stored, the cap is {MAX_SESSIONS}"
    assert first_device.browser.cookies.get("oauth_session_id") in stored, (
        "the cap pruned the session refreshed last, not one nobody refreshed (#34581)"
    )


@pytest.mark.regression
def test_signing_out_drops_the_session_when_another_one_is_kept(sso, idp):
    """Narrow (#34581): the signed-out browser's session goes, the other browser's stays."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        staying, leaving = sign_in(sso), sign_in(sso)
    assert staying.token and leaving.token, "a sign-in failed"
    account_id = session_user(sso, staying.token)["id"]
    leaving_id = leaving.browser.cookies.get("oauth_session_id")
    assert leaving_id in stored_session_ids(sso, account_id)

    signed_out = leaving.browser.post("/api/v1/auths/signout")

    assert signed_out.status_code == 200, signed_out.text
    stored = stored_session_ids(sso, account_id)
    assert leaving_id not in stored, "signing out kept the browser's stored session (#34581)"
    assert staying.browser.cookies.get("oauth_session_id") in stored, "the other browser lost its"


def test_signing_out_keeps_a_persons_only_session(sso, idp):
    """Nearby: a person's only session stays, as automations and API keys use it after sign-out."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        only = sign_in(sso)
    assert only.token, f"the sign-in failed: {only.error}"
    account_id = session_user(sso, only.token)["id"]
    only_id = only.browser.cookies.get("oauth_session_id")

    signed_out = only.browser.post("/api/v1/auths/signout")

    assert signed_out.status_code == 200, signed_out.text
    assert stored_session_ids(sso, account_id) == {only_id}, "the only session was dropped"
