"""The terminal proxy forwards credentials Open WebUI resolved itself, never ones the caller sent.

Before 0.11.0 a `system_oauth` terminal connection took its token from the caller's
`x-oauth-access-token` request header and forwarded it to the terminal server as the Bearer
token, so any signed-in user could present a token that was never issued to them. Fix
`3a9b9a1a7` (PR #26719) resolves the token server-side from the caller's own OAuth session
and ignores the header. Each test proxies one request through a fake terminal server and
reads what it received. The positive half runs on an instance that signs in through a fake
OpenID Connect provider, so the caller holds a real OAuth session whose access token the
terminal must get.

Since the terminal rework on dev (6f2fc7bc7) the proxy reads `X-Session-Id` as the chat the
request belongs to, as the page sends it, and answers 403 "Chat access denied" for a saved chat
the caller may not write to. The pass-through test therefore sends a chat of the caller's own,
and a second test pins that someone else's chat is refused before anything reaches the terminal.

Discriminates: passes on dev ef67cc3fa, fails with 3a9b9a1a7 reverted (the forged header
reaches the terminal server as `Authorization: Bearer forged-by-the-caller`, and the SSO
caller's own session token never does). The two session header tests pass on dev c15f28649; in
a backend copy whose proxy drops `x-session-id` from the headers it forwards the pass-through
test fails (the terminal gets no session header), and with the chat check removed from
`find_terminal_connection` the request naming someone else's chat reaches the terminal.
"""

from __future__ import annotations

import pytest

from harness.listener import json_answer
from harness.oidc_provider import oauth_settings, shared_provider, sign_in, sso_env
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    TerminalRequest,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FORGED = "forged-by-the-caller"
CONNECTION_KEY = "configured-terminal-key"
FORGED_IDENTITY_HEADERS = {
    "x-oauth-access-token": FORGED,
    "x-user-id": FORGED,
    "x-forwarded-user": FORGED,
}


@pytest.fixture
def terminal():
    with serving_terminal() as server:
        server.route("GET", "/probe", json_answer({"ok": True}))
        yield server


@pytest.fixture
def proxy_as(admin, preserve, terminal):
    """`proxy_as(caller, headers=..., **connection)` returns what the terminal server received."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def proxied(caller, headers: dict | None = None, **connection_fields) -> TerminalRequest:
        grants = {"config": {"access_grants": [read_grant(caller.id)]}}
        connection = terminal.connection(key=CONNECTION_KEY, **grants, **connection_fields)
        with admin.client() as client:
            configure_terminals(client, connection)
        terminal.clear()
        with caller.client() as client:
            response = client.get(f"/api/v1/terminals/{connection['id']}/probe", headers=headers)
        assert response.status_code == 200, f"the proxy refused: {response.text}"
        [received] = terminal.received
        return received

    return proxied


def test_a_forged_oauth_header_never_becomes_the_terminal_token(proxy_as, make_user):
    caller = make_user()
    received = proxy_as(caller, headers={"x-oauth-access-token": FORGED}, auth_type="system_oauth")
    assert "authorization" not in received.headers, (
        "a caller-supplied x-oauth-access-token reached the terminal server as its credential, "
        "so any user could act there with a token never issued to them (#26719)"
    )


@pytest.mark.parametrize("auth_type", ["bearer", "session", "system_oauth", "none"])
def test_no_forwarded_header_comes_from_the_caller(proxy_as, make_user, auth_type):
    caller = make_user()
    received = proxy_as(caller, headers=FORGED_IDENTITY_HEADERS, auth_type=auth_type)
    echoed = sorted(name for name, value in received.headers.items() if FORGED in value)
    assert echoed == [], f"{auth_type}: {echoed} carried a value the caller chose (#26719)"
    assert received.headers["x-user-id"] == caller.id


@pytest.mark.parametrize("auth_type", ["bearer", "session", "none", "system_oauth"])
def test_each_auth_type_sends_its_own_credential(proxy_as, make_user, auth_type):
    caller = make_user()
    expected = {
        "bearer": f"Bearer {CONNECTION_KEY}",
        "session": f"Bearer {caller.token}",
        "none": None,
        "system_oauth": None,  # a password sign-in has no OAuth session to take a token from
    }[auth_type]
    received = proxy_as(caller, auth_type=auth_type)
    assert received.headers.get("authorization") == expected


def _saved_chat(owner) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": "Terminal chat"}})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_session_and_content_type_headers_still_pass_through(proxy_as, make_user):
    caller = make_user()
    chat_id = _saved_chat(caller)
    sent = {"x-session-id": chat_id, "content-type": "application/json"}
    received = proxy_as(caller, headers=sent)
    assert received.headers.get("x-session-id") == chat_id, "the chat id never reached the terminal"
    assert received.headers.get("content-type") == "application/json"


def test_a_session_header_naming_someone_elses_chat_is_refused(
    admin, preserve, make_user, terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    caller, someone_else = make_user(), make_user()
    connection = terminal.connection(config={"access_grants": [read_grant(caller.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    terminal.clear()
    probe = f"/api/v1/terminals/{connection['id']}/probe"

    with caller.client() as client:
        refused = client.get(probe, headers={"x-session-id": _saved_chat(someone_else)})
        own = client.get(probe, headers={"x-session-id": _saved_chat(caller)})

    assert (refused.status_code, refused.json()) == (403, {"detail": "Chat access denied"}), (
        f"a request naming someone else's chat answered {refused.status_code} {refused.text}"
    )
    assert own.status_code == 200, own.text
    assert len(terminal.requests_to("/probe")) == 1, (
        "the request naming someone else's chat reached the terminal"
    )


@pytest.mark.parametrize("forward_cookies", [True, False])
def test_cookies_reach_the_terminal_only_when_the_connection_opts_in(
    proxy_as, make_user, forward_cookies
):
    caller = make_user()
    cookie_header = {"cookie": "oauth_session_id=caller-session; theme=dark"}
    received = proxy_as(
        caller, headers=cookie_header, auth_type="system_oauth", forward_cookies=forward_cookies
    )
    expected = {"oauth_session_id": "caller-session", "theme": "dark"} if forward_cookies else {}
    assert received.cookies == expected


# A caller signed in through single sign-on, whose own session token the terminal gets.


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


@pytest.mark.slow
def test_the_callers_own_oauth_session_token_reaches_the_terminal(sso, idp, preserve, terminal):
    preserve(TERMINAL_SERVERS_CONFIG, on=sso)
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        signed_in = sign_in(sso)
    assert signed_in.token, f"the sign-in failed: {signed_in.error}"
    connection = terminal.connection(auth_type="system_oauth")
    with sso.client() as admin:
        configure_terminals(admin, connection)
    terminal.clear()

    response = signed_in.browser.get(
        f"/api/v1/terminals/{connection['id']}/probe", headers={"x-oauth-access-token": FORGED}
    )

    assert response.status_code == 200, response.text
    [received] = terminal.received
    session_token = idp.issued[-1]["access_token"]
    assert received.headers.get("authorization") == f"Bearer {session_token}", (
        "single sign-on to the terminal broke: the token of the caller's own OAuth session "
        "never reached it (#26719)"
    )
