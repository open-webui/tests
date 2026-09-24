"""The terminal proxy forwards credentials Open WebUI resolved itself, never ones the caller sent.

Before 0.11.0 a `system_oauth` terminal connection took its token from the caller's
`x-oauth-access-token` request header and forwarded it to the terminal server as the Bearer
token, so any signed-in user could present a token that was never issued to them. Fix
`3a9b9a1a7` (PR #26719) resolves the token server-side from the caller's own OAuth session
and ignores the header. Each test proxies one request through a fake terminal server and
reads what it received. Twin of unit/security/test_terminal_sso_token_provenance.py.

Discriminates: passes on dev bbfa876af, fails with 3a9b9a1a7 reverted (the forged header
reaches the terminal server as `Authorization: Bearer forged-by-the-caller`).
"""

from __future__ import annotations

import pytest

from harness.listener import json_answer
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


def test_session_and_content_type_headers_still_pass_through(proxy_as, make_user):
    caller = make_user()
    sent = {"x-session-id": "terminal-session-42", "content-type": "application/json"}
    received = proxy_as(caller, headers=sent)
    assert received.headers["x-session-id"] == "terminal-session-42"
    assert received.headers["content-type"] == "application/json"


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
