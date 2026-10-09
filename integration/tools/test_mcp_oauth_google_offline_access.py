"""Regression: an MCP connection signed in through Google dropped once its first token expired.

PR open-webui/open-webui#31395 (a5176f4cd, issue #28319): Google only issues a refresh token when
the sign-in asks for offline access (`access_type=offline`), and only issues a new one on a fresh
consent (`prompt=consent`). Open WebUI's MCP OAuth sign-in asked for neither, so the connection to
a Google-hosted MCP server (Gmail, Drive, Calendar) had no refresh token, the refresh after the
one-hour access token failed and the connection was removed. The sign-in now asks for both when
the authorization server's sign-in page is Google's.

The harness authorization server plays Google: its metadata names Google's sign-in page as the
authorize endpoint (the test follows that redirect to the stand-in, as the browser would reach
Google), and a code grant carries a refresh token only after an authorize that asked for offline
access. Its access tokens live two seconds. Nearby: an authorization server that is not Google's
is sent neither parameter, and its expired token is still refreshed as before.

`test_the_tool_still_works_after_the_first_token_expired` is red on dev 1c010b438: a tool written
with the official MCP SDK that returns plain text also sends it as structured data, and the model
gets the text twice, the second time wrapped as `{"result": ...}` (open-webui/open-webui#32126).

Discriminates: passes on dev a5bc78300, fails with a5176f4cd reverted (the sign-in asks no offline
access, the stand-in issues no refresh token, and the tool call after expiry finds the connection
gone).
"""

from __future__ import annotations

import secrets
import time
import urllib.parse

import pytest

from harness.mcp_oauth import GOOGLE_ACCOUNTS, serving_protected_mcp
from harness.oidc_provider import browser_for
from harness.terminal_server import read_grant
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
PHRASE = "the tide turns at noon"
TOKEN_LIFETIME = 2


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture(params=["google", "other"])
def mcp(request):
    with serving_protected_mcp() as server:
        server.auth_server.access_token_lifetime = TOKEN_LIFETIME
        if request.param == "google":
            server.auth_server.authorize_base = GOOGLE_ACCOUNTS
            server.auth_server.offline_access_only = True
        yield server


@pytest.fixture
def server_id(admin, preserve, mcp, person) -> str:
    """The MCP server, registered by the admin as the admin panel does and readable by `person`."""
    preserve(TOOL_SERVERS)
    server_id = f"google_mcp_{secrets.token_hex(4)}"
    with admin.client() as client:
        registered = client.post(
            "/api/v1/configs/oauth/clients/register",
            params={"type": "mcp"},
            json={"url": mcp.url, "client_id": server_id},
        )
        assert registered.status_code == 200, registered.text
        connection = {
            "url": mcp.url,
            "path": "",
            "type": "mcp",
            "auth_type": "oauth_2.1",
            "key": "",
            "config": {"enable": True, "access_grants": [read_grant(person.id)]},
            "info": {
                "id": server_id,
                "name": server_id,
                "oauth_client_info": registered.json()["oauth_client_info"],
            },
        }
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    return server_id


def connect(instance, actor, mcp, server_id: str) -> dict[str, str]:
    """Press "Connect" and sign in; returns what the sign-in page was asked for."""
    headers = {"Authorization": f"Bearer {actor.token}"}
    with browser_for(instance) as browser:
        authorize = browser.get(f"/oauth/clients/mcp:{server_id}/authorize", headers=headers)
        assert authorize.status_code == 302, authorize.text
        sign_in_page = authorize.headers["location"]
        approved = browser.get(mcp.auth_server.reached(sign_in_page))
        assert approved.status_code == 302, f"the authorization server refused: {approved.text}"
        callback = browser.get(approved.headers["location"], headers=headers)
    landing = urllib.parse.urlsplit(callback.headers["location"])
    assert "error" not in dict(urllib.parse.parse_qsl(landing.query)), landing.query
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(sign_in_page).query))


def wait_until_expired(mcp) -> None:
    expires_at = max(token["expires_at"] for token in mcp.auth_server.access_tokens.values())
    time.sleep(max(0.0, expires_at - time.time()) + 0.5)


def echo_through_chat(actor, upstream, server_id: str) -> str:
    with actor.client() as client:
        return run_tool(
            client,
            upstream,
            f"{server_id}_echo",
            {"text": PHRASE},
            tool_ids=[f"server:mcp:{server_id}"],
        )


@pytest.mark.parametrize("mcp", ["google"], indirect=True)
def test_the_google_sign_in_asks_for_offline_access_and_a_fresh_consent(
    instance, mcp, person, server_id
):
    asked = connect(instance, person, mcp, server_id)

    assert asked.get("access_type") == "offline", (
        f"the Google sign-in did not ask for offline access (#28319): {asked}"
    )
    assert asked.get("prompt") == "consent", f"the Google sign-in asked no fresh consent: {asked}"


@pytest.mark.parametrize("mcp", ["other"], indirect=True)
def test_another_authorization_server_is_not_sent_googles_parameters(
    instance, mcp, person, server_id
):
    asked = connect(instance, person, mcp, server_id)

    assert "access_type" not in asked, asked
    assert "prompt" not in asked, asked


def test_the_tool_still_works_after_the_first_token_expired(
    instance, upstream, mcp, person, server_id
):
    connect(instance, person, mcp, server_id)
    [connected] = mcp.auth_server.issued
    wait_until_expired(mcp)

    assert echo_through_chat(person, upstream, server_id) == PHRASE, (
        "the connection did not outlive its first access token (#28319)"
    )

    [refresh] = mcp.auth_server.token_grants("refresh_token")
    assert refresh["refresh_token"] == connected["refresh_token"]
    assert mcp.presented[-1] == mcp.auth_server.issued[-1]["access_token"]
