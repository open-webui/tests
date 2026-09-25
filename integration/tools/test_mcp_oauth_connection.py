"""Journey: an MCP tool server behind OAuth 2.1, from registration to a refreshed tool call.

The admin registers the server as the admin panel does: Open WebUI follows the server's 401 to
its protected-resource metadata and the authorization server's metadata, then registers itself
as a client. A user connects through `/oauth/clients/{id}/authorize` and the callback with a
PKCE S256 code, and the model's tool call reaches the server with that user's token. A token
about to expire is refreshed first, spending and rotating the refresh token; once the refresh
token is revoked the connection is dropped and no tool is offered.

Discriminates: in a backend copy, never adding the S256 code challenge fails every test at the
connect (the authorization server refuses the authorize), and returning the stored token without
checking its expiry fails the refresh and the revoked-token tests (no refresh is tried).
"""

from __future__ import annotations

import secrets
import urllib.parse

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_oauth import SCOPE, serving_protected_mcp
from harness.oidc_provider import browser_for
from harness.terminal_server import read_grant
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
PHRASE = "herons wait in the shallows"
EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes


@pytest.fixture
def mcp():
    with serving_protected_mcp() as server:
        yield server


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def server_id(admin, preserve, mcp, person) -> str:
    """The MCP server, registered by the admin and readable by `person`."""
    preserve(TOOL_SERVERS)
    server_id = f"oauth_mcp_{secrets.token_hex(4)}"
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


def bearer(actor) -> dict[str, str]:
    return {"Authorization": f"Bearer {actor.token}"}


def connect(instance, actor, server_id: str) -> None:
    """Press "Connect" on the tool server: authorize, the provider approves, the callback."""
    with browser_for(instance) as browser:
        authorize = browser.get(f"/oauth/clients/mcp:{server_id}/authorize", headers=bearer(actor))
        assert authorize.status_code == 302, authorize.text
        approved = browser.get(authorize.headers["location"])
        assert approved.status_code == 302, f"the authorization server refused: {approved.text}"
        callback = browser.get(approved.headers["location"], headers=bearer(actor))
    landing = urllib.parse.urlsplit(callback.headers["location"])
    assert "error" not in dict(urllib.parse.parse_qsl(landing.query)), landing.query


def echo_through_chat(actor, upstream, server_id: str) -> str:
    with actor.client() as client:
        return run_tool(
            client,
            upstream,
            f"{server_id}_echo",
            {"text": PHRASE},
            tool_ids=[f"server:mcp:{server_id}"],
        )


def test_a_user_connects_the_server_and_the_tool_call_carries_their_token(
    instance, upstream, mcp, person, server_id
):
    auth_server = mcp.auth_server
    [registration] = auth_server.requests_to("/register")
    callback_url = f"{instance.base_url}/oauth/clients/mcp:{server_id}/callback"
    assert registration.json()["redirect_uris"] == [callback_url]
    assert registration.json()["scope"] == SCOPE, "the resource's scope was not asked for"

    connect(instance, person, server_id)

    [exchange] = auth_server.token_grants("authorization_code")
    assert exchange["code_verifier"], "the code was exchanged without a PKCE verifier"
    assert echo_through_chat(person, upstream, server_id) == PHRASE
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"]
    assert auth_server.token_grants("refresh_token") == [], "a fresh token was refreshed"


def test_an_expiring_token_is_refreshed_before_the_tool_call(
    instance, upstream, mcp, person, server_id
):
    auth_server = mcp.auth_server
    auth_server.access_token_lifetime = EXPIRING_SOON
    connect(instance, person, server_id)
    connected = auth_server.issued[-1]

    assert echo_through_chat(person, upstream, server_id) == PHRASE

    refreshes = auth_server.token_grants("refresh_token")
    assert refreshes, "the expiring token was used without a refresh"
    assert refreshes[0]["refresh_token"] == connected["refresh_token"]
    assert connected["access_token"] not in mcp.presented, "the MCP server saw the old token"
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"]


def test_a_revoked_refresh_token_drops_the_connection(instance, upstream, mcp, person, server_id):
    auth_server = mcp.auth_server
    auth_server.access_token_lifetime = EXPIRING_SOON
    connect(instance, person, server_id)
    issued_before = len(auth_server.issued)
    auth_server.revoke_refresh_tokens()

    upstream.queue(reply.text("no tools today"))
    with person.client() as client:
        ask(client, "echo it", tool_ids=[f"server:mcp:{server_id}"])
        disconnected = client.delete(f"/api/v1/auths/oauth/sessions/mcp:{server_id}")

    assert auth_server.token_grants("refresh_token"), "the refresh was never tried"
    assert len(auth_server.issued) == issued_before, "a revoked refresh token got new tokens"
    offered = upstream.chat_requests()[-1].get("tools") or []
    assert not [tool for tool in offered if server_id in str(tool)], "the tool was still offered"
    assert disconnected.status_code == 404, "the failed refresh left the session behind"
