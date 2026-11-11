"""Journey: an MCP tool server behind OAuth 2.1, from registration to a refreshed tool call.

The admin registers the server as the admin panel does: Open WebUI follows the server's 401 to
its protected-resource metadata and the authorization server's metadata, then registers itself
as a client. A user connects through `/oauth/clients/{id}/authorize` and the callback with a
PKCE S256 code, and the model's tool call reaches the server with that user's token. A token
about to expire is refreshed first, spending and rotating the refresh token; once the refresh
token is revoked the connection is dropped and no tool is offered. Registering the server again
replaces its client: authlib keeps every client it built by name, so saving the connection has to
drop the old one from authlib's registry, or the next connect still signs in as the old client
(twin of the registry case in unit/deps/test_authlib.py).

`test_a_user_connects_the_server_and_the_tool_call_carries_their_token`,
`test_an_expiring_token_is_refreshed_before_the_tool_call` and
`test_registering_the_server_again_connects_with_the_new_client` are red on dev 1c010b438: a tool
written with the official MCP SDK that returns plain text also sends it as structured data, and the
model gets the text twice, the second time wrapped as `{"result": ...}`
(open-webui/open-webui#32126, fixed by ab15284a4).

Discriminates: in a backend copy, never adding the S256 code challenge fails every test at the
connect (the authorization server refuses the authorize), and returning the stored token without
checking its expiry fails the refresh and the revoked-token tests (no refresh is tried). On dev
ef67cc3fa, `remove_client` leaving authlib's `_clients` entry in place fails the registration
test (the connect uses the first client).

The bot challenge test pins a fixed bug (PR open-webui/open-webui#32364, fix 4f9f04cee, issue
#32363): before sending the browser to sign in, Open WebUI fetches the authorize URL itself to
catch a stale client. A provider behind Cloudflare answers that fetch with a bot challenge page
that repeats the URL, redirect URL included, so Open WebUI took it for a rejected client,
registered again and failed with "OAuth client registration is still invalid after
re-registration". A challenge now skips the check. Nearby: a provider that no longer knows the
client still gets a new registration and the connect goes through with it. The challenge test
passes on dev ad8da5921 and fails on a backend copy with 4f9f04cee reverted (the authorize answers
500 after a second registration).

The outage test pins a fixed bug (PR open-webui/open-webui#33805, fix 14c2cad84, issue #33803):
a refresh the provider answered with a 5xx or a 429 deleted the user's stored sign-in, so one
provider blip made them connect again, and for a provider that rotates refresh tokens the session
was the only copy. Such an answer now keeps the session and the next call refreshes with the same
refresh token, while a refused refresh (the revoked-token test) still drops it. Passes on dev
f659c0d12, fails on a backend copy with 14c2cad84 reverted (the session is gone after the 503 and
after the 429).

The quiet listing test pins PR open-webui/open-webui#33663 (fix 8957ac139): every time a user
loaded Open WebUI, each OAuth MCP server they had not connected yet logged a warning. Passes on
dev e8c0ad85e, fails on a backend copy with 8957ac139 reverted (the warning is in the log).

The large headers test pins issue open-webui/open-webui#34078: an authorization server that sends
a header line over 8190 bytes (monday.com sends an 11 KB Content-Security-Policy on every OAuth
endpoint) failed the connection check and the registration with "Got more than 8190 bytes", and
the refresh the same way. Red on dev c15f28649 until the fix (fix branch
fix/mcp-oauth-large-headers) merges: there the connection check answers 400. Passes on the fix
branch, and fails on a backend copy of it with only the refresh left at the default limit (the
tool is gone after the failed refresh).
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


def register(admin, mcp, person, server_id: str) -> None:
    """Register the server as the admin panel does and save the connection to it."""
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


@pytest.fixture
def server_id(admin, preserve, mcp, person) -> str:
    """The MCP server, registered by the admin and readable by `person`."""
    preserve(TOOL_SERVERS)
    server_id = f"oauth_mcp_{secrets.token_hex(4)}"
    register(admin, mcp, person, server_id)
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


def stored_sessions(admin, person) -> list[str]:
    """The providers of the OAuth sessions Open WebUI keeps for `person`."""
    with admin.client() as client:
        listed = client.get(f"/api/v1/users/{person.id}/oauth/sessions")
    if listed.status_code == 400:  # the route's answer for a user without sessions
        return []
    assert listed.status_code == 200, listed.text
    return [session["provider"] for session in listed.json()]


@pytest.mark.regression
@pytest.mark.parametrize("status", [503, 429], ids=["outage", "rate-limited"])
def test_a_provider_outage_during_the_refresh_keeps_the_connection(
    instance, admin, upstream, mcp, person, server_id, status
):
    auth_server = mcp.auth_server
    auth_server.access_token_lifetime = EXPIRING_SOON
    connect(instance, person, server_id)
    connected = auth_server.issued[-1]
    auth_server.refresh_outage = status

    prompt = f"echo it during a {status}"
    upstream.queue(reply.text("the server is busy", match=reply.answering(prompt)))
    with person.client() as client:
        ask(client, prompt, tool_ids=[f"server:mcp:{server_id}"])

    assert auth_server.token_grants("refresh_token"), "the refresh was never tried"
    assert f"mcp:{server_id}" in stored_sessions(admin, person), (
        f"a {status} from the provider during the refresh deleted the user's MCP sign-in, so "
        "they have to connect again (#33803)"
    )
    auth_server.refresh_outage = None
    assert echo_through_chat(person, upstream, server_id) == PHRASE
    refreshes = auth_server.token_grants("refresh_token")
    assert refreshes[-1]["refresh_token"] == connected["refresh_token"]
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"]


def test_registering_the_server_again_connects_with_the_new_client(
    instance, admin, upstream, mcp, person, server_id
):
    earlier_clients = set(mcp.auth_server.clients)
    register(admin, mcp, person, server_id)
    [replacement] = set(mcp.auth_server.clients) - earlier_clients

    connect(instance, person, server_id)

    authorize = mcp.auth_server.requests_to("/authorize")[-1]
    assert authorize.query["client_id"] == replacement, (
        "the connection still signs in with the client it was first registered as"
    )
    assert echo_through_chat(person, upstream, server_id) == PHRASE


def preflights(auth_server) -> list:
    """The authorize requests Open WebUI sent itself before sending the browser there."""
    requests = auth_server.requests_to("/authorize")
    return [entry for entry in requests if "aiohttp" in entry.headers.get("User-Agent", "")]


def listed_server(actor, server_id: str) -> dict:
    with actor.client() as client:
        listed = client.get("/api/v1/tools/")
    assert listed.status_code == 200, listed.text
    [entry] = [tool for tool in listed.json() if tool["id"] == f"server:mcp:{server_id}"]
    return entry


@pytest.mark.regression
def test_listing_a_server_the_user_has_not_connected_logs_no_warning(
    instance, mcp, person, server_id
):
    log_offset = instance.log_size()

    assert listed_server(person, server_id)["authenticated"] is False

    assert "No OAuth session found" not in instance.log_since(log_offset), (
        "loading the tools of a user who has not connected the MCP server yet logged a warning, "
        "so every page load fills the log (#33663)"
    )
    connect(instance, person, server_id)
    assert listed_server(person, server_id)["authenticated"] is True


@pytest.mark.regression
def test_a_bot_challenge_on_the_sign_in_page_still_connects(instance, mcp, person, server_id):
    auth_server = mcp.auth_server
    auth_server.bot_challenge = True

    connect(instance, person, server_id)

    assert preflights(auth_server), "Open WebUI never checked the sign-in page itself"
    assert len(auth_server.requests_to("/register")) == 1, (
        "the challenge page was read as a rejected client and the server registered again"
    )


@pytest.mark.regression
def test_a_forgotten_client_registers_again_and_connects(instance, mcp, person, server_id):
    auth_server = mcp.auth_server
    auth_server.forget_clients()

    connect(instance, person, server_id)

    assert len(auth_server.requests_to("/register")) == 2, "the unknown client was not replaced"
    [replacement] = auth_server.clients
    authorize = auth_server.requests_to("/authorize")[-1]
    assert authorize.query["client_id"] == replacement


@pytest.mark.regression
def test_an_authorization_server_with_large_headers_registers_connects_and_refreshes(
    instance, admin, preserve, upstream, mcp, person
):
    auth_server = mcp.auth_server
    auth_server.large_headers = True
    auth_server.access_token_lifetime = EXPIRING_SOON
    preserve(TOOL_SERVERS)
    server_id = f"oauth_mcp_{secrets.token_hex(4)}"

    with admin.client() as client:
        verified = client.post(
            "/api/v1/configs/tool_servers/verify",
            json={
                "url": mcp.url,
                "path": "",
                "type": "mcp",
                "auth_type": "oauth_2.1",
                "key": "",
                "config": {},
            },
        )
    assert verified.status_code == 200, (
        f"the connection check failed on a large response header (#34078): {verified.text}"
    )
    register(admin, mcp, person, server_id)
    connect(instance, person, server_id)
    connected = auth_server.issued[-1]

    assert echo_through_chat(person, upstream, server_id) == PHRASE
    refreshes = auth_server.token_grants("refresh_token")
    assert refreshes and refreshes[0]["refresh_token"] == connected["refresh_token"]
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"], (
        "the refresh failed on a large response header (#34078)"
    )
