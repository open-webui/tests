"""Regression: MCP servers whose registration answer leaves out the scope got tokens without one.

Issue open-webui/open-webui#29967, fix f7ce4024d (PR open-webui/open-webui#30384). Open WebUI
registers itself with an MCP server's authorization server (dynamic client registration) asking
for the scopes the server needs. RFC 7591 lets the answer leave the scope out, as Atlassian and
Notion do, and Open WebUI kept only what the answer said, so the user's sign-in asked for no
scope and every tool call was refused. The requested scope is now kept, and a connection
registered before the fix gets its scopes back from the server's protected-resource metadata
when the tool servers are saved, without registering again.

Every test here is red on dev 1c010b438: a tool written with the official MCP SDK that returns plain
text also sends it as structured data, and the model gets the text twice, the second time wrapped as
`{"result": ...}` (open-webui/open-webui#32126).

Discriminates: passes on dev efe63bd34, fails with f7ce4024d reverted (the sign-in asks for no
scope). Undoing only the recovery fails the recovered-connection test; undoing only the kept
registration scope leaves both green, as the recovery on save then fills it in.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.parse

import pytest
from cryptography.fernet import Fernet

from harness import upstream as reply
from harness.chat import ask
from harness.instance import WEBUI_SECRET_KEY
from harness.mcp_oauth import SCOPE, serving_protected_mcp
from harness.oidc_provider import browser_for
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
PHRASE = "the tide comes in twice a day"


@pytest.fixture
def mcp():
    with serving_protected_mcp() as server:
        yield server


@pytest.fixture
def person(make_user):
    return make_user()


def register(admin, preserve, mcp, person, edit_client_info=None) -> str:
    """Register the server as the admin panel does and save it; returns its id."""
    preserve(TOOL_SERVERS)
    server_id = f"scoped_mcp_{secrets.token_hex(4)}"
    with admin.client() as client:
        registered = client.post(
            "/api/v1/configs/oauth/clients/register",
            params={"type": "mcp"},
            json={"url": mcp.url, "client_id": server_id},
        )
        assert registered.status_code == 200, registered.text
        client_info = registered.json()["oauth_client_info"]
        if edit_client_info:
            client_info = edit_client_info(client_info)
        connection = {
            "url": mcp.url,
            "path": "",
            "type": "mcp",
            "auth_type": "oauth_2.1",
            "key": "",
            "config": {"enable": True, "access_grants": [read_grant(person.id)]},
            "info": {"id": server_id, "name": server_id, "oauth_client_info": client_info},
        }
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    return server_id


def without_scope(client_info: str) -> str:
    """The stored client info as a registration from before the fix left it: no scope."""
    key = base64.urlsafe_b64encode(hashlib.sha256(WEBUI_SECRET_KEY.encode()).digest())
    fernet = Fernet(key)
    decrypted = json.loads(fernet.decrypt(client_info.encode()))
    decrypted.pop("scope", None)
    return fernet.encrypt(json.dumps(decrypted).encode()).decode()


def connect(instance, actor, server_id: str) -> dict[str, str]:
    """Press "Connect" and walk the sign-in; returns what the authorize request asked for."""
    bearer = {"Authorization": f"Bearer {actor.token}"}
    with browser_for(instance) as browser:
        authorize = browser.get(f"/oauth/clients/mcp:{server_id}/authorize", headers=bearer)
        assert authorize.status_code == 302, authorize.text
        approved = browser.get(authorize.headers["location"])
        assert approved.status_code == 302, f"the authorization server refused: {approved.text}"
        browser.get(approved.headers["location"], headers=bearer)
    location = urllib.parse.urlsplit(authorize.headers["location"])
    return dict(urllib.parse.parse_qsl(location.query))


def echo(actor, upstream, server_id: str) -> str:
    """The tool result the model got back for one echo call through the MCP server."""
    upstream.queue(reply.tool_call(f"{server_id}_echo", {"text": PHRASE}), reply.text("done"))
    with actor.client() as client:
        ask(client, "echo it", tool_ids=[f"server:mcp:{server_id}"])
    results = [
        entry["content"]
        for entry in upstream.chat_requests()[-1]["messages"]
        if entry["role"] == "tool"
    ]
    return results[-1] if results else ""


def test_a_registration_answer_without_scope_still_signs_in_with_it(
    instance, admin, preserve, upstream, mcp, person
):
    mcp.auth_server.registration_names_scope = False
    server_id = register(admin, preserve, mcp, person)

    asked = connect(instance, person, server_id)

    assert asked.get("scope") == SCOPE, asked
    assert echo(person, upstream, server_id) == PHRASE


def test_a_connection_registered_without_scope_recovers_it(
    instance, admin, preserve, upstream, mcp, person
):
    server_id = register(admin, preserve, mcp, person, edit_client_info=without_scope)

    asked = connect(instance, person, server_id)

    assert asked.get("scope") == SCOPE, asked
    assert echo(person, upstream, server_id) == PHRASE


def test_a_registration_answer_naming_the_scope_is_unchanged(
    instance, admin, preserve, upstream, mcp, person
):
    server_id = register(admin, preserve, mcp, person)

    [registration] = mcp.auth_server.requests_to("/register")
    assert registration.json()["scope"] == SCOPE
    assert connect(instance, person, server_id).get("scope") == SCOPE
    assert echo(person, upstream, server_id) == PHRASE
