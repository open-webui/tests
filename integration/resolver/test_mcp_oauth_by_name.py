"""Journey: an MCP tool server behind OAuth 2.1 reached by host name, under both resolvers.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` (off by default since c5ec01b1f, PR #28242, after c-ares broke
name resolution in #28013 and #28215) decides which resolver every aiohttp connector Open WebUI
opens uses. The MCP OAuth client makes six aiohttp calls of its own: the anonymous `initialize`
post and the protected-resource metadata fetch that follow the server's 401, the authorization
server's metadata fetch, dynamic client registration, the static-credentials variant of that
discovery, the preflight `GET` of the authorize URL and, for an expiring token, the token post
of the refresh. Since a0e606bfa the refresh reuses the authorization server's metadata authlib
loaded for the connect and keeps until a restart, so a token endpoint that moved is only read
after one. Each test serves the MCP server and its authorization server by a host name
(`localhost`, a hosts-file name for `::1` alone and a hosts-file name for another local address)
and expects the same outcome under both resolvers: the server registered with the scope its
resource names, the user connected and the tool call carrying the token, an expiring token
refreshed and the new one presented. A server or a token endpoint whose name does not resolve
fails the same way under both: a refused registration, and a dropped connection with no tool
offered.

The connect itself (authlib's code exchange) and the tool calls (the MCP SDK) run over httpx and
are not affected by the flag.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271).
"""

from __future__ import annotations

import secrets
import urllib.parse

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.host_names import (
    FAILS_WITHIN,
    UNRESOLVABLE,
    name_forms,
    timed,
    without_resolver_reason,
)
from harness.host_names import name_form as form_named
from harness.mcp_oauth import SCOPE, ProtectedMcp, serving_protected_mcp
from harness.mcp_server import TOOL_SERVERS
from harness.oidc_provider import browser_for
from harness.terminal_server import read_grant
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PHRASE = "herons wait in the shallows"
EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes
REGISTER = "/api/v1/configs/oauth/clients/register"
VERIFY = "/api/v1/configs/tool_servers/verify"


def _serving(label: str):
    form = form_named(label)
    return serving_protected_mcp(host=form.address, name=form.host)


def _aiohttp_requests(entries) -> list:
    return [entry for entry in entries if "aiohttp" in entry.headers.get("User-Agent", "")]


def _register(client, url: str, server_id: str, **credentials):
    return client.post(
        REGISTER, params={"type": "mcp"}, json={"url": url, "client_id": server_id, **credentials}
    )


def _save_connection(client, mcp: ProtectedMcp, person, server_id: str, registered: dict) -> None:
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
            "oauth_client_info": registered["oauth_client_info"],
        },
    }
    saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def registered(resolving_instance, resolving_admin, preserve):
    """`registered(mcp)` registers the server as the admin panel does; returns its id and a user."""
    preserve(TOOL_SERVERS, on=resolving_instance)

    def register(mcp: ProtectedMcp):
        person = create_user(resolving_instance)
        server_id = f"oauth_mcp_{secrets.token_hex(4)}"
        with resolving_admin.client() as client:
            answer = _register(client, mcp.url, server_id)
            assert answer.status_code == 200, answer.text
            _save_connection(client, mcp, person, server_id, answer.json())
        return server_id, person

    return register


def _connect(instance, actor, server_id: str) -> None:
    """Press "Connect" on the tool server: authorize, the provider approves, the callback."""
    headers = {"Authorization": f"Bearer {actor.token}"}
    with browser_for(instance) as browser:
        authorize = browser.get(f"/oauth/clients/mcp:{server_id}/authorize", headers=headers)
        assert authorize.status_code == 302, authorize.text
        approved = browser.get(authorize.headers["location"])
        assert approved.status_code == 302, f"the authorization server refused: {approved.text}"
        callback = browser.get(approved.headers["location"], headers=headers)
    landing = urllib.parse.urlsplit(callback.headers["location"])
    assert "error" not in dict(urllib.parse.parse_qsl(landing.query)), landing.query


def _echo_through_chat(actor, upstream, server_id: str) -> str:
    with actor.client() as client:
        return run_tool(
            client,
            upstream,
            f"{server_id}_echo",
            {"text": PHRASE},
            tool_ids=[f"server:mcp:{server_id}"],
        )


@pytest.mark.parametrize("name_form", name_forms())
def test_a_server_named_by_host_is_registered_and_connected(
    resolver, name_form, resolving_instance, registered
):
    with _serving(name_form) as mcp:
        server_id, person = registered(mcp)
        host = urllib.parse.urlsplit(mcp.url).netloc.rsplit(":", 1)[0]
        _connect(resolving_instance, person, server_id)
        echoed = _echo_through_chat(person, resolving_instance.upstream, server_id)

        auth_server = mcp.auth_server
        [registration] = auth_server.requests_to("/register")
        discoveries = auth_server.requests_to("/.well-known/oauth-authorization-server")
        authorizations = auth_server.requests_to("/authorize")

    assert registration.json()["scope"] == SCOPE, "the resource's scope was not asked for"
    assert registration.headers["Host"].startswith(f"{host}:")
    assert {entry.headers["Host"].split(":")[0] for entry in discoveries} == {host}
    assert _aiohttp_requests([registration]) == [registration]
    assert _aiohttp_requests(discoveries), "the metadata was never fetched by Open WebUI itself"
    assert _aiohttp_requests(authorizations), (
        "the authorize URL was never checked before connecting"
    )
    assert echoed == PHRASE
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"]


@pytest.mark.parametrize("name_form", name_forms())
def test_an_expiring_token_is_refreshed_at_a_server_named_by_host(
    resolver, name_form, resolving_instance, registered
):
    with _serving(name_form) as mcp:
        auth_server = mcp.auth_server
        auth_server.access_token_lifetime = EXPIRING_SOON
        server_id, person = registered(mcp)
        _connect(resolving_instance, person, server_id)
        connected = auth_server.issued[-1]

        echoed = _echo_through_chat(person, resolving_instance.upstream, server_id)

        [refresh] = [
            entry
            for entry in auth_server.requests_to("/token")
            if entry.form.get("grant_type") == "refresh_token"
        ]

    assert echoed == PHRASE
    assert refresh.form["refresh_token"] == connected["refresh_token"]
    assert "aiohttp" in refresh.headers["User-Agent"]
    assert connected["access_token"] not in mcp.presented, "the MCP server saw the old token"
    assert mcp.presented[-1] == auth_server.issued[-1]["access_token"]


@pytest.mark.parametrize("name_form", name_forms())
def test_static_credentials_discover_a_server_named_by_host(
    resolver, name_form, resolving_admin, preserve, resolving_instance
):
    preserve(TOOL_SERVERS, on=resolving_instance)
    with _serving(name_form) as mcp, resolving_admin.client() as client:
        answer = _register(client, mcp.url, "static_mcp", client_secret="static-secret")

        discoveries = mcp.auth_server.requests_to("/.well-known/oauth-authorization-server")
        registrations = mcp.auth_server.requests_to("/register")

    assert answer.status_code == 200, answer.text
    assert registrations == [], "static credentials registered a client anyway"
    assert _aiohttp_requests(discoveries), "the metadata was never fetched by Open WebUI itself"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_token_endpoint_that_does_not_resolve_drops_the_connection(
    resolver, resolving_instance, registered
):
    with _serving("localhost") as mcp:
        auth_server = mcp.auth_server
        auth_server.access_token_lifetime = EXPIRING_SOON
        server_id, person = registered(mcp)
        _connect(resolving_instance, person, server_id)
        auth_server.token_base = f"http://{UNRESOLVABLE}:1"
        resolving_instance.restart()

        resolving_instance.upstream.queue(reply.text("no tools today"))
        with person.client() as client:
            _, seconds = timed(ask, client, "echo it", tool_ids=[f"server:mcp:{server_id}"])
            disconnected = client.delete(f"/api/v1/auths/oauth/sessions/mcp:{server_id}")

    offered = resolving_instance.upstream.chat_requests()[-1].get("tools") or []
    assert not [tool for tool in offered if server_id in str(tool)], "the tool was still offered"
    assert disconnected.status_code == 404, "the failed refresh left the session behind"
    assert seconds < FAILS_WITHIN, f"the refresh took {seconds:.1f}s to fail"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_server_that_does_not_resolve_fails_registration_the_same_way(resolver, resolving_admin):
    with resolving_admin.client() as client:
        dynamic, dynamic_seconds = timed(
            _register, client, f"http://{UNRESOLVABLE}:8000/mcp", "gone_dynamic"
        )
        static, static_seconds = timed(
            _register,
            client,
            f"http://{UNRESOLVABLE}:8000/mcp",
            "gone_static",
            client_secret="static-secret",
        )

    assert (dynamic.status_code, static.status_code) == (400, 400), (dynamic.text, static.text)
    expected = (
        f"Failed to register OAuth client: Cannot connect to host {UNRESOLVABLE}:8000 "
        "ssl:default [<resolver reason>]"
    )
    assert without_resolver_reason(dynamic.json()["detail"]) == expected
    assert without_resolver_reason(static.json()["detail"]) == expected
    assert max(dynamic_seconds, static_seconds) < FAILS_WITHIN, "the lookup failed only slowly"


def _unsaved_connection(url: str) -> dict:
    return {
        "url": url,
        "path": "",
        "type": "mcp",
        "auth_type": "oauth_2.1",
        "key": "",
        "config": {"enable": True, "access_grants": []},
        "info": {"id": "verified", "name": "verified"},
    }


@pytest.mark.parametrize("name_form", name_forms())
def test_a_server_named_by_host_is_verified_with_its_authorization_metadata(
    resolver, name_form, resolving_admin
):
    with _serving(name_form) as mcp, resolving_admin.client() as client:
        verified = client.post(VERIFY, json=_unsaved_connection(mcp.url))

    assert verified.status_code == 200, verified.text
    assert verified.json()["status"] is True
    metadata = verified.json()["oauth_server_metadata"]
    assert metadata["token_endpoint"] == f"{mcp.auth_server.base_url}/token"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_server_that_does_not_resolve_fails_verification_the_same_way(resolver, resolving_admin):
    with resolving_admin.client() as client:
        verified, seconds = timed(
            client.post, VERIFY, json=_unsaved_connection(f"http://{UNRESOLVABLE}:8000/mcp")
        )

    assert (verified.status_code, verified.json()) == (
        400,
        {"detail": "Failed to connect to the tool server"},
    )
    assert seconds < FAILS_WITHIN, f"{resolver} took {seconds:.1f}s to refuse an unknown name"
