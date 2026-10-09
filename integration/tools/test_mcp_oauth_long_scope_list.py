"""Regression: an MCP OAuth sign-in for a server asking for many scopes failed on its state.

Issue open-webui/open-webui#26382, fix cd64930c0 (PR open-webui/open-webui#31894). Starting the
sign-in saved the whole authorization URL with the state in the session cookie. A server that asks
for dozens of scopes (Google Workspace's MCP server asks for 42) made that URL so long that the
cookie went past the 4096 bytes a browser keeps, the browser dropped it, and the callback answered
"OAuth callback state is invalid or expired". The URL now stays out of the cookie.

The MCP server here names 42 Google Workspace scopes in its protected-resource metadata. The
browser stand-in drops a cookie over 4096 bytes the way Chromium does, and the sign-in then has to
land connected with the tool call carrying the user's token. Nearby: one scope signs in as before.

`test_a_sign_in_asking_for_many_scopes_keeps_the_session_cookie_small` and
`test_a_sign_in_asking_for_one_scope_is_unchanged` are red on dev 1c010b438: a tool written with the
official MCP SDK that returns plain text also sends it as structured data, and the model gets the
text twice, the second time wrapped as `{"result": ...}` (open-webui/open-webui#32126).

Discriminates: passes on dev b859124f9, fails with cd64930c0 reverted (the session cookie is over
4096 bytes, the browser stand-in drops it and the callback is refused over its state).
"""

from __future__ import annotations

import secrets
import urllib.parse

import httpx
import pytest

from harness.mcp_oauth import (
    GOOGLE_WORKSPACE_SCOPES,
    SCOPE,
    register_oauth_mcp,
    serving_protected_mcp,
)
from harness.mcp_server import TOOL_SERVERS
from harness.oidc_provider import browser_for
from harness.terminal_server import read_grant
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PHRASE = "the kettle sings at dawn"
COOKIE_LIMIT = 4096  # bytes of name and value a browser keeps in one cookie


def serve_and_register(admin, preserve, person, scopes: list[str] | None):
    """A protected MCP server asking for `scopes`, registered by the admin; yields it and its id."""
    preserve(TOOL_SERVERS)
    server_id = f"wide_mcp_{secrets.token_hex(4)}"
    with serving_protected_mcp(scopes=scopes) as mcp:
        register_oauth_mcp(admin, mcp, server_id, [read_grant(person.id)])
        yield mcp, server_id


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def wide_server(admin, preserve, person):
    yield from serve_and_register(admin, preserve, person, GOOGLE_WORKSPACE_SCOPES)


@pytest.fixture
def narrow_server(admin, preserve, person):
    yield from serve_and_register(admin, preserve, person, None)


def set_cookie_sizes(response: httpx.Response) -> dict[str, int]:
    """Bytes of name and value in each cookie a response sets."""
    sizes = {}
    for header in response.headers.get_list("set-cookie"):
        name, _, rest = header.partition("=")
        sizes[name] = len(name) + 1 + len(rest.split(";")[0])
    return sizes


def sign_in(instance, actor, server_id: str) -> tuple[dict[str, int], str]:
    """Press "Connect" in a browser that drops oversize cookies; (cookie sizes, callback error)."""
    bearer = {"Authorization": f"Bearer {actor.token}"}
    with browser_for(instance) as browser:
        authorize = browser.get(f"/oauth/clients/mcp:{server_id}/authorize", headers=bearer)
        assert authorize.status_code == 302, authorize.text
        sizes = set_cookie_sizes(authorize)
        for name, size in sizes.items():
            if size > COOKIE_LIMIT:
                browser.cookies.delete(name)
        approved = browser.get(authorize.headers["location"])
        assert approved.status_code == 302, f"the authorization server refused: {approved.text}"
        callback = browser.get(approved.headers["location"], headers=bearer)
    landing = urllib.parse.urlsplit(callback.headers["location"])
    return sizes, dict(urllib.parse.parse_qsl(landing.query)).get("error", "")


def echo(actor, upstream, server_id: str) -> str:
    with actor.client() as client:
        return run_tool(
            client,
            upstream,
            f"{server_id}_echo",
            {"text": PHRASE},
            tool_ids=[f"server:mcp:{server_id}"],
        )


def test_a_sign_in_asking_for_many_scopes_keeps_the_session_cookie_small(
    instance, upstream, person, wide_server
):
    mcp, server_id = wide_server

    sizes, error = sign_in(instance, person, server_id)

    asked = mcp.auth_server.requests_to("/authorize")[-1].query["scope"].split()
    assert asked == GOOGLE_WORKSPACE_SCOPES, "the sign-in did not ask for every scope"
    assert error == "", error
    assert sizes["owui-session"] <= COOKIE_LIMIT, (
        f"the session cookie is {sizes['owui-session']} bytes"
    )
    assert echo(person, upstream, server_id) == PHRASE
    assert mcp.presented[-1] == mcp.auth_server.issued[-1]["access_token"]


def test_a_sign_in_asking_for_one_scope_is_unchanged(instance, upstream, person, narrow_server):
    mcp, server_id = narrow_server

    sizes, error = sign_in(instance, person, server_id)

    assert mcp.auth_server.requests_to("/authorize")[-1].query["scope"] == SCOPE
    assert error == "", error
    assert sizes["owui-session"] <= COOKIE_LIMIT
    assert echo(person, upstream, server_id) == PHRASE
