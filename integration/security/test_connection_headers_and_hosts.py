"""Regression tests for three 0.11.4 connection fixes, seen from the provider and the web.

* Custom header encoding (`7a4a4b93d`): a connection's custom headers were substituted as raw
  text, so a name or group carrying non-ASCII characters or a line break broke the request or
  reached the provider as something else. `parse_custom_headers` now percent-encodes the
  substituted value and keeps ASCII punctuation and existing escapes.
* Authentication type header (`ee46e2664`): with user info forwarding on, a provider request
  says whether the caller signed in (`jwt`) or used an API key (`api_key`) in the header named
  by FORWARD_USER_INFO_HEADER_AUTH_TYPE, and `{{AUTH_TYPE}}` works in custom headers.
* Dotless host addresses (`6f55514c5`, PR #29945, issue #28161): with local web fetching on, an
  address such as "http://apprise:8000" (here `localhost`) is accepted instead of refused as
  invalid.
* Client IP placeholders (`d39cb17b9`, #33833, issue #33819): `{{USER_IP}}` and
  `{{X_FORWARDED_FOR}}` in a connection's custom headers carry the client address Open WebUI
  resolves for the request, the address a trusted proxy forwarded included, on an API completion
  and on a chat sent the way the web client does, and a tool server connection's custom headers
  take them as well when the model calls its tool.

Twin of unit/security/test_connection_headers_and_hosts.py.

Discriminates: passes on dev bbfa876af; reverting 7a4a4b93d sends the name raw and the group
name's line break fails the provider request; reverting ee46e2664 drops the auth-type header and
leaves `{{AUTH_TYPE}}` unsubstituted; reverting 6f55514c5 makes the localhost fetch a 400.
The four IP placeholder tests pass on dev f659c0d12 and fail with d39cb17b9 reverted in a backend
copy (the provider gets the placeholders as written).
"""

from __future__ import annotations

import secrets

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask, send_message, wait_for_reply
from harness.listener import json_answer, text_answer
from harness.mcp_server import TOOL_SERVERS
from harness.openapi_server import calls_to, openapi_connection, serve_openapi
from harness.python_tools import EVERYONE_READS

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

FORWARDING = {
    "ENABLE_FORWARD_USER_INFO_HEADERS": "true",
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "ENABLE_API_KEYS": "true",
    "FORWARD_USER_INFO_HEADER_AUTH_TYPE": "X-Caller-Auth",
}

CUSTOM_HEADERS = {
    "X-Owner": "{{USER_NAME}}",
    "X-Groups": "{{USER_GROUPS}}",
    "X-Source": "{{AUTH_TYPE}}",
    "X-Tag": "team-a/b (core)",
    "X-Path": "files/a%20b.txt",
    "X-Real-IP": "{{USER_IP}}",
    "X-Client-Chain": "{{X_FORWARDED_FOR}}",
}


@pytest.fixture(scope="module")
def forwarding(instance_with):
    """An instance forwarding user info, fetching local pages and sending `CUSTOM_HEADERS`."""
    launched = instance_with(FORWARDING)
    with launched.client() as client:
        config = client.get("/openai/config").json()
        connection = config["OPENAI_API_CONFIGS"].get("0", {})
        config["OPENAI_API_CONFIGS"] = {"0": {**connection, "headers": CUSTOM_HEADERS}}
        client.post("/openai/config/update", json=config).raise_for_status()
    return launched


@pytest.fixture
def provider(forwarding):
    forwarding.upstream.reset()
    return forwarding.upstream


def _complete(client, headers: dict[str, str] | None = None) -> None:
    """A plain API completion, the way an API-key caller makes one."""
    body = {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post(
        "/api/chat/completions", json={**body, "stream": False}, headers=headers or {}
    )
    assert response.status_code == 200, f"completion failed: {response.text}"


def _sent_headers(provider) -> dict[str, str]:
    sent = provider.requests_to("/chat/completions")
    assert sent, "the provider was never called"
    return {name.lower(): value for name, value in sent[-1].headers.items()}


def test_a_non_ascii_name_reaches_the_provider_percent_encoded(forwarding, provider):
    zoe = create_user(forwarding, name="Zoë Müller")
    with zoe.client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers["x-owner"] == "Zo%C3%AB M%C3%BCller", (
        f"{{{{USER_NAME}}}} reached the provider as {headers['x-owner']!r} instead of "
        "percent-encoded (7a4a4b93d)"
    )
    assert headers["x-openwebui-user-name"] == "Zo%C3%AB M%C3%BCller"


def test_a_line_break_in_a_group_name_is_encoded_not_sent(forwarding, provider):
    member = create_user(forwarding)
    with forwarding.client() as admin_client:
        group = admin_client.post(
            "/api/v1/groups/create", json={"name": "Team\nInjected: true", "description": ""}
        ).json()
        admin_client.post(
            f"/api/v1/groups/id/{group['id']}/users/add", json={"user_ids": [member.id]}
        ).raise_for_status()

    with member.client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers["x-groups"] == "Team%0AInjected: true", (
        f"a group name's line break reached the provider as {headers['x-groups']!r} (7a4a4b93d)"
    )
    assert "injected" not in headers, "a group name smuggled a header of its own (7a4a4b93d)"


def test_ascii_punctuation_and_existing_escapes_stay_as_written(forwarding, provider):
    with create_user(forwarding).client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers["x-tag"] == "team-a/b (core)"
    assert headers["x-path"] == "files/a%20b.txt"


def test_a_browser_session_is_forwarded_as_jwt(forwarding, provider):
    with create_user(forwarding).client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers.get("x-caller-auth") == "jwt", (
        f"a signed-in session sent auth type {headers.get('x-caller-auth')!r} (ee46e2664)"
    )
    assert headers["x-source"] == "jwt", "{{AUTH_TYPE}} did not resolve (ee46e2664)"


def test_an_api_key_is_forwarded_as_api_key(forwarding, provider):
    admin = admin_of(forwarding)
    with admin.client() as client:
        created = client.post("/api/v1/auths/api_key")
    assert created.status_code == 200, created.text
    admin.token = created.json()["api_key"]

    with admin.client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers.get("x-caller-auth") == "api_key", (
        f"an API-key call sent auth type {headers.get('x-caller-auth')!r} (ee46e2664)"
    )
    assert headers["x-source"] == "api_key", "{{AUTH_TYPE}} did not resolve (ee46e2664)"


def test_the_client_address_fills_the_ip_placeholders(forwarding, provider):
    with create_user(forwarding).client() as client:
        _complete(client)

    headers = _sent_headers(provider)
    assert headers["x-real-ip"] == "127.0.0.1", (
        f"{{{{USER_IP}}}} reached the provider as {headers['x-real-ip']!r}, not the caller's IP"
    )
    assert headers["x-client-chain"] == "127.0.0.1", headers["x-client-chain"]


def test_the_address_a_trusted_proxy_forwards_fills_the_ip_placeholders(forwarding, provider):
    # the instance trusts a proxy on loopback, as a reverse proxy in front of it is
    with create_user(forwarding).client() as client:
        _complete(client, headers={"X-Forwarded-For": "203.0.113.7"})

    headers = _sent_headers(provider)
    assert headers["x-real-ip"] == "203.0.113.7", (
        f"{{{{USER_IP}}}} reached the provider as {headers['x-real-ip']!r}, "
        "not the address the proxy forwarded"
    )
    assert headers["x-client-chain"] == "203.0.113.7", headers["x-client-chain"]


def test_a_chat_sent_from_the_web_client_carries_the_client_address(forwarding, provider):
    with create_user(forwarding).client() as client:
        client.headers["X-Forwarded-For"] = "198.51.100.23"
        turn = send_message(client, "where am I writing from?")
        wait_for_reply(client, turn)

    headers = _sent_headers(provider)
    assert headers["x-real-ip"] == "198.51.100.23", (
        f"a web client chat sent {{{{USER_IP}}}} as {headers['x-real-ip']!r}"
    )


def _fetch(client, url: str):
    return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})


def test_a_dotless_host_is_fetched_with_local_fetch_on(forwarding, listener):
    listener.route("GET", "/notify", text_answer("<p>container page body</p>"))
    with create_user(forwarding).client() as client:
        fetched = _fetch(client, f"http://localhost:{listener.port}/notify")

    assert fetched.status_code == 200, (
        f"a dotless host was refused with local web fetch on (#28161): {fetched.text}"
    )
    assert "container page body" in fetched.json()["content"]


def test_a_dotless_host_stays_refused_without_local_fetch(make_user, listener):
    listener.route("GET", "/notify", text_answer("<p>container page body</p>"))
    with make_user().client() as client:
        fetched = _fetch(client, f"http://localhost:{listener.port}/notify")

    assert fetched.status_code == 400, fetched.text
    assert listener.requests_to("/notify") == []


def test_a_tool_server_call_carries_the_client_address(
    admin, make_user, preserve, listener, upstream
):
    # a tool server connection's custom headers take the same placeholders as a model connection's
    preserve(TOOL_SERVERS)
    title = f"Harbour desk {secrets.token_hex(3)}"
    serve_openapi(listener, title, {"get_tides": json_answer({"high": "12:04"})})
    connection = openapi_connection(
        listener.base_url, title, [EVERYONE_READS], headers={"X-Real-IP": "{{USER_IP}}"}
    )
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text

    upstream.queue(reply.tool_call("get_tides", {}), reply.text("High water at 12:04."))
    with make_user().client() as client:
        client.headers["X-Forwarded-For"] = "198.51.100.77"
        ask(client, "when is high water?", tool_ids=[f"server:{connection['info']['id']}"])

    calls = calls_to(listener, "get_tides")
    assert calls, "the model's call never reached the tool server"
    sent = {name.lower(): value for name, value in calls[-1].headers.items()}.get("x-real-ip")
    assert sent == "198.51.100.77", f"{{{{USER_IP}}}} reached the tool server as {sent!r}"
