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

Twin of unit/security/test_connection_headers_and_hosts.py.

Discriminates: passes on dev bbfa876af; reverting 7a4a4b93d sends the name raw and the group
name's line break fails the provider request; reverting ee46e2664 drops the auth-type header and
leaves `{{AUTH_TYPE}}` unsubstituted; reverting 6f55514c5 makes the localhost fetch a 400.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user
from harness.listener import text_answer

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


def _complete(client) -> None:
    """A plain API completion, the way an API-key caller makes one."""
    body = {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/api/chat/completions", json={**body, "stream": False})
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
