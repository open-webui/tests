"""Regression: an Ollama connection's headers and sign-in only reached the server on chats.

Fix commit `bc2416c5d` (open-webui/open-webui#29868) moved the header builder the OpenAI
connections use into a shared helper and has the Ollama model list, the loaded-model list, the
version check and the admin's connection check call it too. Before it those requests carried
only the connection's key as a bearer token: the custom headers, the authentication type and
cookie forwarding were applied to chats alone, so an Ollama server behind a gateway such as
Cloudflare Access answered the model list and the check with the gateway's refusal.

The Ollama stand-in records every request, so each test reads what the listing, the version
check and the connection check actually sent.

Discriminates: passes on dev bc2416c5d; with the fix reverted in `routers/ollama.py` the model
list, loaded-model list, version and verify requests carry neither the custom header nor the
forwarded cookie, and the key goes out as a bearer token despite `auth_type: none`; the chat
test passes on both.

The client IP placeholders (#33833, d39cb17b9) fill an Ollama connection's custom headers on a
chat as they do an OpenAI connection's: passes on dev f659c0d12 and fails with d39cb17b9 reverted
in a backend copy (the header goes out as written).
"""

from __future__ import annotations

import pytest

from harness.chat import ask
from harness.listener import ReceivedRequest
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"
GATEWAY_HEADER = "CF-Access-Client-Id"
GATEWAY_VALUE = "gateway-client.access"
GATED = {"headers": {GATEWAY_HEADER: GATEWAY_VALUE}}


def header(request: ReceivedRequest, name: str) -> str | None:
    by_name = {key.lower(): value for key, value in request.headers.items()}
    return by_name.get(name.lower())


def last_request_to(server, path: str) -> ReceivedRequest:
    requests = server.listener.requests_to(path)
    assert requests, f"the Ollama server was never asked for {path}"
    return requests[-1]


@pytest.fixture
def operator(make_user):
    """A fresh admin, so no model list cached for another account answers in its place."""
    return make_user(role="admin")


@pytest.fixture
def ollama(preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    server.loaded.append(OLLAMA_MODEL)
    return server


def connect(operator, server, **config) -> None:
    with operator.client() as client:
        connect_ollama(client, server.listener, **config)


# --- narrow: the listing and the checks carry the connection's settings ---------------------


@pytest.mark.parametrize("path", ["/api/tags", "/api/ps"])
def test_the_model_list_carries_the_custom_headers(operator, ollama, path):
    connect(operator, ollama, key="sk-ollama", **GATED)

    with operator.client() as client:
        listed = client.get("/ollama/api/tags")
    assert listed.status_code == 200, listed.text

    assert header(last_request_to(ollama, path), GATEWAY_HEADER) == GATEWAY_VALUE


def test_the_version_check_carries_the_custom_headers(operator, ollama):
    connect(operator, ollama, **GATED)

    with operator.client() as client:
        version = client.get("/ollama/api/version")
    assert version.status_code == 200, version.text

    assert header(last_request_to(ollama, "/api/version"), GATEWAY_HEADER) == GATEWAY_VALUE


def test_verifying_a_connection_sends_its_headers(operator, ollama):
    form = {"url": ollama.listener.base_url, "key": "", "config": GATED}

    with operator.client() as client:
        verified = client.post("/ollama/verify", json=form)
    assert verified.status_code == 200, verified.text

    assert header(last_request_to(ollama, "/api/version"), GATEWAY_HEADER) == GATEWAY_VALUE


def test_auth_type_none_keeps_the_key_off_the_model_list(operator, ollama):
    connect(operator, ollama, key="sk-not-for-the-gateway", auth_type="none")

    with operator.client() as client:
        client.get("/ollama/api/tags").raise_for_status()

    assert header(last_request_to(ollama, "/api/tags"), "Authorization") is None


def test_forwarded_cookies_reach_the_model_list(operator, ollama):
    connect(operator, ollama, forward_cookies=True)

    with operator.client() as client:
        listed = client.get("/ollama/api/tags", headers={"Cookie": "CF_Authorization=gate-jwt"})
    assert listed.status_code == 200, listed.text

    sent_cookie = header(last_request_to(ollama, "/api/tags"), "Cookie") or ""
    assert "CF_Authorization=gate-jwt" in sent_cookie


# --- broad: every call the model picker and admin page make on a gated connection -------------


def test_every_listing_call_on_a_gated_connection_passes_the_gateway(operator, ollama):
    connect(operator, ollama, **GATED)

    with operator.client() as client:
        models = client.get("/api/models", params={"refresh": "true"})
        version = client.get("/ollama/api/version")
    assert models.status_code == 200 and version.status_code == 200

    assert OLLAMA_MODEL in {model["id"] for model in models.json()["data"]}
    ungated = [
        request.path
        for request in ollama.listener.received
        if request.method == "GET" and header(request, GATEWAY_HEADER) != GATEWAY_VALUE
    ]
    assert not ungated, f"requests without the gateway header: {ungated}"


# --- nearby: what already worked keeps working ----------------------------------------------


def test_a_chat_still_carries_the_custom_headers(operator, ollama):
    connect(operator, ollama, **GATED)

    with operator.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        _, message = ask(client, "hi", model=OLLAMA_MODEL)

    assert message["content"] == "pong"
    assert header(last_request_to(ollama, "/api/chat"), GATEWAY_HEADER) == GATEWAY_VALUE


def test_the_key_is_still_sent_as_a_bearer_token_by_default(operator, ollama):
    connect(operator, ollama, key="sk-ollama")

    with operator.client() as client:
        client.get("/ollama/api/tags").raise_for_status()

    assert header(last_request_to(ollama, "/api/tags"), "Authorization") == "Bearer sk-ollama"


def test_verifying_without_a_config_still_works(operator, ollama):
    form = {"url": ollama.listener.base_url, "key": "sk-ollama"}

    with operator.client() as client:
        verified = client.post("/ollama/verify", json=form)

    assert verified.status_code == 200, verified.text
    assert verified.json() == {"version": ollama.version}
    assert header(last_request_to(ollama, "/api/version"), "Authorization") == "Bearer sk-ollama"


def test_cookies_are_not_forwarded_unless_asked(operator, ollama):
    connect(operator, ollama, **GATED)

    with operator.client() as client:
        client.get("/ollama/api/tags", headers={"Cookie": "CF_Authorization=gate-jwt"})

    assert "gate-jwt" not in (header(last_request_to(ollama, "/api/tags"), "Cookie") or "")


def test_a_chat_fills_the_ip_placeholders_in_the_custom_headers(operator, ollama):
    # the client IP placeholders (#33833) work on Ollama connections as on OpenAI ones
    connect(operator, ollama, headers={"X-Real-IP": "{{USER_IP}}"})

    with operator.client() as client:
        client.headers["X-Forwarded-For"] = "203.0.113.41"
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        _, message = ask(client, "hi", model=OLLAMA_MODEL)

    assert message["content"] == "pong"
    sent = header(last_request_to(ollama, "/api/chat"), "X-Real-IP")
    assert sent == "203.0.113.41", f"{{{{USER_IP}}}} reached the Ollama server as {sent!r}"
