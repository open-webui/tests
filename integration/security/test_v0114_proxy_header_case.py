"""Regression: the OpenAI and Ollama proxies must drop stale upstream headers in any casing.

open-webui 0.11.4 fix `44f9a4f7f` (PR #29843): `_clean_proxy_headers` in `routers/openai.py`
and `routers/ollama.py` dropped the upstream's Content-Encoding, Content-Length and
Transfer-Encoding (stale once aiohttp has decoded the body) by comparing names against
title-case entries. Servers behind uvicorn (vLLM, LiteLLM) send lowercase names, so a gzip
stream went out decoded but still labelled `content-encoding: gzip` and clients failed to
decompress it. The upstream's Server and Date were forwarded as well, next to the ones uvicorn
adds itself. The filter now compares lowercased names and strips Server and Date too.

The provider is a listener behind each proxy that gzips its stream under a lowercase
`content-encoding` and sends its own Server and Date. The instance runs aiohttp's pure-Python
parser, which keeps header names as sent; the C parser renames known headers to title case,
which is why the bug only showed on some installs.

Twin of unit/security/test_v0114_proxy_header_case.py.

Discriminates: passes on dev bbfa876af, fails with the strip set restored to the three
title-case names and the case-sensitive comparison (both proxies forward `content-encoding`
and a second Server and Date).
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest

from harness.listener import json_answer

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MODEL = "gzip-model"
REPLY = "decoded reply"


def _openai_stream() -> bytes:
    chunk = {
        "id": "gz",
        "object": "chat.completion.chunk",
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": REPLY}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _ollama_stream() -> bytes:
    lines = [
        {"model": MODEL, "message": {"role": "assistant", "content": REPLY}, "done": False},
        {"model": MODEL, "message": {"role": "assistant", "content": ""}, "done": True},
    ]
    return "".join(f"{json.dumps(line)}\n" for line in lines).encode()


def _gzipped(body: bytes, content_type: str):
    headers = {"Content-Type": content_type, "content-encoding": "gzip", "x-request-id": "abc123"}
    return 200, headers, gzip.compress(body)


def _connect_openai(client: httpx.Client, listener) -> str:
    listener.route("GET", "/models", json_answer({"data": [{"id": MODEL, "object": "model"}]}))
    listener.route("POST", "/chat/completions", _gzipped(_openai_stream(), "text/event-stream"))
    config = client.get("/openai/config").json()
    config["OPENAI_API_BASE_URLS"].append(listener.base_url)
    config["OPENAI_API_KEYS"].append("sk-listener")
    client.post("/openai/config/update", json=config).raise_for_status()
    return "/openai/chat/completions"


def _connect_ollama(client: httpx.Client, listener) -> str:
    listener.route("GET", "/api/tags", json_answer({"models": [{"name": MODEL, "model": MODEL}]}))
    listener.route("POST", "/api/chat", _gzipped(_ollama_stream(), "application/x-ndjson"))
    config = {
        "ENABLE_OLLAMA_API": True,
        "OLLAMA_BASE_URLS": [listener.base_url],
        "OLLAMA_API_CONFIGS": {},
    }
    client.post("/ollama/config/update", json=config).raise_for_status()
    return "/ollama/api/chat"


@pytest.fixture(scope="module")
def wire_casing_instance(instance_with):
    # aiohttp's C parser renames known headers to title case; the pure-Python one keeps the wire.
    return instance_with({"AIOHTTP_NO_EXTENSIONS": "1"})


@pytest.fixture(params=["openai", "ollama"])
def proxied_stream(request, wire_casing_instance, listener) -> tuple[httpx.Headers, bytes]:
    """The headers and raw body a client gets from a proxy streaming the listener's reply."""
    connect = _connect_openai if request.param == "openai" else _connect_ollama
    with wire_casing_instance.client() as client:
        restore = client.get(f"/{request.param}/config").json()
        chat_path = connect(client, listener)
        client.get("/api/models").raise_for_status()
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}
        with client.stream("POST", chat_path, json=payload) as response:
            assert response.status_code == 200, response.read()
            received = response.headers, b"".join(response.iter_raw())
        yield received
        client.post(f"/{request.param}/config/update", json=restore).raise_for_status()


def test_a_lowercase_content_encoding_is_not_forwarded(proxied_stream):
    headers, body = proxied_stream

    assert "content-encoding" not in headers, (
        "the proxy decoded a gzip stream and still labelled it content-encoding: gzip, so the "
        "client fails to decompress it (#29843)"
    )
    assert REPLY.encode() in body


def test_the_upstream_server_and_date_are_not_forwarded(proxied_stream):
    headers, _ = proxied_stream

    assert len(headers.get_list("server")) == 1, headers.get_list("server")
    assert len(headers.get_list("date")) == 1, headers.get_list("date")


def test_unrelated_headers_are_still_forwarded(proxied_stream):
    headers, _ = proxied_stream

    assert headers.get("x-request-id") == "abc123"
    assert headers["content-type"].split(";")[0] in ("text/event-stream", "application/x-ndjson")
