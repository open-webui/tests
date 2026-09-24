"""Dependency smoke: code and services Open WebUI reaches out to, each through its feature.

The mcp SDK connects to an MCP tool server when an admin verifies the connection and lists its
tools. validators decides which links a page fetch accepts, black formats code for the code
editor, and opentelemetry exports request traces to an OTLP/HTTP collector (requests carries
them there, as it carries a document to Tika). The collector's address is only read at boot, so
the tracing test boots an instance of its own.

Discriminates: passes on dev bbfa876af; in a backend copy, skipping `session.initialize()` fails
the MCP verification, dropping the `validators.url` check lets the malformed link through to the
fetch, returning the code unformatted fails the formatter and never adding the span processor
leaves the collector empty.
"""

from __future__ import annotations

import time

import pytest

from harness.instance import free_port
from harness.listener import listening, text_answer
from harness.mcp_server import ECHO_DESCRIPTION, serving_mcp
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

PAGE_TEXT = "Herons wait motionless in the shallows"
EXPORT_WAIT = 30.0


def _verify_mcp(admin, url: str):
    connection = {
        "type": "mcp",
        "url": url,
        "path": "",
        "auth_type": "none",
        "key": "",
        "config": {},
    }
    with admin.client() as client:
        return client.post("/api/v1/configs/tool_servers/verify", json=connection)


def test_verifying_an_mcp_server_lists_its_tools(admin):
    with serving_mcp() as url:
        verified = _verify_mcp(admin, url)

    assert verified.status_code == 200, verified.text
    [echo] = verified.json()["specs"]
    assert echo["name"] == "echo"
    assert echo["description"] == ECHO_DESCRIPTION
    assert echo["parameters"]["properties"]["text"]["type"] == "string"


def test_verifying_an_mcp_server_that_is_not_there_fails(admin):
    verified = _verify_mcp(admin, f"http://127.0.0.1:{free_port()}/mcp")

    assert verified.status_code == 400, verified.text


@pytest.fixture
def local_fetch(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


def _fetch(instance, url: str):
    with instance.client() as client:
        return client.post("/api/v1/retrieval/process/web?process=false", json={"url": url})


@pytest.mark.slow
def test_a_page_link_is_fetched_and_a_malformed_one_refused_unfetched(local_fetch, listener):
    listener.route("GET", "/page", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))

    fetched = _fetch(local_fetch, f"{listener.base_url}/page")
    assert fetched.status_code == 200, fetched.text
    assert PAGE_TEXT in fetched.json()["content"]

    fetches_so_far = len(listener.received)
    # a space in the path: fetchers would quote it and load the page, validators refuses it
    refused = _fetch(local_fetch, f"{listener.base_url}/pa ge")
    assert refused.status_code == 400, refused.text
    assert len(listener.received) == fetches_so_far, "a link validators refuses was fetched"
    assert _fetch(local_fetch, "not a url").status_code == 400


def test_black_formats_code_and_reports_what_it_cannot_parse(admin):
    with admin.client() as client:
        formatted = client.post("/api/v1/utils/code/format", json={"code": "x=[1,2]\nprint( x )"})
        broken = client.post("/api/v1/utils/code/format", json={"code": "def broken(:\n"})

    assert formatted.status_code == 200, formatted.text
    assert formatted.json()["code"] == "x = [1, 2]\nprint(x)\n"
    assert broken.status_code == 400, broken.text
    assert "Cannot parse" in broken.json()["detail"]


@pytest.fixture(scope="module")
def collector():
    with listening() as service:
        service.route("POST", "/v1/traces", (200, {"Content-Type": "application/x-protobuf"}, b""))
        yield service


@pytest.fixture
def traced(instance_with, collector):
    return instance_with(
        {
            "ENABLE_OTEL": "true",
            "ENABLE_OTEL_TRACES": "true",
            "OTEL_OTLP_SPAN_EXPORTER": "http",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"{collector.base_url}/v1/traces",
            "OTEL_BSP_SCHEDULE_DELAY": "200",
        }
    )


def _exported_spans(collector) -> bytes:
    return b"".join(request.body for request in collector.requests_to("/v1/traces"))


@pytest.mark.slow
def test_request_traces_are_exported_over_otlp_http(traced, collector):
    with traced.client() as client:
        client.get("/api/version").raise_for_status()

    deadline = time.monotonic() + EXPORT_WAIT
    while b"/api/version" not in _exported_spans(collector) and time.monotonic() < deadline:
        time.sleep(0.2)

    exported = _exported_spans(collector)
    assert b"/api/version" in exported, "no span for the request reached the collector"
    assert b"open-webui" in exported, "the spans do not name the service"
    sent = collector.requests_to("/v1/traces")[-1]
    assert sent.headers["Content-Type"] == "application/x-protobuf"
