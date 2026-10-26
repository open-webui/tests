"""Dependency smoke: code and services Open WebUI reaches out to, each through its feature.

The mcp SDK connects to an MCP tool server when an admin verifies the connection and lists its
tools, and calls a tool when the model asks for one, over an httpx client Open WebUI builds: it
sends the connection's bearer key, follows a server that moved within its own address (a 307 on
the same scheme, host and port) but never to another address (since mcp 2 the connection fails
there, without a single request to the new address), and checks the server's certificate unless
`AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL` is off. validators decides which links a page fetch accepts
(a bare hostname only with local fetch on) and which search results survive a domain filter (twin
of unit/deps/test_validators.py), BeautifulSoup reads the fetched page (its text, and the title,
description and language stored with it; a `.xml` link through its XML parser), black formats code
for the code editor, and opentelemetry exports request traces to an OTLP/HTTP collector (requests
carries them there, as it carries a document to Tika) or, by default, over gRPC, where a busy
collector's `RetryInfo` (googleapis-common-protos) tells the exporter when to try again. The
collector's address is only read at boot, so each tracing test boots an instance of its own; the
gRPC collector is `harness/otlp_collector.py`.

Discriminates: passes on dev bbfa876af; in a backend copy, skipping `session.initialize()` fails the
MCP verification, building the httpx client without `headers` or with `verify=False` always fails
the bearer and certificate tests in turn. On dev 302b8cfd5, an SDK that counts every redirect as
within the origin (following it as mcp 1 with redirects on did) fails the other-address test, and
one that counts none as within it fails the same-address test. Taking an `isError` result for a
success fails the failing tool test, and so does raising the `isError` content list
in place of its text (#32918, red on dev 63eed95c8), dropping the `validators.url` check lets the
malformed link through to the fetch, calling it without `simple_host` refuses the bare hostname,
dropping it from the search result filter keeps the malformed results, a loader that stores no page
title fails the metadata test, asking BeautifulSoup for an unknown parser in place of "xml" fails
the feed test, returning the code unformatted fails the formatter, black without string
normalisation fails the wrapping test and never adding the span processor leaves the collector
empty. On dev ef67cc3fa, a `google.rpc.error_details_pb2` whose `RetryInfo` reads no delay, placed
ahead of the real one in a backend copy (a bump that stops parsing it), fails the retry test: the
batch comes back long before the delay the collector asked for. Exporting with the HTTP exporter in
the gRPC branch fails both gRPC tests.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field

import pytest
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.instance import free_port
from harness.listener import listening, text_answer
from harness.mcp_server import (
    CAPSIZE_ERROR,
    ECHO_DESCRIPTION,
    TOOL_SERVERS,
    mcp_connection,
    serving_mcp,
)
from harness.otlp_collector import serving_trace_collector, trace_collector_env
from harness.terminal_server import read_grant
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

PAGE_TEXT = "Herons wait motionless in the shallows"
EXPORT_WAIT = 30.0


def _verify_mcp(admin, url: str, key: str = ""):
    connection = {
        "type": "mcp",
        "url": url,
        "path": "",
        "auth_type": "bearer" if key else "none",
        "key": key,
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


@dataclass
class KeyCheck:
    """An MCP server's bearer check that takes one key and keeps every token it was shown."""

    key: str
    presented: list[str] = field(default_factory=list)

    async def verify_token(self, token: str) -> AccessToken | None:
        self.presented.append(token)
        if token != self.key:
            return None
        return AccessToken(token=token, client_id="open-webui", scopes=[])


def test_the_connection_key_reaches_an_mcp_server_that_requires_it(admin):
    port = free_port()
    check = KeyCheck(key=f"harbour-{secrets.token_hex(8)}")
    settings = AuthSettings(
        issuer_url="http://127.0.0.1:9", resource_server_url=f"http://127.0.0.1:{port}/mcp"
    )
    with serving_mcp(port, auth=settings, token_verifier=check) as url:
        with_key = _verify_mcp(admin, url, key=check.key)
        with_wrong_key = _verify_mcp(admin, url, key="not-the-key")
        without_key = _verify_mcp(admin, url)

    assert with_key.status_code == 200, with_key.text
    assert [spec["name"] for spec in with_key.json()["specs"]] == ["echo"]
    assert with_wrong_key.status_code == 400
    assert without_key.status_code == 400
    assert check.key in check.presented


def test_an_mcp_server_that_moved_within_its_address_is_followed(admin):
    with serving_mcp(moved_from="/old/mcp") as url:
        verified = _verify_mcp(admin, url.replace("/mcp", "/old/mcp"))

    assert verified.status_code == 200, verified.text
    assert [spec["name"] for spec in verified.json()["specs"]] == ["echo"]


def test_an_mcp_server_that_moved_to_another_address_is_not_followed(admin, listener):
    with listening() as new_home:
        for method in ("GET", "POST", "DELETE"):
            listener.route(method, "/old/mcp", (307, {"Location": f"{new_home.base_url}/mcp"}, b""))
        verified = _verify_mcp(admin, f"{listener.base_url}/old/mcp")

    assert verified.status_code == 400, verified.text
    assert verified.json()["detail"] == "Failed to create MCP client"
    assert listener.requests_to("/old/mcp"), "the old address was never asked"
    assert not new_home.received, "the connection followed the server to another address"


def test_an_mcp_server_with_an_untrusted_certificate_is_refused(admin):
    with serving_mcp(tls=True) as url:
        verified = _verify_mcp(admin, url)

    assert verified.status_code == 400, verified.text


@pytest.mark.slow
def test_with_certificate_checks_off_an_untrusted_mcp_server_is_used(instance_with):
    unchecked = instance_with({"AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL": "false"})
    with serving_mcp(tls=True) as url:
        verified = _verify_mcp(admin_of(unchecked), url)

    assert verified.status_code == 200, verified.text
    assert [spec["name"] for spec in verified.json()["specs"]] == ["echo"]


@pytest.fixture
def harbour_tools(admin, make_user, preserve):
    """(person, server id): an MCP server with a tool that fails, which `person` may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"harbour_{secrets.token_hex(4)}"
    with serving_mcp(failing=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


def _tool_result(person, upstream, server_id: str, tool: str, arguments: dict) -> str:
    """Have the model call one MCP tool; returns what the tool message told the model."""
    upstream.queue(reply.tool_call(f"{server_id}_{tool}", arguments), reply.text("Noted."))
    with person.client() as client:
        ask(client, f"use {tool}", tool_ids=[f"server:mcp:{server_id}"])
    follow_up = upstream.chat_requests()[-1]["messages"]
    [tool_message] = [entry for entry in follow_up if entry["role"] == "tool"]
    return tool_message["content"]


def test_the_model_calls_an_mcp_tool_and_gets_its_answer(harbour_tools, upstream):
    person, server_id = harbour_tools

    answered = _tool_result(person, upstream, server_id, "echo", {"text": "high tide at six"})

    assert "high tide at six" in answered
    assert "error" not in answered


def test_a_failing_mcp_tool_is_reported_to_the_model_as_an_error(harbour_tools, upstream):
    """#32918: the error reached the model as a dump of the MCP content list, not its text."""
    person, server_id = harbour_tools

    answered = _tool_result(person, upstream, server_id, "capsize", {})

    assert CAPSIZE_ERROR in answered
    reported = json.loads(answered)
    assert "error" in reported, f"the failure came back as a result: {answered}"
    assert reported["error"].endswith(CAPSIZE_ERROR), (
        f"the model got the content list instead of the server's error text: {reported['error']}"
    )


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


@pytest.mark.slow
def test_with_local_fetch_on_a_link_to_a_bare_hostname_is_fetched(local_fetch, listener):
    listener.route("GET", "/page", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))
    bare_hostname = listener.base_url.replace("127.0.0.1", "localhost")

    fetched = _fetch(local_fetch, f"{bare_hostname}/page")

    assert fetched.status_code == 200, fetched.text
    assert PAGE_TEXT in fetched.json()["content"]


def test_a_search_result_with_a_malformed_link_is_dropped(admin, preserve, listener):
    preserve(RETRIEVAL_CONFIG)
    links = [
        "https://example.org/herons",
        "https://example.org/pa ge",
        "not a url",
        "https://example.org/egrets",
    ]
    with admin.client() as client:
        save_web_settings(
            client,
            **serve_search_results(listener, links),
            WEB_SEARCH_DOMAIN_FILTER_LIST=["!blocked.example"],
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
        )
        searched = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["herons"]})

    assert searched.status_code == 200, searched.text
    found = [item["link"] for item in searched.json()["items"]]
    assert found == ["https://example.org/herons", "https://example.org/egrets"]


PAGE = f"""<!DOCTYPE html>
<html lang="en-GB"><head><title>Harbour notices</title>
<meta name="description" content="Notices for the harbour"></head>
<body><h1>Notices</h1><p>{PAGE_TEXT}</p></body></html>"""
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Harbour feed</title>
<item><title>Ferry timetable</title><link>https://harbour.example/ferry</link></item>
</channel></rss>"""


@pytest.mark.slow
def test_a_fetched_page_is_stored_with_its_title_description_and_language(local_fetch, listener):
    listener.route("GET", "/notices", text_answer(PAGE))

    with local_fetch.client() as client:
        fetched = client.post(
            "/api/v1/retrieval/process/web", json={"url": f"{listener.base_url}/notices"}
        )
        assert fetched.status_code == 200, fetched.text
        stored = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": fetched.json()["collection_name"], "query": "herons"},
        )

    assert stored.status_code == 200, stored.text
    [[document]], [[metadata]] = stored.json()["documents"], stored.json()["metadatas"]
    assert PAGE_TEXT in document and "<p>" not in document
    assert metadata["title"] == "Harbour notices"
    assert metadata["description"] == "Notices for the harbour"
    assert metadata["language"] == "en-GB"


@pytest.mark.slow
def test_a_fetched_xml_feed_is_read_as_text(local_fetch, listener):
    listener.route("GET", "/feed.xml", text_answer(FEED, content_type="application/xml"))

    fetched = _fetch(local_fetch, f"{listener.base_url}/feed.xml")

    assert fetched.status_code == 200, fetched.text
    content = fetched.json()["content"]
    assert "Ferry timetable" in content and "https://harbour.example/ferry" in content
    assert "<item>" not in content


def test_black_formats_code_and_reports_what_it_cannot_parse(admin):
    with admin.client() as client:
        formatted = client.post("/api/v1/utils/code/format", json={"code": "x=[1,2]\nprint( x )"})
        broken = client.post("/api/v1/utils/code/format", json={"code": "def broken(:\n"})

    assert formatted.status_code == 200, formatted.text
    assert formatted.json()["code"] == "x = [1, 2]\nprint(x)\n"
    assert broken.status_code == 400, broken.text
    assert "Cannot parse" in broken.json()["detail"]


ARGUMENTS = "argument_one, argument_two, argument_three, argument_four, argument_five"
LONG_CALL = f"result = some_function({ARGUMENTS})\n"
WRAPPED_CALL = f"result = some_function(\n    {ARGUMENTS}\n)\n"


def test_black_wraps_past_88_columns_and_leaves_formatted_code_alone(admin):
    with admin.client() as client:
        quoted = client.post("/api/v1/utils/code/format", json={"code": "s = 'harbour'\n"})
        wrapped = client.post("/api/v1/utils/code/format", json={"code": LONG_CALL})
        again = client.post("/api/v1/utils/code/format", json={"code": WRAPPED_CALL})

    assert quoted.json()["code"] == 's = "harbour"\n'
    assert wrapped.json()["code"] == WRAPPED_CALL
    assert again.json()["code"] == WRAPPED_CALL


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


@pytest.fixture(scope="module")
def grpc_collector():
    with serving_trace_collector() as service:
        yield service


@pytest.fixture
def traced_over_grpc(instance_with, grpc_collector):
    return instance_with(trace_collector_env(grpc_collector))


def _wait_for(condition, timeout: float = EXPORT_WAIT) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.mark.slow
def test_request_traces_are_exported_over_otlp_grpc(traced_over_grpc, grpc_collector):
    with traced_over_grpc.client() as client:
        client.get("/api/version").raise_for_status()

    def version_span_arrived() -> bool:
        return any("/api/version" in name for name in grpc_collector.span_names())

    _wait_for(version_span_arrived)
    assert version_span_arrived(), "no span for the request reached the gRPC collector"


@pytest.mark.slow
def test_a_busy_collector_gets_the_batch_again_when_its_retry_info_says(
    traced_over_grpc, grpc_collector
):
    retry_after = 3.0
    grpc_collector.refuse_next(retry_after)
    with traced_over_grpc.client() as client:
        client.get("/api/version").raise_for_status()

    _wait_for(lambda: grpc_collector.refusals)
    assert grpc_collector.refusals, "the collector was never sent the batch"
    refused_at = grpc_collector.refusals[-1]
    _wait_for(lambda: grpc_collector.first_export_after(refused_at) is not None)
    retried_at = grpc_collector.first_export_after(refused_at)

    assert retried_at is not None, "the refused batch was never sent again"
    # the exporter's own backoff would have come back after about a second
    assert retried_at - refused_at >= retry_after - 0.5, retried_at - refused_at
