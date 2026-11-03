"""Dependency smoke: OpenTelemetry export of traces, metrics and logs, with psutil's host metrics.

With `ENABLE_OTEL` and its three signals on, Open WebUI builds a tracer, meter and logger
provider on an SDK `Resource` named after `OTEL_SERVICE_NAME` and exports each over OTLP to a
collector, here a local one (`harness.otel_collector`) that decodes what it gets. Tracing wires
the instrumentors: FastAPI's opens a span per request, which the sign-in check stamps with the
user and how they signed in (a session or an API key). The client instrumentors open a span per
outgoing call, which Open WebUI's hooks name after the method and URL and mark as an error on a
4xx or 5xx answer: aiohttp's for the model provider, requests' for a Tika server and httpx's for a
Chroma server (sync). An MCP server's calls run over the mcp SDK's vendored httpx2, which the
httpx instrumentor does not wrap, so the SDK exports its own protocol span per request instead.
Redis's request hook stamps each command with the
server and the statement, and SQLAlchemy's traces the synchronous engine. The metrics middleware
counts requests by route template, observable gauges count the accounts, and the system metrics
instrumentor reports the server's own memory and CPU time as psutil reads them. Every log line is
exported carrying the trace of the request that wrote it, and the server log prints that trace id
beside the line. The default gRPC exporters send all three signals to a gRPC collector with the
configured basic auth; that instance keeps its state in Redis and its vectors on a Chroma server.

Each log line reaches the collector once, with traces and logs both on or with logs alone (issue
#31524, fixed in PR #31528). Open WebUI hands each record to its OTLP log handler, and the logging
instrumentor it enables for traces once installed a second handler on the root logger
(opentelemetry-instrumentation-logging 0.63b1 does so unless log auto-instrumentation is off), so
with traces on every line was exported twice.

A request no route matches (a static file, a page of the web app, a probe for a missing path) is
counted without a route label (issue open-webui/open-webui#33576). The middleware once labelled it
with its raw path, so every such path became a metric series of its own.

Twin of unit/deps/test_opentelemetry.py and unit/deps/test_psutil.py.

Discriminates: passes on dev a5bc78300; in a backend copy, calling the logging instrumentor
without `enable_log_auto_instrumentation=False` (the parent of bff0492b5) fails the once-only test
of the full setup, which sees two copies, while the logs-only test passes on both; also,
dropping the user attributes from the sign-in check fails both identity tests, request hooks that
return at once fail the three client span tests, a Redis hook that does fails the last of them,
instrumenting no engine fails the database test, the metrics middleware counting the raw path
fails the route test, a user gauge that yields nothing fails the gauge test, a logger that binds
no trace to the server log fails the log trace test, `SystemMetricsInstrumentor` left out fails
the host metrics test and headers left off the gRPC span exporter fail the gRPC test. With
`psutil.Process.memory_info` patched to report a tenth of the memory the host metrics test fails.
The route test fails on dev f3ef8b949, which labels the unmatched request with its raw path, and
passes with the fix for #33576.
"""

from __future__ import annotations

import base64
import time
import uuid

import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.chroma_server import chroma_env, serving_chroma
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import json_answer
from harness.mcp_server import serving_mcp
from harness.otel_collector import (
    SERVICE_NAME,
    grpc_collector_env,
    http_collector_env,
    serving_grpc_collector,
    serving_http_collector,
)

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

FASTAPI_SCOPE = "opentelemetry.instrumentation.fastapi"
AIOHTTP_SCOPE = "opentelemetry.instrumentation.aiohttp_client"
REQUESTS_SCOPE = "opentelemetry.instrumentation.requests"
HTTPX_SCOPE = "opentelemetry.instrumentation.httpx"
MCP_SDK_SCOPE = "mcp-python-sdk"
REDIS_SCOPE = "opentelemetry.instrumentation.redis"
RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
SQLALCHEMY_SCOPE = "opentelemetry.instrumentation.sqlalchemy"
STATUS_ERROR = 2
GRPC_USER, GRPC_PASSWORD = "collector", "otel-secret"


@pytest.fixture(scope="module")
def collector():
    with serving_http_collector() as served:
        yield served


@pytest.fixture
def exporting(instance_with, collector):
    return instance_with({**http_collector_env(collector), "ENABLE_API_KEYS": "true"})


def _server_spans(collector, route: str, user_id: str | None = None) -> list:
    return [
        span
        for span in collector.spans()
        if span.scope == FASTAPI_SCOPE
        and span.attributes.get("http.route") == route
        and (user_id is None or span.attributes.get("client.user.id") == user_id)
    ]


def _spans_named(collector, scope: str, name: str) -> list:
    return [span for span in collector.spans() if span.scope == scope and span.name == name]


def test_a_request_span_carries_its_route_status_and_signed_in_user(exporting, collector):
    account = create_user(exporting)
    with account.client() as client:
        client.get("/api/v1/auths/").raise_for_status()

    [span] = collector.wait_for(
        lambda: _server_spans(collector, "/api/v1/auths/", account.id), "the request's span"
    )
    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.status_code"] == 200
    assert span.attributes["client.user.email"] == account.email
    assert span.attributes["client.user.role"] == "user"
    assert span.attributes["client.auth.type"] == "jwt"
    assert span.resource["service.name"] == SERVICE_NAME


def test_a_request_signed_with_an_api_key_is_marked_as_one(exporting, collector):
    admin = admin_of(exporting)
    with admin.client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    with exporting.client(generated.json()["api_key"]) as client:
        client.get("/api/v1/auths/api_key").raise_for_status()

    def keyed_reads():
        spans = _server_spans(collector, "/api/v1/auths/api_key", admin.id)
        return [span for span in spans if span.attributes["http.method"] == "GET"]

    [span] = collector.wait_for(keyed_reads, "the keyed request's span")
    assert span.attributes["client.auth.type"] == "api_key"


def test_calls_to_the_model_provider_are_client_spans_named_by_the_hooks(exporting, collector):
    provider = exporting.upstream.base_url
    exporting.upstream.queue(reply.error(500, "the provider is down"))
    with create_user(exporting).client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        ask(client, "are you there?")

    listed = collector.wait_for(
        lambda: _spans_named(collector, AIOHTTP_SCOPE, f"GET {provider}/models"), "the list call"
    )
    failed = collector.wait_for(
        lambda: _spans_named(collector, AIOHTTP_SCOPE, f"POST {provider}/chat/completions"),
        "the chat call",
    )
    assert listed[-1].attributes["http.url"] == f"{provider}/models"
    assert listed[-1].attributes["http.status_code"] == 200
    assert listed[-1].status != STATUS_ERROR
    assert failed[-1].attributes["http.method"] == "POST"
    assert failed[-1].attributes["http.status_code"] == 500
    assert failed[-1].status == STATUS_ERROR


def test_calls_through_requests_are_hook_named_and_an_mcp_server_is_a_sdk_span(
    exporting, collector, preserve, listener
):
    preserve(RETRIEVAL_CONFIG, on=exporting)
    answers = iter([(503, {}, b""), json_answer({"X-TIKA:content": "extracted by tika"})])
    listener.route("PUT", "/tika/text", lambda _request: next(answers))
    tika = f"PUT {listener.base_url}/tika/text"
    with admin_of(exporting).client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"CONTENT_EXTRACTION_ENGINE": "tika", "TIKA_SERVER_URL": listener.base_url},
        )
        assert saved.status_code == 200, saved.text
        for _ in range(2):
            client.post(
                "/api/v1/files/?process_in_background=false",
                files={"file": ("blob.bin", b"\x00\x01\x02", "application/octet-stream")},
            )
        with serving_mcp() as mcp_url:
            verified = client.post("/api/v1/configs/tool_servers/verify", json=_mcp(mcp_url))
    assert verified.status_code == 200, verified.text

    def both_tika_calls() -> list:
        calls = _spans_named(collector, REQUESTS_SCOPE, tika)
        return calls if len(calls) == 2 else []

    refused, extracted = collector.wait_for(both_tika_calls, "both Tika calls")
    assert refused.attributes["http.status_code"] == 503
    assert refused.status == STATUS_ERROR
    assert extracted.attributes["http.status_code"] == 200
    assert extracted.status != STATUS_ERROR
    assert extracted.attributes["http.url"] == f"{listener.base_url}/tika/text"
    # mcp 2 carries its traffic over the vendored httpx2, which the httpx instrumentor never sees;
    # the SDK exports its own protocol span per request instead, under its own scope.
    mcp_spans = collector.wait_for(
        lambda: [
            span
            for span in collector.spans()
            if span.scope == MCP_SDK_SCOPE
            and span.attributes.get("mcp.method.name") == "tools/list"
        ],
        "the MCP tools/list span",
    )
    assert all(span.status != STATUS_ERROR for span in mcp_spans)
    assert all(span.attributes.get("jsonrpc.request.id") for span in mcp_spans)


def _mcp(url: str) -> dict:
    return {"type": "mcp", "url": url, "path": "", "auth_type": "none", "key": "", "config": {}}


def test_database_statements_are_traced(exporting, collector):
    with create_user(exporting).client() as client:
        client.get("/api/v1/auths/").raise_for_status()

    statements = collector.wait_for(
        lambda: [
            span.attributes.get("db.statement", "")
            for span in collector.spans()
            if span.scope == SQLALCHEMY_SCOPE and span.attributes.get("db.statement")
        ],
        "a traced database statement",
    )
    assert any(statement.lstrip().upper().startswith("SELECT") for statement in statements)


def test_requests_are_counted_by_route_template(exporting, collector):
    looked_up = create_user(exporting)
    unrouted_path = f"/no-such-page-{uuid.uuid4().hex[:8]}"
    with admin_of(exporting).client() as client:
        client.get(f"/api/v1/users/{looked_up.id}").raise_for_status()
        client.get(unrouted_path)

    def counted():
        return [
            value
            for attributes, value in collector.metric_points("http.server.requests")
            if attributes.get("http.route") == "/api/v1/users/{user_id}"
            and attributes.get("http.status_code") == 200
            and attributes.get("http.method") == "GET"
        ]

    assert collector.wait_for(counted, "a count for the templated route")[-1] >= 1
    assert not [
        attributes
        for attributes, _ in collector.metric_points("http.server.requests")
        if looked_up.id in str(attributes.get("http.route"))
    ], "a user id reached the route label"

    def unrouted_counts():
        return [
            attributes
            for attributes, _ in collector.metric_points("http.server.requests")
            if "http.route" not in attributes or unrouted_path in str(attributes["http.route"])
        ]

    unrouted = collector.wait_for(unrouted_counts, "a count for the request no route matched")
    raw_paths = [attributes["http.route"] for attributes in unrouted if "http.route" in attributes]
    assert not raw_paths, f"a path no route matched became a route label: {raw_paths}"


def test_the_account_gauge_counts_every_account(exporting, collector):
    create_user(exporting)
    with admin_of(exporting).client() as client:
        accounts = client.get("/api/v1/users/all")
    assert accounts.status_code == 200, accounts.text
    total = len(accounts.json()["users"])

    def latest_total():
        points = collector.metric_points("webui.users.total")
        return points and points[-1][1] == total

    collector.wait_for(latest_total, f"a user count of {total}")


def test_the_server_process_memory_and_cpu_come_from_psutil(exporting, collector):
    def latest(name: str, **labels) -> float | None:
        points = [
            value
            for attributes, value in collector.metric_points(name)
            if all(attributes.get(key) == wanted for key, wanted in labels.items())
        ]
        return points[-1] if points else None

    collector.wait_for(lambda: latest("process.memory.usage"), "the process memory")
    resident = exporting.rss_bytes()

    # the export trails the read above by at most an interval, so allow for growth in between
    assert 0.5 * resident < latest("process.memory.usage") < 2 * resident
    assert latest("process.cpu.time", type="user") > 0
    assert latest("system.memory.usage", state="used") > 0


def _failed_sign_in(instance) -> str:
    email = f"nobody-{uuid.uuid4().hex[:8]}@example.com"
    with instance.client() as client:
        refused = client.post("/api/v1/auths/signin", json={"email": email, "password": "wrong"})
    assert refused.status_code == 400, refused.text
    return email


def _log_lines_naming(collector, email: str) -> list:
    return [record for record in collector.log_records() if email in str(record.body)]


def test_a_log_line_carries_the_trace_of_the_request_that_wrote_it(exporting, collector):
    offset = exporting.log_size()
    email = _failed_sign_in(exporting)

    record = collector.wait_for(lambda: _log_lines_naming(collector, email), "the log line")[0]

    def sign_in_span():
        spans = _server_spans(collector, "/api/v1/auths/signin")
        return [span for span in spans if span.trace_id == record.trace_id]

    [span] = collector.wait_for(sign_in_span, "a sign-in span on the log line's trace")
    assert span.attributes["http.status_code"] == 400
    [server_line] = [line for line in exporting.log_since(offset).splitlines() if email in line]
    assert f'"trace_id":"{record.trace_id}"' in server_line.replace('": "', '":"'), server_line


def _assert_exported_once(instance, collector) -> None:
    email = _failed_sign_in(instance)

    collector.wait_for(lambda: _log_lines_naming(collector, email), "the log line")
    time.sleep(2)  # ten batch intervals for a second copy to follow

    copies = _log_lines_naming(collector, email)
    assert len(copies) == 1, (
        f"the log line reached the collector {len(copies)} times: Open WebUI's log handler and "
        "the one the logging instrumentor adds to the root logger both export it"
    )


def test_each_log_line_is_exported_once(exporting, collector):
    _assert_exported_once(exporting, collector)


def test_each_log_line_is_exported_once_with_logs_alone(instance_with, collector):
    logs_only = instance_with(
        {
            **http_collector_env(collector),
            "ENABLE_OTEL_TRACES": "false",
            "ENABLE_OTEL_METRICS": "false",
        }
    )

    _assert_exported_once(logs_only, collector)


@pytest.fixture(scope="module")
def grpc_collector():
    with serving_grpc_collector() as served:
        yield served


@pytest.fixture(scope="module")
def redis_url():
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def chroma_url():
    with serving_chroma() as base_url:
        yield base_url


@pytest.fixture
def grpc_exporting(instance_with, grpc_collector, redis_url, chroma_url):
    """An instance on Redis and a Chroma server, exporting to the gRPC collector."""
    return instance_with(
        {
            **grpc_collector_env(grpc_collector, GRPC_USER, GRPC_PASSWORD),
            **chroma_env(chroma_url),
            "REDIS_URL": redis_url,
        }
    )


def test_the_grpc_exporters_send_every_signal_with_basic_auth(grpc_exporting, grpc_collector):
    with grpc_exporting.client() as client:
        client.get("/api/v1/auths/").raise_for_status()
    _failed_sign_in(grpc_exporting)

    grpc_collector.wait_for(
        lambda: _server_spans(grpc_collector, "/api/v1/auths/"), "a span over gRPC"
    )
    grpc_collector.wait_for(
        lambda: grpc_collector.metric_points("http.server.requests"), "a metric over gRPC"
    )
    grpc_collector.wait_for(lambda: grpc_collector.log_records(), "a log line over gRPC")
    basic = "Basic " + base64.b64encode(f"{GRPC_USER}:{GRPC_PASSWORD}".encode()).decode()
    assert {headers.get("authorization") for headers in grpc_collector.headers} == {basic}


def test_redis_and_chroma_calls_are_spans_named_by_the_hooks(
    grpc_exporting, grpc_collector, redis_url, chroma_url
):
    with admin_of(grpc_exporting).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "ferry.txt", "The ferry leaves every hour.")

    redis_host = redis_url.removeprefix("redis://").split(":")[0]
    redis_calls = grpc_collector.wait_for(
        lambda: [span for span in grpc_collector.spans() if span.scope == REDIS_SCOPE],
        "a Redis call",
    )
    stamped = [span for span in redis_calls if span.attributes.get("db.type") == "redis"]
    assert stamped, "no Redis span carries what the request hook sets"
    assert all(span.attributes["db.ip"] == redis_host for span in stamped)
    assert all(
        span.attributes["db.statement"].split()[0] == span.attributes["db.operation"]
        for span in stamped
    )
    chroma_calls = grpc_collector.wait_for(
        lambda: [
            span
            for span in grpc_collector.spans()
            if span.scope == HTTPX_SCOPE
            and span.name.startswith(("GET ", "POST ", "PUT "))
            and chroma_url in span.name
        ],
        "a call to the Chroma server",
    )
    assert all(span.attributes["http.url"] in span.name for span in chroma_calls)
    assert any(span.attributes["http.status_code"] == 200 for span in chroma_calls)
