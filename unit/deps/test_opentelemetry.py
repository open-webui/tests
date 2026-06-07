"""Dependency contract: the OpenTelemetry family.

Open WebUI ships optional distributed tracing, metrics and log export
built on OpenTelemetry. When enabled (`ENABLE_OTEL`), the backend:

  * builds a `TracerProvider`/`MeterProvider`/`LoggerProvider` from an SDK
    `Resource`, wires a `BatchSpanProcessor`/`PeriodicExportingMetricReader`
    and pushes to an OTLP collector (gRPC or HTTP) — see
    `open_webui/utils/telemetry/{setup,metrics,logs}.py`;
  * auto-patches the libraries it talks to via a stack of *instrumentors*
    (fastapi, sqlalchemy, redis, requests, httpx, aiohttp-client, logging,
    system-metrics) wrapped in a single `BaseInstrumentor` subclass — see
    `open_webui/utils/telemetry/instrumentors.py`;
  * reads the current span on every request to stamp user identity and to
    emit `trace_id`/`span_id` into logs — see `utils/auth.py`,
    `utils/logger.py`.

These twelve packages are released and version-locked together, but the
exact *dotted import paths* (especially exporters and the `_logs`/`_log_exporter`
private-looking modules) and the instrumentor hook kwargs drift between
releases. This module pins the slice of that API the codebase actually
imports and calls, so an OTel bump that moved or renamed any of it fails
loudly here instead of at process start-up behind the `ENABLE_OTEL` flag.

Pattern (see `test_requests.py`): symbol-existence checks per package +
offline behavioural contracts. Strictly offline — no collector, no network.
Any instrumentor we actually `.instrument()` is `.uninstrument()`-ed in a
`finally` so global monkey-patching never leaks to sibling test modules.
Uses the `depcheck` fixture from `unit/deps/conftest.py`.
"""

from __future__ import annotations

import inspect
import logging

import pytest

pytestmark = pytest.mark.depcheck


# ---------------------------------------------------------------------------
# Distribution presence / version reporting
# ---------------------------------------------------------------------------

# (import_name probe, distribution name). The import probe is a module the
# distribution provides; if it imports, the dist is installed.
OTEL_DISTRIBUTIONS = [
    ("opentelemetry.trace", "opentelemetry-api"),
    ("opentelemetry.sdk.trace", "opentelemetry-sdk"),
    ("opentelemetry.exporter.otlp.proto.grpc.trace_exporter", "opentelemetry-exporter-otlp"),
    ("opentelemetry.instrumentation.instrumentor", "opentelemetry-instrumentation"),
    ("opentelemetry.instrumentation.fastapi", "opentelemetry-instrumentation-fastapi"),
    ("opentelemetry.instrumentation.sqlalchemy", "opentelemetry-instrumentation-sqlalchemy"),
    ("opentelemetry.instrumentation.redis", "opentelemetry-instrumentation-redis"),
    ("opentelemetry.instrumentation.requests", "opentelemetry-instrumentation-requests"),
    ("opentelemetry.instrumentation.logging", "opentelemetry-instrumentation-logging"),
    ("opentelemetry.instrumentation.httpx", "opentelemetry-instrumentation-httpx"),
    (
        "opentelemetry.instrumentation.aiohttp_client",
        "opentelemetry-instrumentation-aiohttp-client",
    ),
    (
        "opentelemetry.instrumentation.system_metrics",
        "opentelemetry-instrumentation-system-metrics",
    ),
]


@pytest.mark.parametrize("import_name,dist_name", OTEL_DISTRIBUTIONS)
def test_distribution_importable_and_versioned(depcheck, import_name, dist_name):
    """Each of the 12 OTel packages imports and reports an installed version.

    Skips (does not fail) per package when that sub-package is absent, so the
    suite stays runnable in trimmed environments.
    """
    depcheck.load(import_name)
    assert depcheck.dist_version(dist_name) is not None, (
        f"{dist_name} importable but has no resolvable distribution version"
    )


# ===========================================================================
# opentelemetry-api  (opentelemetry.trace / .metrics / ._logs / .context)
# ===========================================================================

TRACE_SYMBOLS = [
    "set_tracer_provider",
    "get_tracer_provider",
    "get_tracer",
    "get_current_span",
    "format_trace_id",
    "format_span_id",
    "Span",
    "SpanContext",
    "StatusCode",
    "Status",
]

METRICS_SYMBOLS = [
    "set_meter_provider",
    "get_meter_provider",
    "get_meter",
    "CallbackOptions",
    "Observation",
]


def test_api_trace_symbols(depcheck):
    """`opentelemetry.trace` surface used by setup/auth/logger.

    NB: `trace` is a submodule, not an attribute of the top-level
    `opentelemetry` package until imported — the backend imports it directly
    (`from opentelemetry import trace`), so probe the submodule itself.
    """
    trace = depcheck.try_load("opentelemetry.trace")
    if trace is None:
        pytest.skip("opentelemetry.trace not importable")
    depcheck.assert_symbols(trace, TRACE_SYMBOLS)
    for name in ("set_tracer_provider", "get_tracer_provider", "get_tracer", "get_current_span"):
        depcheck.assert_callable(trace, name)


def test_api_trace_format_helpers_offline(depcheck):
    """logger.py emits `trace.format_trace_id`/`format_span_id` into logs;
    they must return fixed-width hex of the right length, no provider needed."""
    trace = depcheck.try_load("opentelemetry.trace")
    if trace is None:
        pytest.skip("opentelemetry.trace not importable")
    assert trace.format_trace_id(1) == "0" * 31 + "1"  # 128-bit -> 32 hex chars
    assert trace.format_span_id(1) == "0" * 15 + "1"  # 64-bit -> 16 hex chars


def test_api_get_current_span_default_offline(depcheck):
    """auth.py/logger.py call `trace.get_current_span()` unconditionally and
    read `.get_span_context()`/`.set_attribute(...)`; outside any span this
    must return a non-recording span (not raise) exposing that shape."""
    trace = depcheck.try_load("opentelemetry.trace")
    if trace is None:
        pytest.skip("opentelemetry.trace not importable")
    span = trace.get_current_span()
    assert span is not None
    ctx = span.get_span_context()
    # logger.py branches on `context.is_valid`; the field must exist.
    assert hasattr(ctx, "is_valid")
    assert hasattr(ctx, "trace_id") and hasattr(ctx, "span_id")
    # auth.py calls set_attribute on whatever get_current_span returns.
    assert callable(span.set_attribute)
    span.set_attribute("client.user.id", "x")  # no-op on non-recording span


def test_api_statuscode_members(depcheck):
    """instrumentors.py uses `StatusCode.ERROR`/`StatusCode.OK`; both members
    (and UNSET) must exist on the enum."""
    trace = depcheck.try_load("opentelemetry.trace")
    if trace is None:
        pytest.skip("opentelemetry.trace not importable")
    members = {m.name for m in trace.StatusCode}
    assert {"OK", "ERROR", "UNSET"} <= members


def test_api_span_hook_methods(depcheck):
    """The request/response hooks call span.update_name / set_attribute(s) /
    set_status on the `Span` they receive; pin those on the abstract Span."""
    trace = depcheck.try_load("opentelemetry.trace")
    if trace is None:
        pytest.skip("opentelemetry.trace not importable")
    for m in ("update_name", "set_attribute", "set_attributes", "set_status"):
        assert hasattr(trace.Span, m), f"Span.{m} missing (used by telemetry hooks)"


def test_api_metrics_symbols(depcheck):
    """`opentelemetry.metrics` surface used by metrics.py (provider + meter +
    observable-gauge callback types)."""
    metrics = depcheck.try_load("opentelemetry.metrics")
    if metrics is None:
        pytest.skip("opentelemetry.metrics not importable")
    depcheck.assert_symbols(metrics, METRICS_SYMBOLS)
    for name in ("set_meter_provider", "get_meter"):
        depcheck.assert_callable(metrics, name)


def test_api_logs_set_logger_provider(depcheck):
    """logs.py imports `set_logger_provider` from the `opentelemetry._logs`
    namespace; the module + symbol must exist (this path is unstable)."""
    logs = depcheck.try_load("opentelemetry._logs")
    if logs is None:
        pytest.skip("opentelemetry._logs not importable")
    depcheck.assert_symbols(logs, ["set_logger_provider"])
    depcheck.assert_callable(logs, "set_logger_provider")


def test_api_context_module_importable(depcheck):
    """`opentelemetry.context` underpins span/baggage propagation the SDK
    relies on; ensure the module is importable in this OTel build."""
    ctx = depcheck.try_load("opentelemetry.context")
    if ctx is None:
        pytest.skip("opentelemetry.context not importable")
    assert ctx.__name__ == "opentelemetry.context"


# ===========================================================================
# opentelemetry-semantic-conventions  (pulled by api; used in constants.py)
# ===========================================================================

# constants.py subclasses semconv SpanAttributes and the hooks read these.
SEMCONV_ATTRS = [
    "HTTP_URL",
    "HTTP_METHOD",
    "HTTP_STATUS_CODE",
    "DB_NAME",
    "DB_SYSTEM",
    "DB_STATEMENT",
    "DB_OPERATION",
]


def test_semconv_span_attributes(depcheck):
    """`opentelemetry.semconv.trace.SpanAttributes` provides the canonical
    attribute-key constants the redis/http/aiohttp hooks set on spans."""
    semconv = depcheck.try_load("opentelemetry.semconv.trace")
    if semconv is None:
        pytest.skip("opentelemetry.semconv.trace not importable")
    SpanAttributes = semconv.SpanAttributes
    # Class attributes are plain strings — reading them executes nothing.
    missing = [a for a in SEMCONV_ATTRS if not hasattr(SpanAttributes, a)]
    assert not missing, f"semconv SpanAttributes missing keys: {missing}"
    assert SpanAttributes.HTTP_METHOD == "http.method"
    assert SpanAttributes.HTTP_URL == "http.url"
    assert SpanAttributes.HTTP_STATUS_CODE == "http.status_code"


def test_semconv_subclassable_like_constants_module(depcheck):
    """constants.py does `class SpanAttributes(_SpanAttributes): ...` adding
    extra keys; the base must be a subclassable class, and the subclass must
    keep inherited keys alongside new ones (mirrors the real pattern)."""
    semconv = depcheck.try_load("opentelemetry.semconv.trace")
    if semconv is None:
        pytest.skip("opentelemetry.semconv.trace not importable")
    base = semconv.SpanAttributes
    assert inspect.isclass(base)

    class _Extended(base):
        DB_INSTANCE = "db.instance"
        DB_TYPE = "db.type"

    assert _Extended.DB_INSTANCE == "db.instance"
    assert _Extended.HTTP_METHOD == "http.method"  # inherited


# ===========================================================================
# opentelemetry-sdk  (trace / resources / metrics / _logs)
# ===========================================================================

SDK_TRACE_PATHS = {
    "opentelemetry.sdk.trace": ["TracerProvider"],
    "opentelemetry.sdk.trace.export": [
        "BatchSpanProcessor",
        "SimpleSpanProcessor",
        "SpanExporter",
    ],
    "opentelemetry.sdk.trace.export.in_memory_span_exporter": ["InMemorySpanExporter"],
    "opentelemetry.sdk.resources": ["Resource", "SERVICE_NAME"],
}

SDK_METRICS_PATHS = {
    "opentelemetry.sdk.metrics": ["MeterProvider"],
    "opentelemetry.sdk.metrics.export": ["PeriodicExportingMetricReader"],
    "opentelemetry.sdk.metrics.view": ["View"],
}

SDK_LOGS_PATHS = {
    "opentelemetry.sdk._logs": ["LoggerProvider", "LoggingHandler"],
    "opentelemetry.sdk._logs.export": ["BatchLogRecordProcessor"],
}


@pytest.mark.parametrize("modpath,symbols", sorted(SDK_TRACE_PATHS.items()))
def test_sdk_trace_symbols(depcheck, modpath, symbols):
    mod = depcheck.try_load(modpath)
    if mod is None:
        pytest.skip(f"{modpath} not importable")
    depcheck.assert_symbols(mod, symbols)


@pytest.mark.parametrize("modpath,symbols", sorted(SDK_METRICS_PATHS.items()))
def test_sdk_metrics_symbols(depcheck, modpath, symbols):
    mod = depcheck.try_load(modpath)
    if mod is None:
        pytest.skip(f"{modpath} not importable")
    depcheck.assert_symbols(mod, symbols)


@pytest.mark.parametrize("modpath,symbols", sorted(SDK_LOGS_PATHS.items()))
def test_sdk_logs_symbols(depcheck, modpath, symbols):
    mod = depcheck.try_load(modpath)
    if mod is None:
        pytest.skip(f"{modpath} not importable")
    depcheck.assert_symbols(mod, symbols)


def test_sdk_tracerprovider_constructor_params(depcheck):
    """setup.py builds `TracerProvider(resource=...)`; the `resource` kwarg
    must remain accepted."""
    sdk_trace = depcheck.try_load("opentelemetry.sdk.trace")
    if sdk_trace is None:
        pytest.skip("opentelemetry.sdk.trace not importable")
    depcheck.assert_params(sdk_trace.TracerProvider.__init__, ["resource"])


def test_sdk_tracerprovider_add_span_processor(depcheck):
    """setup.py/logs.py call `provider.add_span_processor(...)` /
    `add_log_record_processor(...)`; the method must exist on the provider."""
    sdk_trace = depcheck.try_load("opentelemetry.sdk.trace")
    if sdk_trace is None:
        pytest.skip("opentelemetry.sdk.trace not importable")
    assert hasattr(sdk_trace.TracerProvider, "add_span_processor")
    assert hasattr(sdk_trace.TracerProvider, "get_tracer")
    assert hasattr(sdk_trace.TracerProvider, "shutdown")


def test_sdk_batchspanprocessor_constructor_params(depcheck):
    """setup.py wraps the exporter in `BatchSpanProcessor(exporter)` — first
    positional must be the span exporter."""
    exp = depcheck.try_load("opentelemetry.sdk.trace.export")
    if exp is None:
        pytest.skip("opentelemetry.sdk.trace.export not importable")
    depcheck.assert_params(exp.BatchSpanProcessor.__init__, ["span_exporter"])


def test_sdk_resource_create_params(depcheck):
    """setup.py/logs.py/metrics.py call `Resource.create(attributes={...})`."""
    res = depcheck.try_load("opentelemetry.sdk.resources")
    if res is None:
        pytest.skip("opentelemetry.sdk.resources not importable")
    depcheck.assert_params(res.Resource.create, ["attributes"])
    assert res.SERVICE_NAME == "service.name"


def test_sdk_meterprovider_constructor_params(depcheck):
    """metrics.py builds `MeterProvider(resource=, metric_readers=, views=)`."""
    sdk_metrics = depcheck.try_load("opentelemetry.sdk.metrics")
    if sdk_metrics is None:
        pytest.skip("opentelemetry.sdk.metrics not importable")
    depcheck.assert_params(
        sdk_metrics.MeterProvider.__init__,
        ["resource", "metric_readers", "views"],
    )


def test_sdk_periodic_reader_constructor_params(depcheck):
    """metrics.py builds `PeriodicExportingMetricReader(exporter,
    export_interval_millis=...)`."""
    exp = depcheck.try_load("opentelemetry.sdk.metrics.export")
    if exp is None:
        pytest.skip("opentelemetry.sdk.metrics.export not importable")
    depcheck.assert_params(
        exp.PeriodicExportingMetricReader.__init__,
        ["exporter", "export_interval_millis"],
    )


def test_sdk_view_constructor_params(depcheck):
    """metrics.py builds `View(instrument_name=..., attribute_keys=...)`."""
    view_mod = depcheck.try_load("opentelemetry.sdk.metrics.view")
    if view_mod is None:
        pytest.skip("opentelemetry.sdk.metrics.view not importable")
    depcheck.assert_params(
        view_mod.View.__init__,
        ["instrument_name", "attribute_keys"],
    )


def test_sdk_loggerprovider_and_handler(depcheck):
    """logs.py builds `LoggerProvider(resource=)`, registers a processor, and
    wraps it in `LoggingHandler(logger_provider=)` which must be a stdlib
    `logging.Handler` so it can be attached to loguru/std logging."""
    sdk_logs = depcheck.try_load("opentelemetry.sdk._logs")
    if sdk_logs is None:
        pytest.skip("opentelemetry.sdk._logs not importable")
    depcheck.assert_params(sdk_logs.LoggerProvider.__init__, ["resource"])
    assert hasattr(sdk_logs.LoggerProvider, "add_log_record_processor")
    depcheck.assert_params(sdk_logs.LoggingHandler.__init__, ["logger_provider"])
    assert issubclass(sdk_logs.LoggingHandler, logging.Handler)


# ---- SDK behavioural contract: in-memory span export (no network) ----------


def test_sdk_in_memory_span_export_roundtrip(depcheck):
    """End-to-end offline trace contract mirroring setup.py wiring:
    Resource -> TracerProvider -> SimpleSpanProcessor -> InMemorySpanExporter.
    Create a span with the attributes the hooks set, and assert it is exported
    with the expected name/attributes/resource. No collector, no network."""
    sdk_trace = depcheck.try_load("opentelemetry.sdk.trace")
    exp_mod = depcheck.try_load("opentelemetry.sdk.trace.export")
    inmem_mod = depcheck.try_load("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    res_mod = depcheck.try_load("opentelemetry.sdk.resources")
    trace = depcheck.try_load("opentelemetry.trace")
    if not all((sdk_trace, exp_mod, inmem_mod, res_mod, trace)):
        pytest.skip("opentelemetry SDK trace stack not fully importable")

    exporter = inmem_mod.InMemorySpanExporter()
    resource = res_mod.Resource.create({res_mod.SERVICE_NAME: "owui-test"})
    provider = sdk_trace.TracerProvider(resource=resource)
    provider.add_span_processor(exp_mod.SimpleSpanProcessor(exporter))
    try:
        tracer = provider.get_tracer("owui.deps.test")
        with tracer.start_as_current_span("GET /probe") as span:
            span.set_attribute("http.method", "GET")
            span.set_attribute("http.status_code", 200)
            span.set_status(trace.StatusCode.OK)
        spans = exporter.get_finished_spans()
    finally:
        provider.shutdown()

    assert len(spans) == 1
    s = spans[0]
    assert s.name == "GET /probe"
    assert s.attributes["http.method"] == "GET"
    assert s.attributes["http.status_code"] == 200
    assert s.resource.attributes[res_mod.SERVICE_NAME] == "owui-test"


def test_sdk_span_update_name_offline(depcheck):
    """The request hooks call `span.update_name(...)`; verify a live recording
    span honours it (the exported span carries the updated name)."""
    sdk_trace = depcheck.try_load("opentelemetry.sdk.trace")
    exp_mod = depcheck.try_load("opentelemetry.sdk.trace.export")
    inmem_mod = depcheck.try_load("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    if not all((sdk_trace, exp_mod, inmem_mod)):
        pytest.skip("opentelemetry SDK trace stack not fully importable")

    exporter = inmem_mod.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(exp_mod.SimpleSpanProcessor(exporter))
    try:
        tracer = provider.get_tracer("owui.deps.test")
        with tracer.start_as_current_span("original") as span:
            span.update_name("renamed")
        spans = exporter.get_finished_spans()
    finally:
        provider.shutdown()

    assert spans[0].name == "renamed"


def test_sdk_resource_build_offline(depcheck):
    """`Resource.create({SERVICE_NAME: ...})` must yield a Resource whose
    `.attributes` mapping carries the service name (used everywhere)."""
    res_mod = depcheck.try_load("opentelemetry.sdk.resources")
    if res_mod is None:
        pytest.skip("opentelemetry.sdk.resources not importable")
    resource = res_mod.Resource.create({res_mod.SERVICE_NAME: "svc"})
    assert resource.attributes[res_mod.SERVICE_NAME] == "svc"


def test_sdk_in_memory_metric_collection_offline(depcheck):
    """Offline metrics contract mirroring metrics.py: MeterProvider with an
    in-memory reader, create a counter, record, and confirm collection works
    without a collector. Uses InMemoryMetricReader if present (skip if not)."""
    sdk_metrics = depcheck.try_load("opentelemetry.sdk.metrics")
    exp_mod = depcheck.try_load("opentelemetry.sdk.metrics.export")
    res_mod = depcheck.try_load("opentelemetry.sdk.resources")
    if not all((sdk_metrics, exp_mod, res_mod)):
        pytest.skip("opentelemetry SDK metrics stack not fully importable")
    if not hasattr(exp_mod, "InMemoryMetricReader"):
        pytest.skip("InMemoryMetricReader not available in this SDK build")

    reader = exp_mod.InMemoryMetricReader()
    provider = sdk_metrics.MeterProvider(
        resource=res_mod.Resource.create({res_mod.SERVICE_NAME: "m"}),
        metric_readers=[reader],
    )
    try:
        meter = provider.get_meter("owui.deps.test")
        counter = meter.create_counter("http.server.requests", unit="1")
        counter.add(1, {"http.method": "GET"})
        # create_histogram / create_observable_gauge are also used by metrics.py.
        assert callable(meter.create_histogram)
        assert callable(meter.create_observable_gauge)
        data = reader.get_metrics_data()
    finally:
        provider.shutdown()
    assert data is not None


# ===========================================================================
# opentelemetry-exporter-otlp  (grpc + http: trace / metric / log exporters)
# ===========================================================================

OTLP_EXPORTER_PATHS = {
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": "OTLPSpanExporter",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter": "OTLPSpanExporter",
    "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": "OTLPMetricExporter",
    "opentelemetry.exporter.otlp.proto.http.metric_exporter": "OTLPMetricExporter",
    "opentelemetry.exporter.otlp.proto.grpc._log_exporter": "OTLPLogExporter",
    "opentelemetry.exporter.otlp.proto.http._log_exporter": "OTLPLogExporter",
}


@pytest.mark.parametrize("modpath,symbol", sorted(OTLP_EXPORTER_PATHS.items()))
def test_otlp_exporter_symbols(depcheck, modpath, symbol):
    """Every OTLP exporter class the backend imports must exist at its exact
    dotted path. These paths (esp. the `_log_exporter` private modules and
    the grpc/http split) move between OTel releases — high-value pins."""
    mod = depcheck.try_load(modpath)
    if mod is None:
        pytest.skip(f"{modpath} not importable")
    depcheck.assert_symbols(mod, [symbol])
    depcheck.assert_callable(mod, symbol)


def test_otlp_grpc_span_exporter_params(depcheck):
    """setup.py: `OTLPSpanExporter(endpoint=, insecure=, headers=)` (grpc)."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    if mod is None:
        pytest.skip("grpc trace_exporter not importable")
    depcheck.assert_params(mod.OTLPSpanExporter.__init__, ["endpoint", "insecure", "headers"])


def test_otlp_http_span_exporter_params(depcheck):
    """setup.py: `HttpOTLPSpanExporter(endpoint=, headers=)` (http has no
    `insecure` kwarg — TLS is implied by the URL scheme)."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    if mod is None:
        pytest.skip("http trace_exporter not importable")
    depcheck.assert_params(mod.OTLPSpanExporter.__init__, ["endpoint", "headers"])


def test_otlp_grpc_metric_exporter_params(depcheck):
    """metrics.py: `OTLPMetricExporter(endpoint=, insecure=, headers=)`."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.grpc.metric_exporter")
    if mod is None:
        pytest.skip("grpc metric_exporter not importable")
    depcheck.assert_params(mod.OTLPMetricExporter.__init__, ["endpoint", "insecure", "headers"])


def test_otlp_http_metric_exporter_params(depcheck):
    """metrics.py: `OTLPHttpMetricExporter(endpoint=, headers=)`."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.http.metric_exporter")
    if mod is None:
        pytest.skip("http metric_exporter not importable")
    depcheck.assert_params(mod.OTLPMetricExporter.__init__, ["endpoint", "headers"])


def test_otlp_grpc_log_exporter_params(depcheck):
    """logs.py: `OTLPLogExporter(endpoint=, insecure=, headers=)`."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.grpc._log_exporter")
    if mod is None:
        pytest.skip("grpc _log_exporter not importable")
    depcheck.assert_params(mod.OTLPLogExporter.__init__, ["endpoint", "insecure", "headers"])


def test_otlp_http_log_exporter_params(depcheck):
    """logs.py: `HttpOTLPLogExporter(endpoint=, headers=)`."""
    mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.http._log_exporter")
    if mod is None:
        pytest.skip("http _log_exporter not importable")
    depcheck.assert_params(mod.OTLPLogExporter.__init__, ["endpoint", "headers"])


def test_otlp_exporters_construct_offline(depcheck):
    """Constructing an OTLP exporter must NOT open a connection (export is
    lazy). Build grpc + http span exporters at dummy endpoints with the exact
    kwargs setup.py uses, then `shutdown()` — no network, no collector."""
    grpc_mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    http_mod = depcheck.try_load("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    if grpc_mod is None and http_mod is None:
        pytest.skip("no OTLP span exporter importable")

    if grpc_mod is not None:
        g = grpc_mod.OTLPSpanExporter(
            endpoint="http://localhost:4317",
            insecure=True,
            headers=[("authorization", "Basic dGVzdA==")],
        )
        try:
            assert hasattr(g, "export") and hasattr(g, "shutdown")
        finally:
            g.shutdown()

    if http_mod is not None:
        h = http_mod.OTLPSpanExporter(
            endpoint="http://localhost:4318/v1/traces",
            headers={"authorization": "Basic dGVzdA=="},
        )
        try:
            assert hasattr(h, "export") and hasattr(h, "shutdown")
        finally:
            h.shutdown()


# ===========================================================================
# opentelemetry-instrumentation  (BaseInstrumentor)
# ===========================================================================


def test_base_instrumentor_contract(depcheck):
    """instrumentors.py subclasses `BaseInstrumentor` and implements
    `_instrument`/`_uninstrument`/`instrumentation_dependencies`. Pin the
    abstract surface plus the public `instrument`/`uninstrument` wrappers and
    the `is_instrumented_by_opentelemetry` guard flag the code relies on."""
    mod = depcheck.try_load("opentelemetry.instrumentation.instrumentor")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.instrumentor not importable")
    base = mod.BaseInstrumentor
    for m in (
        "instrument",
        "uninstrument",
        "_instrument",
        "_uninstrument",
        "instrumentation_dependencies",
    ):
        assert hasattr(base, m), f"BaseInstrumentor.{m} missing"
    assert callable(base.instrument)
    assert callable(base.uninstrument)


def test_base_instrumentor_is_singleton(depcheck):
    """BaseInstrumentor subclasses are singletons — instrumentors.py news up
    e.g. `RequestsInstrumentor()` in several places expecting shared global
    state. Verify the singleton invariant on a concrete instrumentor."""
    mod = depcheck.try_load("opentelemetry.instrumentation.requests")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.requests not importable")
    a = mod.RequestsInstrumentor()
    b = mod.RequestsInstrumentor()
    assert a is b, "instrumentor is no longer a singleton"
    assert hasattr(a, "is_instrumented_by_opentelemetry")


# ---------------------------------------------------------------------------
# Per-instrumentor symbol + class-shape checks.
#
# The public `instrument`/`uninstrument` are `(self, **kwargs)` wrappers on
# every instrumentor, so a parameter-name signature check is meaningless
# (it would short-circuit on **kwargs). We instead assert each Instrumentor
# class exists, subclasses BaseInstrumentor, is constructible, and exposes
# callable instrument/uninstrument. Hook kwargs are validated behaviourally
# below where it's safe and offline.
# ---------------------------------------------------------------------------

# (module path, class name) for the 8 instrumentors instrumentors.py uses.
INSTRUMENTOR_CLASSES = [
    ("opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor"),
    ("opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor"),
    ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
    ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ("opentelemetry.instrumentation.logging", "LoggingInstrumentor"),
    ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
    ("opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"),
    ("opentelemetry.instrumentation.system_metrics", "SystemMetricsInstrumentor"),
]


@pytest.mark.parametrize("modpath,clsname", INSTRUMENTOR_CLASSES)
def test_instrumentor_class_shape(depcheck, modpath, clsname):
    """Each instrumentor class must exist, subclass BaseInstrumentor, be
    constructible, and expose callable instrument()/uninstrument()."""
    mod = depcheck.try_load(modpath)
    if mod is None:
        pytest.skip(f"{modpath} not importable")
    base_mod = depcheck.try_load("opentelemetry.instrumentation.instrumentor")
    cls = getattr(mod, clsname, None)
    assert cls is not None, f"{clsname} missing from {modpath}"
    if base_mod is not None:
        assert issubclass(cls, base_mod.BaseInstrumentor), (
            f"{clsname} no longer subclasses BaseInstrumentor"
        )
    inst = cls()  # singleton; constructible without args
    assert callable(inst.instrument)
    assert callable(inst.uninstrument)


# ===========================================================================
# opentelemetry-instrumentation-fastapi
# ===========================================================================


def test_fastapi_instrumentor_instrument_app(depcheck):
    """instrumentors.py uses the classmethod-style
    `FastAPIInstrumentor.instrument_app(app=...)` (not the instance
    `instrument()`); the method + its `app` param must exist."""
    mod = depcheck.try_load("opentelemetry.instrumentation.fastapi")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.fastapi not importable")
    assert hasattr(mod.FastAPIInstrumentor, "instrument_app")
    depcheck.assert_params(mod.FastAPIInstrumentor.instrument_app, ["app"])


# ===========================================================================
# opentelemetry-instrumentation-sqlalchemy
# ===========================================================================


def test_sqlalchemy_instrumentor_engine_kwarg(depcheck):
    """instrumentors.py calls `SQLAlchemyInstrumentor().instrument(engine=...)`.
    The public wrapper is `(self, **kwargs)`, so verify the underlying
    `_instrument` accepts the `engine` kwarg it actually forwards."""
    mod = depcheck.try_load("opentelemetry.instrumentation.sqlalchemy")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.sqlalchemy not importable")
    depcheck.assert_params(mod.SQLAlchemyInstrumentor._instrument, ["engine"])


# ===========================================================================
# opentelemetry-instrumentation-redis
# ===========================================================================


def test_redis_instrumentor_request_hook_kwarg(depcheck):
    """instrumentors.py calls
    `RedisInstrumentor().instrument(request_hook=redis_request_hook)`. The
    RedisInstrumentor's instrument exposes an explicit `request_hook` param."""
    mod = depcheck.try_load("opentelemetry.instrumentation.redis")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.redis not importable")
    # RedisInstrumentor.instrument has explicit request_hook/response_hook
    # params in current builds; fall back to _instrument if the wrapper is
    # the generic **kwargs form.
    target = mod.RedisInstrumentor.instrument
    sig = inspect.signature(target)
    if "request_hook" not in sig.parameters and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        target = mod.RedisInstrumentor._instrument
    depcheck.assert_params(target, ["request_hook"])


# ===========================================================================
# opentelemetry-instrumentation-requests
# ===========================================================================


def test_requests_instrumentor_hook_kwargs(depcheck):
    """instrumentors.py calls `RequestsInstrumentor().instrument(
    request_hook=requests_hook, response_hook=response_hook)`. Verify
    `_instrument` (what the **kwargs wrapper forwards to) accepts both."""
    mod = depcheck.try_load("opentelemetry.instrumentation.requests")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.requests not importable")
    depcheck.assert_params(
        mod.RequestsInstrumentor._instrument,
        ["request_hook", "response_hook"],
    )


# ===========================================================================
# opentelemetry-instrumentation-logging
# ===========================================================================


def test_logging_instrumentor_instrument_roundtrip(depcheck):
    """instrumentors.py calls `LoggingInstrumentor().instrument()` with no
    kwargs. This one is safe to actually exercise offline — it patches the
    stdlib logging factory, not a network library. ALWAYS uninstrument in a
    finally so the patch never leaks to sibling test modules."""
    mod = depcheck.try_load("opentelemetry.instrumentation.logging")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.logging not importable")
    inst = mod.LoggingInstrumentor()
    already = inst.is_instrumented_by_opentelemetry
    if already:
        # Another concurrently-running test file may have left it on; don't
        # fight over global state — just assert the API shape and bail.
        assert callable(inst.instrument) and callable(inst.uninstrument)
        return
    try:
        inst.instrument()
        assert inst.is_instrumented_by_opentelemetry is True
    finally:
        inst.uninstrument()
    assert inst.is_instrumented_by_opentelemetry is False


# ===========================================================================
# opentelemetry-instrumentation-httpx  (+ RequestInfo / ResponseInfo)
# ===========================================================================


def test_httpx_instrumentor_request_response_info(depcheck):
    """instrumentors.py imports `RequestInfo`/`ResponseInfo` from the httpx
    instrumentation and reads `.method`/`.url` (RequestInfo) and
    `.status_code` (ResponseInfo) inside the hooks. Pin those fields."""
    mod = depcheck.try_load("opentelemetry.instrumentation.httpx")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.httpx not importable")
    depcheck.assert_symbols(mod, ["HTTPXClientInstrumentor", "RequestInfo", "ResponseInfo"])
    req_fields = getattr(mod.RequestInfo, "_fields", None)
    resp_fields = getattr(mod.ResponseInfo, "_fields", None)
    if req_fields is not None:
        assert "method" in req_fields and "url" in req_fields
    if resp_fields is not None:
        assert "status_code" in resp_fields


def test_httpx_instrumentor_hook_kwargs(depcheck):
    """instrumentors.py passes request_hook/response_hook/async_request_hook/
    async_response_hook. Verify `_instrument` accepts the async hook kwargs
    (the distinguishing modern surface)."""
    mod = depcheck.try_load("opentelemetry.instrumentation.httpx")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.httpx not importable")
    depcheck.assert_params(
        mod.HTTPXClientInstrumentor._instrument,
        ["request_hook", "response_hook", "async_request_hook", "async_response_hook"],
    )


# ===========================================================================
# opentelemetry-instrumentation-aiohttp-client
# ===========================================================================


def test_aiohttp_client_instrumentor_hook_kwargs(depcheck):
    """instrumentors.py calls `AioHttpClientInstrumentor().instrument(
    request_hook=..., response_hook=...)`."""
    mod = depcheck.try_load("opentelemetry.instrumentation.aiohttp_client")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.aiohttp_client not importable")
    depcheck.assert_params(
        mod.AioHttpClientInstrumentor._instrument,
        ["request_hook", "response_hook"],
    )


# ===========================================================================
# opentelemetry-instrumentation-system-metrics
# ===========================================================================


def test_system_metrics_instrumentor_constructible(depcheck):
    """instrumentors.py calls `SystemMetricsInstrumentor().instrument()` with
    no kwargs. Verify the class is constructible and exposes the wrappers
    (do NOT actually instrument — it starts a background metrics collector)."""
    mod = depcheck.try_load("opentelemetry.instrumentation.system_metrics")
    if mod is None:
        pytest.skip("opentelemetry.instrumentation.system_metrics not importable")
    inst = mod.SystemMetricsInstrumentor()
    assert callable(inst.instrument)
    assert callable(inst.uninstrument)
