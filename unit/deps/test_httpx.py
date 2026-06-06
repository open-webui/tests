"""Dependency contract: httpx.

httpx is Open WebUI's modern HTTP client for both sync and async outbound
calls. The backend uses it directly to build the MCP tool-server transport
(`open_webui/utils/mcp/client.py` constructs `httpx.AsyncClient(...)` with
`follow_redirects`, `verify`, `timeout`, `headers`, `auth`), and it is the
HTTP engine underneath several first-class dependencies the backend drives
(the OpenAI SDK, the MCP SDK's `streamablehttp_client`, the OpenTelemetry
`HTTPXClientInstrumentor`). A bump that renamed a client kwarg, reshaped the
`Response` API, or reorganised the exception tree would otherwise surface as
an AttributeError / TypeError deep inside a provider call at runtime.

This module pins the slice of the httpx API the codebase relies on:
  - the public symbols (Client, AsyncClient, Timeout, Limits, Response,
    Request, MockTransport, and the exception classes);
  - the constructor / method parameter names actually passed;
  - offline behavioural contracts driven entirely through
    `httpx.MockTransport` so nothing here touches the network.

Follows the `unit/deps/` exemplar (`test_requests.py`): symbol-existence
checks for the API surface plus offline behavioural contracts, all via the
`depcheck` fixture from `unit/deps/conftest.py`.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "httpx"
DIST_NAME = "httpx"

# Top-level symbols the Open WebUI backend (and the SDKs it drives over
# httpx) reference. Construction classes + the response/request model +
# the transport hooks used to test offline.
USED_SYMBOLS = [
    "Client",
    "AsyncClient",
    "Timeout",
    "Limits",
    "Response",
    "Request",
    "URL",
    "Headers",
    "BaseTransport",
    "AsyncBaseTransport",
    "HTTPTransport",
    "AsyncHTTPTransport",
    "MockTransport",
    "ASGITransport",
    "stream",
]

# Exception classes the codebase / its HTTP-using deps catch. The backend
# and the OpenAI/MCP SDKs handle these; the tree must stay intact so broad
# `except httpx.HTTPError` / `except httpx.RequestError` handlers keep
# catching what they expect.
USED_EXCEPTIONS = [
    "HTTPError",
    "RequestError",
    "HTTPStatusError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "TimeoutException",
    "TransportError",
    "NetworkError",
    "ReadError",
    "WriteError",
    "ProxyError",
    "DecodingError",
    "TooManyRedirects",
    "RemoteProtocolError",
    "InvalidURL",
    "UnsupportedProtocol",
    "StreamError",
]


# --------------------------------------------------------------------------
# helpers (local only — no cross-test imports)
# --------------------------------------------------------------------------
def _ok_handler(payload=None, status_code=200, headers=None):
    """Build a MockTransport handler returning a fixed JSON response.

    The handler is what makes every behavioural test below offline: httpx
    routes the request through it instead of opening a socket.
    """
    body = {"ok": True} if payload is None else payload

    def handler(request):
        return _httpx().Response(status_code, json=body, headers=headers or {})

    return handler


def _httpx():
    """Import httpx or skip — mirrors depcheck.load for the module helpers."""
    import importlib

    try:
        return importlib.import_module(IMPORT_NAME)
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"{IMPORT_NAME!r} not importable in this env: {e}")


# --------------------------------------------------------------------------
# import / version
# --------------------------------------------------------------------------
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "httpx"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version resolves, so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_attribute_present(depcheck):
    """The backend / tooling reads httpx.__version__; keep it exposed."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    assert mod.__version__  # non-empty


# --------------------------------------------------------------------------
# symbol existence
# --------------------------------------------------------------------------
def test_used_symbols_exist(depcheck):
    """Every top-level httpx symbol the codebase relies on must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_used_exceptions_exist(depcheck):
    """Every httpx exception the codebase / its deps catch must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_EXCEPTIONS)


def test_client_and_asyncclient_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Client)
    assert inspect.isclass(mod.AsyncClient)


def test_construction_helpers_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("Timeout", "Limits", "Response", "Request", "MockTransport"):
        assert inspect.isclass(getattr(mod, name)), f"httpx.{name} is not a class"


def test_module_level_request_helpers_callable(depcheck):
    """httpx exposes module-level request/get/post/stream convenience fns;
    keep them callable for any ad-hoc one-shot use."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("request", "get", "post", "put", "patch", "delete", "head", "stream"):
        if depcheck.has(mod, name):
            depcheck.assert_callable(mod, name)


# --------------------------------------------------------------------------
# Client / AsyncClient constructor signatures
# --------------------------------------------------------------------------
# Kwargs the backend passes (mcp/client.py) plus the broader set the OpenAI /
# MCP SDKs pass when they build clients over httpx.
_CLIENT_INIT_KWARGS = [
    "auth",
    "params",
    "headers",
    "cookies",
    "verify",
    "cert",
    "http1",
    "http2",
    "proxy",
    "mounts",
    "timeout",
    "follow_redirects",
    "limits",
    "max_redirects",
    "event_hooks",
    "base_url",
    "transport",
    "trust_env",
]


def test_client_init_accepts_used_kwargs(depcheck):
    """httpx.Client(base_url=, headers=, timeout=, verify=,
    follow_redirects=, http2=, limits=, transport=, auth=, ...) — the kwargs
    the backend and its httpx-backed deps pass must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Client.__init__, _CLIENT_INIT_KWARGS)


def test_asyncclient_init_accepts_used_kwargs(depcheck):
    """mcp/client.py builds httpx.AsyncClient(follow_redirects=, verify=,
    timeout=, headers=, auth=). Assert AsyncClient.__init__ accepts those
    (and the wider SDK-driven set)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncClient.__init__, _CLIENT_INIT_KWARGS)


def test_asyncclient_init_accepts_backend_kwargs_exactly(depcheck):
    """Narrow guard on the exact kwargs open_webui/utils/mcp/client.py
    passes, so a rename of any one of them fails this test by name."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.AsyncClient.__init__,
        ["follow_redirects", "verify", "timeout", "headers", "auth"],
    )


def test_client_init_uses_proxy_not_proxies(depcheck):
    """httpx renamed `proxies` -> `proxy` (and dropped `proxies` in 0.28).
    The codebase must use `proxy`; assert the modern name is the one present
    so we don't silently regress to the removed kwarg."""
    mod = depcheck.load(IMPORT_NAME)
    params = inspect.signature(mod.Client.__init__).parameters
    assert "proxy" in params, "httpx.Client no longer accepts `proxy`"


# --------------------------------------------------------------------------
# client request-method signatures
# --------------------------------------------------------------------------
def test_client_verb_methods_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for verb in ("request", "get", "post", "put", "patch", "delete", "head", "stream"):
        assert callable(getattr(mod.Client, verb)), f"Client.{verb} missing"


def test_asyncclient_verb_methods_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for verb in ("request", "get", "post", "put", "patch", "delete", "head", "stream"):
        assert callable(getattr(mod.AsyncClient, verb)), f"AsyncClient.{verb} missing"


def test_client_request_signature(depcheck):
    """Client.request(method, url, headers=, json=, content=, data=, params=,
    timeout=, follow_redirects=, auth=) — the kwargs request paths use."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Client.request,
        [
            "method",
            "url",
            "headers",
            "json",
            "content",
            "data",
            "params",
            "timeout",
            "follow_redirects",
            "auth",
        ],
    )


def test_client_get_signature(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Client.get,
        ["url", "params", "headers", "auth", "follow_redirects", "timeout"],
    )


def test_client_post_signature(depcheck):
    """post is used with json=/content=/data=/headers=; pin those kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Client.post,
        ["url", "content", "data", "json", "params", "headers", "timeout"],
    )


def test_asyncclient_post_signature(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.AsyncClient.post,
        ["url", "content", "data", "json", "params", "headers", "timeout"],
    )


def test_stream_signature(depcheck):
    """Client.stream(method, url, ...) returns a streaming context manager
    used to drive iter_bytes/aiter_bytes; pin its leading params."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Client.stream, ["method", "url", "headers", "timeout"])
    depcheck.assert_params(mod.AsyncClient.stream, ["method", "url", "headers", "timeout"])


# --------------------------------------------------------------------------
# Timeout / Limits construction
# --------------------------------------------------------------------------
def test_timeout_signature(depcheck):
    """httpx.Timeout(timeout, connect=, read=, write=, pool=) — config layer
    builds Timeout objects; pin the param names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Timeout.__init__, ["timeout", "connect", "read", "write", "pool"])


def test_timeout_constructs_scalar(depcheck):
    """Timeout(5.0) — a single scalar fans out to all four phases."""
    mod = depcheck.load(IMPORT_NAME)
    t = mod.Timeout(5.0)
    assert t.connect == 5.0
    assert t.read == 5.0
    assert t.write == 5.0
    assert t.pool == 5.0


def test_timeout_constructs_per_phase(depcheck):
    """Timeout(timeout, connect=, read=) — per-phase overrides hold."""
    mod = depcheck.load(IMPORT_NAME)
    t = mod.Timeout(10.0, connect=2.0, read=3.0)
    assert t.connect == 2.0
    assert t.read == 3.0


def test_timeout_accepts_none(depcheck):
    """Timeout(None) (disable timeout) must remain constructible."""
    mod = depcheck.load(IMPORT_NAME)
    t = mod.Timeout(None)
    assert t.connect is None


def test_limits_signature(depcheck):
    """httpx.Limits(max_connections=, max_keepalive_connections=,
    keepalive_expiry=) — connection-pool tuning param names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Limits.__init__,
        ["max_connections", "max_keepalive_connections", "keepalive_expiry"],
    )


def test_limits_constructs(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    limits = mod.Limits(max_connections=100, max_keepalive_connections=20)
    assert limits.max_connections == 100
    assert limits.max_keepalive_connections == 20


def test_client_accepts_timeout_and_limits_objects(depcheck):
    """A Client must accept Timeout/Limits instances (not just floats),
    exercised offline via MockTransport."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(
        transport=mod.MockTransport(_ok_handler()),
        timeout=mod.Timeout(5.0, connect=1.0),
        limits=mod.Limits(max_connections=10),
        base_url="https://api.example.test",
    )
    try:
        r = client.get("/ping")
        assert r.status_code == 200
    finally:
        client.close()


# --------------------------------------------------------------------------
# Response contract
# --------------------------------------------------------------------------
def test_response_attribute_surface(depcheck):
    """Responses are consumed via .status_code/.headers/.text/.content/
    .json()/.raise_for_status()/.iter_bytes()/.url; pin that shape.

    Use dir() on a constructed instance so property getters aren't executed
    (per the exemplar's note about not triggering descriptors)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Response(200)))
    for attr in (
        "status_code",
        "headers",
        "text",
        "content",
        "json",
        "raise_for_status",
        "iter_bytes",
        "aiter_bytes",
        "iter_text",
        "iter_lines",
        "url",
        "request",
        "is_success",
        "is_error",
        "encoding",
        "stream",
    ):
        assert attr in names, f"Response.{attr} missing"


def test_response_methods_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("json", "raise_for_status", "iter_bytes", "aiter_bytes"):
        assert callable(getattr(mod.Response, name)), f"Response.{name} not callable"


def test_response_basic_construction(depcheck):
    """httpx.Response(status_code, json=, headers=) is how handlers and tests
    build responses; pin that constructor shape."""
    mod = depcheck.load(IMPORT_NAME)
    resp = mod.Response(200, json={"a": 1}, headers={"X-H": "v"})
    assert resp.status_code == 200
    assert resp.json() == {"a": 1}
    assert resp.headers["X-H"] == "v"
    assert isinstance(resp.text, str)
    assert isinstance(resp.content, bytes)


def test_response_raise_for_status_returns_self_on_2xx(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    # raise_for_status needs an attached request (else httpx raises RuntimeError),
    # so build the Response with one — mirrors how a real client populates it.
    resp = mod.Response(204, request=mod.Request("GET", "https://api.example.test/"))
    # On success raise_for_status must not raise (returns the response).
    assert resp.raise_for_status() is resp


def test_response_is_success_flags(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.Response(200).is_success is True
    assert mod.Response(404).is_success is False
    assert mod.Response(500).is_error is True


def test_response_text_property_offline(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    resp = mod.Response(200, text="hello world")
    assert resp.text == "hello world"


# --------------------------------------------------------------------------
# Request contract
# --------------------------------------------------------------------------
def test_request_construction(depcheck):
    """httpx.Request(method, url, headers=, json=) — handlers receive Request
    objects and the OTel hooks read .method/.url/.headers."""
    mod = depcheck.load(IMPORT_NAME)
    req = mod.Request("POST", "https://api.example.test/v1/chat", json={"x": 1})
    assert req.method == "POST"
    assert str(req.url) == "https://api.example.test/v1/chat"
    assert "host" in {k.lower() for k in req.headers.keys()}


def test_request_url_is_httpx_url(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    req = mod.Request("GET", "https://api.example.test/path?q=1")
    assert isinstance(req.url, mod.URL)
    assert req.url.path == "/path"


# --------------------------------------------------------------------------
# exception hierarchy
# --------------------------------------------------------------------------
def test_request_and_status_errors_subclass_httperror(depcheck):
    """Broad `except httpx.HTTPError` handlers must keep catching both the
    transport-side RequestError family and HTTPStatusError. In httpx both
    descend from HTTPError (HTTPStatusError is a *sibling* of RequestError,
    not a subclass of it)."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.RequestError, mod.HTTPError)
    assert issubclass(mod.HTTPStatusError, mod.HTTPError)


def test_httpstatuserror_is_not_requesterror(depcheck):
    """Pin the actual tree shape: HTTPStatusError does NOT subclass
    RequestError. Code catching RequestError will not catch a 4xx/5xx raised
    by raise_for_status — that distinction is a real behavioural contract."""
    mod = depcheck.load(IMPORT_NAME)
    assert not issubclass(mod.HTTPStatusError, mod.RequestError)


def test_transport_errors_subclass_requesterror(depcheck):
    """ConnectError/TimeoutException/TransportError must remain under
    RequestError so transport-failure handlers keep working."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("ConnectError", "TimeoutException", "TransportError", "NetworkError"):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.RequestError), f"{name} no longer subclasses RequestError"


def test_timeout_exception_tree(depcheck):
    """The concrete timeout classes the code may catch all descend from
    TimeoutException (which itself is a RequestError)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.TimeoutException), (
            f"{name} no longer subclasses TimeoutException"
        )
        assert issubclass(exc, mod.RequestError)


def test_connect_error_is_network_error(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ConnectError, mod.NetworkError)


def test_all_used_exceptions_descend_from_httperror_or_are_invalidurl(depcheck):
    """Every exception in USED_EXCEPTIONS is either a descendant of HTTPError
    (the catch-all base) or one of the handful httpx keeps outside that tree.
    Guards against a reorg that orphans a class we catch."""
    mod = depcheck.load(IMPORT_NAME)
    # InvalidURL descends from Exception; StreamError from RuntimeError —
    # neither is under HTTPError, which is itself a real httpx contract.
    outside_httperror = {"InvalidURL", "StreamError"}
    for name in USED_EXCEPTIONS:
        exc = getattr(mod, name)
        if name in outside_httperror:
            assert issubclass(exc, Exception)
        else:
            assert issubclass(exc, mod.HTTPError), (
                f"{name} unexpectedly no longer descends from HTTPError"
            )


def test_streamerror_is_runtimeerror_not_httperror(depcheck):
    """StreamError sits outside the HTTPError tree (it's a RuntimeError) —
    pin that so a handler relying on `except RuntimeError` for stream misuse
    keeps catching it and we don't wrongly assume it's an HTTPError."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.StreamError, RuntimeError)
    assert not issubclass(mod.StreamError, mod.HTTPError)


def test_invalidurl_is_outside_httperror(depcheck):
    """InvalidURL descends from Exception, not HTTPError — pin the shape."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.InvalidURL, Exception)
    assert not issubclass(mod.InvalidURL, mod.HTTPError)


def test_httpstatuserror_carries_request_and_response(depcheck):
    """raise_for_status() raises HTTPStatusError with .request and .response
    populated — error handlers read those to log status/body."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.HTTPStatusError.__init__)
    for p in ("request", "response"):
        assert p in sig.parameters, f"HTTPStatusError.__init__ lost `{p}`"


# --------------------------------------------------------------------------
# behavioural — sync Client over MockTransport (offline)
# --------------------------------------------------------------------------
def test_sync_get_returns_mocked_response(depcheck):
    """Drive Client.get through MockTransport and assert status/json/text —
    no network."""
    mod = depcheck.load(IMPORT_NAME)

    def handler(request):
        assert isinstance(request, mod.Request)
        return mod.Response(200, json={"ok": True, "path": request.url.path}, headers={"X-T": "1"})

    client = mod.Client(transport=mod.MockTransport(handler), base_url="https://api.example.test")
    try:
        resp = client.get("/v1/models", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "path": "/v1/models"}
        assert resp.headers["X-T"] == "1"
        assert isinstance(resp.text, str) and resp.text
    finally:
        client.close()


def test_sync_post_sends_json_body(depcheck):
    """Client.post(json=...) serialises the body and the handler can read it
    back off request.content — pins the request-side json contract offline."""
    mod = depcheck.load(IMPORT_NAME)
    import json as _json

    seen = {}

    def handler(request):
        seen["body"] = _json.loads(request.content.decode())
        seen["ctype"] = request.headers.get("content-type")
        return mod.Response(201, json={"created": True})

    client = mod.Client(transport=mod.MockTransport(handler))
    try:
        resp = client.post("https://api.example.test/v1/chat", json={"model": "gpt"})
        assert resp.status_code == 201
        assert resp.json() == {"created": True}
        assert seen["body"] == {"model": "gpt"}
        assert "application/json" in (seen["ctype"] or "")
    finally:
        client.close()


def test_sync_raise_for_status_raises_httpstatuserror(depcheck):
    """A 4xx response → raise_for_status() raises HTTPStatusError exposing
    .response.status_code; offline."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(transport=mod.MockTransport(_ok_handler(status_code=404)))
    try:
        resp = client.get("https://api.example.test/missing")
        assert resp.status_code == 404
        with pytest.raises(mod.HTTPStatusError) as ei:
            resp.raise_for_status()
        assert ei.value.response.status_code == 404
        assert ei.value.request is not None
    finally:
        client.close()


def test_sync_request_method_dispatch(depcheck):
    """Client.request('GET', url) is the generic entrypoint the SDKs use;
    confirm it round-trips through MockTransport."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(transport=mod.MockTransport(_ok_handler({"via": "request"})))
    try:
        resp = client.request("GET", "https://api.example.test/x")
        assert resp.json() == {"via": "request"}
    finally:
        client.close()


def test_sync_iter_bytes_streaming(depcheck):
    """Response.iter_bytes() yields the body in chunks — used by streaming
    download paths. Exercise it offline."""
    mod = depcheck.load(IMPORT_NAME)
    payload = b"x" * 100

    def handler(request):
        return mod.Response(200, content=payload)

    client = mod.Client(transport=mod.MockTransport(handler))
    try:
        with client.stream("GET", "https://api.example.test/blob") as resp:
            assert resp.status_code == 200
            collected = b"".join(resp.iter_bytes())
        assert collected == payload
    finally:
        client.close()


def test_sync_client_is_context_manager(depcheck):
    """`with httpx.Client(...) as c:` is the idiomatic usage; pin the CM
    protocol and that requests work inside it."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod.Client, "__enter__")
    assert hasattr(mod.Client, "__exit__")
    with mod.Client(transport=mod.MockTransport(_ok_handler())) as client:
        resp = client.get("https://api.example.test/")
        assert resp.status_code == 200


def test_sync_base_url_is_joined(depcheck):
    """base_url + relative path is joined into the absolute request URL —
    the backend builds clients with a base_url and issues relative gets."""
    mod = depcheck.load(IMPORT_NAME)
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return mod.Response(200, json={})

    client = mod.Client(
        transport=mod.MockTransport(handler), base_url="https://api.example.test/v1"
    )
    try:
        client.get("/models")
        assert seen["url"] == "https://api.example.test/v1/models"
    finally:
        client.close()


def test_sync_default_headers_merged(depcheck):
    """Headers passed to the Client constructor are sent on every request and
    merged with per-call headers (auth-token injection pattern)."""
    mod = depcheck.load(IMPORT_NAME)
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["x"] = request.headers.get("x-call")
        return mod.Response(200, json={})

    client = mod.Client(
        transport=mod.MockTransport(handler),
        headers={"Authorization": "Bearer tok"},
    )
    try:
        client.get("https://api.example.test/", headers={"X-Call": "1"})
        assert seen["auth"] == "Bearer tok"
        assert seen["x"] == "1"
    finally:
        client.close()


def test_sync_follow_redirects_kwarg_accepted(depcheck):
    """follow_redirects is passed at construction (mcp/client.py) and per
    call; assert it's accepted and a normal response still flows."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Client(transport=mod.MockTransport(_ok_handler()), follow_redirects=True)
    try:
        resp = client.get("https://api.example.test/", follow_redirects=True)
        assert resp.status_code == 200
    finally:
        client.close()


def test_mocktransport_handler_receives_request(depcheck):
    """Pin the MockTransport contract itself: the handler is called with an
    httpx.Request and may return an httpx.Response. This is the offline
    primitive the whole behavioural suite depends on."""
    mod = depcheck.load(IMPORT_NAME)
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        return mod.Response(200, json={"seen": True})

    client = mod.Client(transport=mod.MockTransport(handler))
    try:
        client.get("https://api.example.test/a")
        client.post("https://api.example.test/b", json={})
    finally:
        client.close()
    assert calls == [("GET", "/a"), ("POST", "/b")]


# --------------------------------------------------------------------------
# behavioural — async AsyncClient over MockTransport (offline)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_get_returns_mocked_response(depcheck):
    """AsyncClient.get through MockTransport — the backend's MCP transport is
    async, so pin the async happy path offline."""
    mod = depcheck.load(IMPORT_NAME)

    def handler(request):
        return mod.Response(200, json={"async": True, "path": request.url.path})

    async with mod.AsyncClient(
        transport=mod.MockTransport(handler),
        base_url="https://api.example.test",
        follow_redirects=True,
    ) as client:
        resp = await client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == {"async": True, "path": "/ping"}


@pytest.mark.asyncio
async def test_async_post_json(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    import json as _json

    seen = {}

    def handler(request):
        seen["body"] = _json.loads(request.content.decode())
        return mod.Response(200, json={"echo": seen["body"]})

    async with mod.AsyncClient(transport=mod.MockTransport(handler)) as client:
        resp = await client.post("https://api.example.test/echo", json={"k": "v"})
        assert resp.json() == {"echo": {"k": "v"}}
        assert seen["body"] == {"k": "v"}


@pytest.mark.asyncio
async def test_async_raise_for_status(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    async with mod.AsyncClient(transport=mod.MockTransport(_ok_handler(status_code=500))) as client:
        resp = await client.get("https://api.example.test/boom")
        assert resp.status_code == 500
        with pytest.raises(mod.HTTPStatusError):
            resp.raise_for_status()


@pytest.mark.asyncio
async def test_async_aiter_bytes_streaming(depcheck):
    """Response.aiter_bytes() yields the body asynchronously in chunks —
    used by async streaming paths. Exercise it offline."""
    mod = depcheck.load(IMPORT_NAME)
    payload = b"y" * 64

    def handler(request):
        return mod.Response(200, content=payload)

    async with mod.AsyncClient(transport=mod.MockTransport(handler)) as client:
        async with client.stream("GET", "https://api.example.test/blob") as resp:
            assert resp.status_code == 200
            collected = b""
            async for chunk in resp.aiter_bytes():
                collected += chunk
        assert collected == payload


@pytest.mark.asyncio
async def test_async_client_is_async_context_manager(depcheck):
    """`async with httpx.AsyncClient(...)` — the backend uses the async CM
    protocol; pin __aenter__/__aexit__."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod.AsyncClient, "__aenter__")
    assert hasattr(mod.AsyncClient, "__aexit__")
    async with mod.AsyncClient(transport=mod.MockTransport(_ok_handler())) as client:
        resp = await client.get("https://api.example.test/")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_async_client_accepts_backend_kwargs_runtime(depcheck):
    """Construct an AsyncClient the way mcp/client.py does (follow_redirects,
    verify, timeout, headers) and confirm it works end-to-end offline."""
    mod = depcheck.load(IMPORT_NAME)
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("x-key")
        return mod.Response(200, json={"ok": True})

    client = mod.AsyncClient(
        transport=mod.MockTransport(handler),
        follow_redirects=True,
        verify=True,
        timeout=mod.Timeout(30.0),
        headers={"X-Key": "secret"},
    )
    try:
        resp = await client.get("https://api.example.test/")
        assert resp.status_code == 200
        assert seen["auth"] == "secret"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_client_aclose_callable(depcheck):
    """AsyncClient.aclose() is awaited on teardown (mcp disconnect path);
    pin it exists and is awaitable."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.AsyncClient(transport=mod.MockTransport(_ok_handler()))
    assert callable(client.aclose)
    await client.aclose()
