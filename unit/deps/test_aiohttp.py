"""Dependency contract: aiohttp.

Open WebUI does all of its *asynchronous* outbound HTTP through `aiohttp`:
the retrieval web loaders (`SafeWebBaseLoader._fetch` builds a
`TCPConnector(resolver=...)` + `ClientSession(trust_env=, connector=)` and
calls `session.get(url, ssl=, cookies=, headers=, allow_redirects=)`),
provider streaming (`routers/openai.py`, `routers/ollama.py`,
`utils/anthropic.py` await `session.request(method=, url=, data=, headers=,
cookies=, ssl=, timeout=)` and consume `r.content.iter_chunked(...)`), image
generation/edit and audio STT/TTS (`routers/images.py`, `routers/audio.py`
build `aiohttp.FormData()` and post `data=form`), the OCR loader
(`retrieval/loaders/mistral.py` streams via `MultipartWriter` /
`streams.FilePayload`), and the shared connection pool
(`utils/session_pool.py`).

This module pins the slice of the aiohttp API that backend code relies on,
so an aiohttp bump that removed/renamed/re-typed any of it fails loudly here
instead of as a runtime AttributeError / TypeError deep in a request path.
aiohttp is a known-fragile pin for this project: requirements.txt carries
`aiohttp==3.13.5 # do not update to 3.13.3 - broken`, and issue #24560 was an
aiohttp keyword-argument regression. The signature/param tests below
(ClientSession.get / .request / ._request accepting ssl=, cookies=,
allow_redirects=, timeout=, headers=) are the highest-value guard against a
repeat of that class of breakage.

Pattern (see test_requests.py): symbol-existence checks (API surface) +
offline behavioural contracts. Nothing here touches the network — sessions
and timeouts are constructed but never used to connect, exception classes are
only introspected, and the one event-loop test drives an in-memory
StreamReader. Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "aiohttp"
DIST_NAME = "aiohttp"

# Module-level symbols the Open WebUI backend references on `aiohttp`.
USED_SYMBOLS = [
    # Core client objects
    "ClientSession",
    "ClientTimeout",
    "TCPConnector",
    "ClientResponse",
    "StreamReader",
    # Form / multipart bodies
    "FormData",
    "MultipartWriter",
    # Exceptions caught in the codebase
    "ClientError",
    "ClientConnectionError",
    "ClientResponseError",
    "ServerTimeoutError",
    # Submodule symbols used by retrieval (SSRF resolver) and the OCR loader.
    # NOTE: `streams.FilePayload` (referenced in retrieval/loaders/mistral.py)
    # is intentionally NOT here — it is absent in the pinned aiohttp; see
    # test_streams_file_payload_divergence below for the documented contract.
    "resolver.DefaultResolver",
    "streams.StreamReader",
    "payload.Payload",
    "payload.BytesPayload",
]

# Exceptions the backend catches via `except aiohttp.X` / isinstance(e, X).
# Every one must remain a subclass of ClientError so the broad
# `except aiohttp.ClientError` handlers in openai.py / ollama.py keep working.
CLIENT_ERROR_SUBCLASSES = [
    "ClientConnectionError",
    "ClientResponseError",
    "ServerTimeoutError",
]


# --------------------------------------------------------------------------
# Import + version
# --------------------------------------------------------------------------
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "aiohttp"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test). aiohttp is a
    deliberately-pinned dependency for this project."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_attribute_present(depcheck):
    """aiohttp exposes __version__; some telemetry/UA strings read it."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    assert mod.__version__  # non-empty


# --------------------------------------------------------------------------
# Symbol existence (API surface)
# --------------------------------------------------------------------------
def test_used_symbols_exist(depcheck):
    """Every aiohttp symbol the codebase references must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_client_session_exists_and_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientSession")
    assert inspect.isclass(mod.ClientSession)


def test_client_timeout_exists_and_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientTimeout")
    assert inspect.isclass(mod.ClientTimeout)


def test_tcp_connector_exists_and_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "TCPConnector")
    assert inspect.isclass(mod.TCPConnector)


def test_formdata_exists_and_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "FormData")
    assert inspect.isclass(mod.FormData)


def test_multipart_writer_exists_and_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "MultipartWriter")
    assert inspect.isclass(mod.MultipartWriter)


def test_client_response_is_class(depcheck):
    """utils/misc.py and utils/session_pool.py annotate aiohttp.ClientResponse."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.ClientResponse)


def test_stream_reader_is_class(depcheck):
    """utils/misc.py annotates aiohttp.StreamReader (stream_chunks_handler)."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.StreamReader)


def test_resolver_submodule_and_default_resolver(depcheck):
    """retrieval/web/utils.py subclasses aiohttp.resolver.DefaultResolver
    to build the SSRF-safe resolver passed to TCPConnector(resolver=...)."""
    mod = depcheck.load(IMPORT_NAME)
    resolver = depcheck.resolve(mod, "resolver")
    assert inspect.isclass(resolver.DefaultResolver)


def test_streams_submodule_and_stream_reader(depcheck):
    """utils/misc.py annotates aiohttp.streams.StreamReader and the streaming
    paths iterate it; the submodule + class must exist."""
    mod = depcheck.load(IMPORT_NAME)
    streams = depcheck.resolve(mod, "streams")
    assert inspect.isclass(streams.StreamReader)


def test_payload_submodule_and_payload_classes(depcheck):
    """The OCR multipart path builds payload objects; aiohttp.payload exposes
    the Payload base plus the concrete bytes/stream payload classes the
    MultipartWriter machinery relies on."""
    mod = depcheck.load(IMPORT_NAME)
    payload = depcheck.resolve(mod, "payload")
    assert inspect.isclass(payload.Payload)
    assert inspect.isclass(payload.BytesPayload)
    assert inspect.isclass(payload.StreamReaderPayload)


def test_streams_file_payload_divergence(depcheck):
    """retrieval/loaders/mistral.py references `aiohttp.streams.FilePayload`
    inside its lazy upload closure, but that symbol does NOT exist in the
    pinned aiohttp (the payload classes live in `aiohttp.payload`, and there
    is no FilePayload there either). The reference is unreached on import, so
    it does not break startup — but the OCR-upload code path is latently
    broken against this pin.

    This test documents the divergence so it stays visible: it asserts the
    symbol is *absent* today. If a future aiohttp bump (re)introduces
    `streams.FilePayload`, this test flips to failing — a signal to revisit
    whether mistral.py's reference now resolves (and to update this contract)."""
    mod = depcheck.load(IMPORT_NAME)
    streams = depcheck.resolve(mod, "streams")
    assert not hasattr(streams, "FilePayload"), (
        "aiohttp.streams.FilePayload now EXISTS — mistral.py's reference may "
        "resolve again; update this contract and re-check the OCR upload path."
    )


# --------------------------------------------------------------------------
# Exception hierarchy
# --------------------------------------------------------------------------
def test_client_error_is_exception(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ClientError, Exception)


@pytest.mark.parametrize("name", CLIENT_ERROR_SUBCLASSES)
def test_exceptions_subclass_client_error(depcheck, name):
    """Code uses `except aiohttp.ClientError` as the catch-all; the specific
    exceptions it also catches must remain ClientError subclasses."""
    mod = depcheck.load(IMPORT_NAME)
    exc = getattr(mod, name)
    assert issubclass(exc, mod.ClientError), (
        f"aiohttp.{name} no longer subclasses ClientError; broad "
        f"`except aiohttp.ClientError` handlers would stop catching it."
    )


def test_client_connection_error_hierarchy(depcheck):
    """SafeWebBaseLoader._fetch retries on `except aiohttp.ClientConnectionError`."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ClientConnectionError, mod.ClientError)


def test_client_response_error_hierarchy(depcheck):
    """images.py/audio.py/oauth.py/mistral.py do isinstance(e,
    aiohttp.ClientResponseError) then read e.status / e.message."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ClientResponseError, mod.ClientError)


def test_server_timeout_error_hierarchy(depcheck):
    """mistral.py treats aiohttp.ServerTimeoutError as retryable alongside
    ClientConnectionError."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.ServerTimeoutError, mod.ClientError)


def test_client_response_error_has_status_and_message(depcheck):
    """mistral.py/oauth.py read e.status and e.message off the raised
    ClientResponseError. Construct one offline and assert the attributes
    are present and carry the values we passed."""
    mod = depcheck.load(IMPORT_NAME)
    # ClientResponseError(request_info, history, *, status=, message=, headers=)
    exc = mod.ClientResponseError(
        request_info=None,
        history=(),
        status=503,
        message="Service Unavailable",
    )
    assert exc.status == 503
    assert exc.message == "Service Unavailable"


# --------------------------------------------------------------------------
# Signature / keyword-argument contracts (HIGH VALUE — issue #24560)
# --------------------------------------------------------------------------
def _session_request_callable(mod):
    """Return the bound-ish callable that actually defines the request kwargs.

    aiohttp routes ClientSession.get/post/etc. through ClientSession._request,
    while the public verb methods take (url, **kwargs) and a _RequestContextManager
    wrapper. The real keyword surface (ssl/allow_redirects/cookies/timeout/...)
    lives on ClientSession._request. Prefer that; fall back to .request."""
    if hasattr(mod.ClientSession, "_request"):
        return mod.ClientSession._request
    return mod.ClientSession.request


def test_client_session_request_accepts_our_kwargs(depcheck):
    """The single most important contract: the kwargs the backend passes to
    session.get / session.post / session.request must remain accepted.
    Covers retrieval/web/utils.py (ssl, cookies, headers, allow_redirects),
    openai.py/ollama.py (method, url, data, headers, cookies, ssl, timeout)."""
    mod = depcheck.load(IMPORT_NAME)
    func = _session_request_callable(mod)
    # Note: _request's URL parameter is positional `str_or_url`, not `url`
    # (the public verb methods expose `url`); URL acceptance is checked via
    # the public ClientSession.request/.get in the dedicated tests below.
    depcheck.assert_params(
        func,
        [
            "method",
            "headers",
            "cookies",
            "timeout",
            "ssl",
            "allow_redirects",
            "data",
            "json",
        ],
    )


def test_client_session_get_signature(depcheck):
    """`session.get(url, **kwargs)` — get must accept url and forward **kwargs
    (ssl/cookies/headers/allow_redirects are routed through _request)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientSession.get")
    depcheck.assert_params(mod.ClientSession.get, ["url"])


def test_client_session_post_signature(depcheck):
    """`session.post(url, data=/json=, ...)` — images.py/audio.py post forms,
    openai.py posts json."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientSession.post")
    depcheck.assert_params(mod.ClientSession.post, ["url"])


def test_client_session_request_method_signature(depcheck):
    """`session.request(method, url, ...)` is used directly in openai.py,
    ollama.py and terminals.py."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientSession.request")
    depcheck.assert_params(mod.ClientSession.request, ["method", "url"])


def test_client_session_init_accepts_our_kwargs(depcheck):
    """ClientSession is constructed with trust_env=, connector=, timeout=
    (session_pool.py, retrieval/web/utils.py, openai.py, ollama.py)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.ClientSession.__init__,
        ["connector", "timeout", "trust_env"],
    )


def test_client_timeout_accepts_total(depcheck):
    """Every ClientTimeout in the codebase is `ClientTimeout(total=...)`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ClientTimeout.__init__, ["total"])


def test_tcp_connector_accepts_our_kwargs(depcheck):
    """TCPConnector is built with resolver= (web/utils.py) and ttl_dns_cache=,
    enable_cleanup_closed=, limit=, limit_per_host= (session_pool.py)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.TCPConnector.__init__,
        [
            "resolver",
            "ttl_dns_cache",
            "enable_cleanup_closed",
            "limit",
            "limit_per_host",
        ],
    )


def test_formdata_add_field_signature(depcheck):
    """images.py/audio.py call form.add_field(name, value, filename=,
    content_type=)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "FormData.add_field")
    depcheck.assert_params(
        mod.FormData.add_field,
        ["name", "value", "content_type", "filename"],
    )


def test_client_response_json_accepts_content_type(depcheck):
    """images.py calls `await r.json(content_type=None)` to parse responses
    served with a non-JSON content-type."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ClientResponse.json")
    depcheck.assert_params(mod.ClientResponse.json, ["content_type"])


def test_stream_reader_iter_chunked_signature(depcheck):
    """openai.py/ollama.py: `async for chunk in r.content.iter_chunked(8192)`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "StreamReader.iter_chunked")
    depcheck.assert_params(mod.StreamReader.iter_chunked, ["n"])


def test_stream_reader_iter_any_exists(depcheck):
    """terminals.py streams an upstream response via `r.content.iter_any()`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "StreamReader.iter_any")


# --------------------------------------------------------------------------
# ClientResponse / ClientSession method surface (consumed shapes)
# --------------------------------------------------------------------------
def test_client_response_method_surface(depcheck):
    """Responses are consumed via .status/.headers/.text()/.json()/.read()/
    .raise_for_status()/.content/.close()/.closed across the codebase."""
    mod = depcheck.load(IMPORT_NAME)
    cls = mod.ClientResponse
    for attr in (
        "status",
        "headers",
        "text",
        "json",
        "read",
        "raise_for_status",
        "content",
        "close",
        "closed",
    ):
        assert hasattr(cls, attr), f"ClientResponse.{attr} missing"
    for m in ("text", "json", "read", "raise_for_status", "close"):
        assert callable(getattr(cls, m)), f"ClientResponse.{m} not callable"


def test_client_session_method_surface(depcheck):
    """ClientSession is used as an async context manager exposing
    get/post/request/close plus the .closed flag (session_pool.py checks it)."""
    mod = depcheck.load(IMPORT_NAME)
    cls = mod.ClientSession
    for attr in (
        "get",
        "post",
        "request",
        "close",
        "closed",
        "__aenter__",
        "__aexit__",
    ):
        assert hasattr(cls, attr), f"ClientSession.{attr} missing"
    for m in ("get", "post", "request", "close"):
        assert callable(getattr(cls, m)), f"ClientSession.{m} not callable"


def test_stream_reader_method_surface(depcheck):
    """StreamReader is iterated and chunk-read by the streaming paths."""
    mod = depcheck.load(IMPORT_NAME)
    cls = mod.StreamReader
    for m in ("iter_chunked", "iter_any", "read", "__aiter__"):
        assert hasattr(cls, m), f"StreamReader.{m} missing"


def test_multipart_writer_method_surface(depcheck):
    """mistral.py uses writer.append(...) and writer.append_payload(...)."""
    mod = depcheck.load(IMPORT_NAME)
    cls = mod.MultipartWriter
    for m in ("append", "append_payload"):
        assert hasattr(cls, m) and callable(getattr(cls, m)), (
            f"MultipartWriter.{m} missing/not callable"
        )


# --------------------------------------------------------------------------
# Behavioural contracts — offline, no network
# --------------------------------------------------------------------------
def test_client_timeout_construct_offline(depcheck):
    """ClientTimeout(total=...) is a plain dataclass-like config object; build
    one and confirm the value round-trips. No connection involved."""
    mod = depcheck.load(IMPORT_NAME)
    t = mod.ClientTimeout(total=12.5)
    assert t.total == 12.5


def test_formdata_construct_and_add_field_offline(depcheck):
    """Build a FormData exactly like images.py does (string + bytes fields)
    without ever posting it. Must not raise."""
    mod = depcheck.load(IMPORT_NAME)
    form = mod.FormData()
    form.add_field("model", "dall-e-3")
    form.add_field(
        "image",
        b"\x89PNG\r\n",
        filename="image.png",
        content_type="image/png",
    )
    # No public, stable read-back API for fields; the contract is that the
    # calls above are accepted without raising.


def test_tcp_connector_resolver_kwarg_offline(depcheck):
    """retrieval/web/utils.py passes a custom resolver:
    `TCPConnector(resolver=_SSRFSafeResolver())`. Build a connector with the
    stock DefaultResolver to prove the kwarg path holds, then close it so no
    file descriptors leak. No DNS/connection is performed by construction."""
    mod = depcheck.load(IMPORT_NAME)

    async def _build_and_close():
        resolver = mod.resolver.DefaultResolver()
        connector = mod.TCPConnector(resolver=resolver)
        try:
            assert connector is not None
        finally:
            await connector.close()

    asyncio.run(_build_and_close())


def test_client_session_is_async_context_manager_offline(depcheck):
    """`async with aiohttp.ClientSession(...) as session:` is the dominant
    pattern. Construct a session inside a loop, assert it exposes the verb
    methods + async-CM protocol, and close it — never connecting anywhere."""
    mod = depcheck.load(IMPORT_NAME)

    async def _check():
        session = mod.ClientSession(timeout=mod.ClientTimeout(total=1))
        try:
            assert hasattr(session, "__aenter__")
            assert hasattr(session, "__aexit__")
            for m in ("get", "post", "request", "close"):
                assert callable(getattr(session, m))
            assert session.closed is False
        finally:
            await session.close()
        assert session.closed is True

    asyncio.run(_check())


def test_client_session_trust_env_and_connector_offline(depcheck):
    """Mirror session_pool.get_session(): ClientSession(connector=, timeout=,
    trust_env=True) must construct cleanly offline and report not-closed."""
    mod = depcheck.load(IMPORT_NAME)

    async def _check():
        connector = mod.TCPConnector(ttl_dns_cache=300, enable_cleanup_closed=True)
        session = mod.ClientSession(
            connector=connector,
            timeout=mod.ClientTimeout(total=5),
            trust_env=True,
        )
        try:
            assert session.closed is False
        finally:
            await session.close()  # closes the connector it owns too

    asyncio.run(_check())


@pytest.mark.asyncio
async def test_stream_reader_iter_chunked_behaviour(depcheck):
    """Drive a real StreamReader entirely in memory (no socket): feed bytes,
    EOF, then consume via iter_chunked exactly like the provider-streaming
    code does. Proves the async-iteration + chunking contract holds."""
    mod = depcheck.load(IMPORT_NAME)
    reader = make_stream_reader(mod, asyncio.get_event_loop())
    reader.feed_data(b"hello world, streaming chunk")
    reader.feed_eof()

    chunks = []
    async for chunk in reader.iter_chunked(8):
        chunks.append(chunk)

    assert b"".join(chunks) == b"hello world, streaming chunk"
    assert all(len(c) <= 8 for c in chunks)


@pytest.mark.asyncio
async def test_stream_reader_iter_any_behaviour(depcheck):
    """terminals.py forwards an upstream body with `r.content.iter_any()`.
    Confirm iter_any yields the fed data in order, in memory."""
    mod = depcheck.load(IMPORT_NAME)
    reader = make_stream_reader(mod, asyncio.get_event_loop())
    reader.feed_data(b"part-one")
    reader.feed_data(b"part-two")
    reader.feed_eof()

    out = []
    async for chunk in reader.iter_any():
        out.append(chunk)

    assert b"".join(out) == b"part-onepart-two"


def test_get_returns_request_context_manager_type(depcheck):
    """`async with session.get(url) as resp:` requires session.get(...) to
    return an async context manager whose __aenter__ fires the request. Pin
    that shape via the wrapper class itself (aiohttp.client._RequestContextManager)
    so we assert the contract without invoking get() — which would create an
    un-awaited coroutine and never touch the network either way."""
    mod = depcheck.load(IMPORT_NAME)
    client = depcheck.resolve(mod, "client")
    ctx_cls = client._RequestContextManager
    assert hasattr(ctx_cls, "__aenter__")
    assert hasattr(ctx_cls, "__aexit__")


# --------------------------------------------------------------------------
# Local helpers
# --------------------------------------------------------------------------
def make_stream_reader(mod, loop):
    """Build an in-memory aiohttp.StreamReader for offline iteration tests.

    StreamReader requires a real flow-control protocol (it reads
    `_reading_paused` on feed_eof), so we hand it aiohttp's own
    BaseProtocol rather than a hand-rolled stub — robust across versions and
    no socket is ever involved (we only feed_data + feed_eof, then read)."""
    from aiohttp.base_protocol import BaseProtocol

    protocol = BaseProtocol(loop=loop)
    return mod.StreamReader(protocol=protocol, limit=2**16, loop=loop)
