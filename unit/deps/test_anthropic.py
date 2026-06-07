"""Dependency contract: anthropic (the official Anthropic Python SDK).

`anthropic==0.86.0` is a pinned runtime dependency of the Open WebUI backend
(declared in `backend/pyproject.toml`). Note the usage shape it guards:
the backend's `open_webui/utils/anthropic.py` currently talks to the
Anthropic API over **raw `aiohttp`** (it sets the `x-api-key` /
`anthropic-version` headers and hits `/v1/models` and the Messages API
itself), and the rest of the codebase only does Anthropic<->OpenAI payload
*format conversion*. So at the time of writing nothing under
`backend/open_webui/` does `import anthropic`.

Because the SDK is nevertheless a declared, version-pinned dependency that a
provider integration is expected to reach for, this module pins the slice of
the `anthropic` public API that any consumer (and a future SDK-based code
path) relies on, so a bump that removed/renamed a client class, reshaped the
`Anthropic(...)` constructor kwargs or the `messages.create(...)` parameter
set, or reorganised the exception tree fails loudly here instead of surfacing
as an AttributeError / TypeError deep in a provider call.

All contracts are offline: clients are *constructed* (which the SDK does
without any network or key validation) but never used to issue a request, and
the exception/type checks are pure introspection. Nothing here touches the
Anthropic API.

Follows the `unit/deps/` exemplar (`test_requests.py` / `test_httpx.py`):
symbol-existence checks for the API surface + offline behavioural contracts,
all via the `depcheck` fixture from `unit/deps/conftest.py`. Skips cleanly if
`anthropic` is not importable.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "anthropic"
DIST_NAME = "anthropic"

# Top-level client + helper symbols the SDK exposes and an integration relies
# on: the sync/async clients, the streaming wrappers, the sentinel types used
# in signatures, and the default-httpx-client escape hatch.
USED_SYMBOLS = [
    "Anthropic",
    "AsyncAnthropic",
    # The SDK also exports these as aliases; pin them so a rename is caught.
    "Client",
    "AsyncClient",
    # Streaming result wrappers returned by create(stream=True) / stream().
    "Stream",
    "AsyncStream",
    "MessageStream",
    "MessageStreamManager",
    # Sentinels used in the public signatures (timeout default, etc.).
    "NOT_GIVEN",
    "NotGiven",
    # httpx-client override hooks the constructor accepts via http_client=.
    "DefaultHttpxClient",
    "DefaultAsyncHttpxClient",
    # Typed namespaces.
    "types",
    "resources",
]

# The exception classes a caller catches around client calls. The whole tree
# must stay intact so a broad `except anthropic.APIError` (or the narrower
# subclasses) keeps catching what it expects after a bump.
USED_EXCEPTIONS = [
    "AnthropicError",
    "APIError",
    "APIStatusError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseValidationError",
    "RateLimitError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "InternalServerError",
]

# Members of `anthropic.types` an integration reads off responses / passes in
# request bodies. Pin existence so a types reshuffle is caught.
USED_TYPES = [
    "Message",
    "MessageParam",
    "TextBlock",
    "ToolUseBlock",
    "ToolParam",
    "Usage",
    "ContentBlock",
    "RawMessageStreamEvent",
    "MessageStreamEvent",
    "Model",
]

# Constructor kwargs the task brief calls out (and any SDK-based provider path
# would pass): the auth key, endpoint override, request timeout, retry budget,
# plus the header/http-client hooks.
_CLIENT_INIT_KWARGS = [
    "api_key",
    "base_url",
    "timeout",
    "max_retries",
    "default_headers",
    "http_client",
]

# messages.create kwargs the Messages API is driven with. `max_tokens`,
# `messages`, `model` are required; the rest map onto Open WebUI's
# Anthropic<->OpenAI conversion surface (system / tools / tool_choice /
# temperature / top_p / top_k / stop_sequences / stream / metadata).
_CREATE_KWARGS = [
    "max_tokens",
    "messages",
    "model",
    "system",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "tools",
    "tool_choice",
    "metadata",
    "extra_headers",
    "timeout",
]


# --------------------------------------------------------------------------
# helpers (local only — no cross-test imports)
# --------------------------------------------------------------------------
def _client(mod):
    """A constructed sync client with a dummy key (offline; no request)."""
    return mod.Anthropic(api_key="test-key-not-used")


# --------------------------------------------------------------------------
# import / version
# --------------------------------------------------------------------------
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "anthropic"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version resolves, so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_attribute_present(depcheck):
    """The SDK exposes anthropic.__version__; keep it a non-empty str."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    assert mod.__version__


# --------------------------------------------------------------------------
# symbol existence
# --------------------------------------------------------------------------
def test_used_symbols_exist(depcheck):
    """Every top-level anthropic symbol a consumer relies on must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_used_exceptions_exist(depcheck):
    """Every anthropic exception a caller catches must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_EXCEPTIONS)


def test_used_types_exist(depcheck):
    """Every anthropic.types member read off responses must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod.types, USED_TYPES)


def test_clients_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Anthropic)
    assert inspect.isclass(mod.AsyncAnthropic)


def test_client_aliases_point_at_clients(depcheck):
    """`anthropic.Client` / `anthropic.AsyncClient` are documented aliases of
    the concrete client classes; pin that identity so code importing either
    name keeps getting the real client."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.Client is mod.Anthropic
    assert mod.AsyncClient is mod.AsyncAnthropic


def test_stream_wrappers_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("Stream", "AsyncStream", "MessageStream", "MessageStreamManager"):
        assert inspect.isclass(getattr(mod, name)), f"anthropic.{name} is not a class"


def test_default_httpx_clients_are_classes(depcheck):
    """DefaultHttpxClient / DefaultAsyncHttpxClient are the http_client=
    override hooks; keep them constructible classes."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.DefaultHttpxClient)
    assert inspect.isclass(mod.DefaultAsyncHttpxClient)


# --------------------------------------------------------------------------
# constructor signatures
# --------------------------------------------------------------------------
def test_anthropic_init_accepts_used_kwargs(depcheck):
    """Anthropic(api_key=, base_url=, timeout=, max_retries=, default_headers=,
    http_client=) — the kwargs an integration passes must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Anthropic.__init__, _CLIENT_INIT_KWARGS)


def test_async_anthropic_init_accepts_used_kwargs(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncAnthropic.__init__, _CLIENT_INIT_KWARGS)


def test_anthropic_init_accepts_core_kwargs_exactly(depcheck):
    """Narrow guard on the four kwargs the brief pins (api_key, base_url,
    timeout, max_retries) so a rename of any one fails by name."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.Anthropic.__init__,
        ["api_key", "base_url", "timeout", "max_retries"],
    )


def test_constructor_kwargs_are_keyword_only(depcheck):
    """The client constructor is keyword-only (`def __init__(self, *, ...)`);
    a positional api_key must be rejected. Pin that so call sites keep using
    explicit kwargs and a positional regression is caught."""
    mod = depcheck.load(IMPORT_NAME)
    params = inspect.signature(mod.Anthropic.__init__).parameters
    # `self` is the only non-keyword-only parameter.
    positional = [
        p.name
        for p in params.values()
        if p.name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == [], f"client ctor unexpectedly accepts positional args: {positional}"
    with pytest.raises(TypeError):
        mod.Anthropic("positional-api-key")  # type: ignore[misc]


def test_max_retries_default_is_int(depcheck):
    """max_retries defaults to a concrete int (the SDK's built-in retry
    budget), not a sentinel — code reads/relies on the numeric default."""
    mod = depcheck.load(IMPORT_NAME)
    default = inspect.signature(mod.Anthropic.__init__).parameters["max_retries"].default
    assert isinstance(default, int)


# --------------------------------------------------------------------------
# messages resource + create() signature
# --------------------------------------------------------------------------
def test_messages_resource_present(depcheck):
    """A constructed client exposes `.messages` with a callable `.create`
    (and `.stream` / `.count_tokens`). Constructing the client does NOT hit
    the network, so this is safe offline."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    messages = client.messages
    assert messages is not None
    assert callable(messages.create)
    assert callable(messages.stream)
    assert callable(messages.count_tokens)


def test_async_messages_resource_present(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    client = mod.AsyncAnthropic(api_key="test-key-not-used")
    messages = client.messages
    assert messages is not None
    assert callable(messages.create)
    assert callable(messages.stream)


def test_messages_create_signature(depcheck):
    """messages.create(max_tokens=, messages=, model=, system=, temperature=,
    top_p=, top_k=, stop_sequences=, stream=, tools=, tool_choice=,
    metadata=, extra_headers=, timeout=) — the parameters an integration
    passes (mirroring Open WebUI's Anthropic<->OpenAI conversion fields)
    must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    depcheck.assert_params(client.messages.create, _CREATE_KWARGS)


def test_async_messages_create_signature(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    client = mod.AsyncAnthropic(api_key="test-key-not-used")
    depcheck.assert_params(client.messages.create, _CREATE_KWARGS)


def test_messages_create_required_params_are_keyword_only(depcheck):
    """max_tokens / messages / model are required and keyword-only on
    create(); pin that so the required-by-name contract holds (a positional
    call would be a regression)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    params = inspect.signature(client.messages.create).parameters
    for required in ("max_tokens", "messages", "model"):
        assert required in params, f"messages.create lost required param `{required}`"
        assert params[required].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"messages.create `{required}` is no longer keyword-only"
        )
        # Required → no default value.
        assert params[required].default is inspect.Parameter.empty, (
            f"messages.create `{required}` unexpectedly became optional"
        )


# --------------------------------------------------------------------------
# NOT_GIVEN sentinel
# --------------------------------------------------------------------------
def test_not_given_is_instance_of_notgiven(depcheck):
    """NOT_GIVEN is the sentinel for 'omit this param'; it must be an instance
    of NotGiven so `isinstance(x, NotGiven)` guards keep working."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.NotGiven)
    assert isinstance(mod.NOT_GIVEN, mod.NotGiven)


# --------------------------------------------------------------------------
# exception hierarchy
# --------------------------------------------------------------------------
def test_anthropic_error_is_root(depcheck):
    """AnthropicError is the package's root error and a plain Exception
    subclass; everything else descends from it so a broad
    `except anthropic.AnthropicError` is the catch-all."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.AnthropicError, Exception)


def test_api_error_subclasses_anthropic_error(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.APIError, mod.AnthropicError)


def test_status_and_connection_errors_subclass_api_error(depcheck):
    """APIStatusError / APIConnectionError / APIResponseValidationError all
    descend from APIError so `except anthropic.APIError` catches both the
    HTTP-status family and transport failures."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("APIStatusError", "APIConnectionError", "APIResponseValidationError"):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.APIError), f"{name} no longer subclasses APIError"


def test_status_code_errors_subclass_api_status_error(depcheck):
    """The concrete HTTP-status errors (4xx/5xx) all descend from
    APIStatusError, so a handler catching APIStatusError keeps catching e.g.
    RateLimitError and AuthenticationError."""
    mod = depcheck.load(IMPORT_NAME)
    for name in (
        "RateLimitError",
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
        "InternalServerError",
    ):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.APIStatusError), f"{name} no longer subclasses APIStatusError"


def test_rate_limit_error_is_api_error(depcheck):
    """RateLimitError must remain catchable via the broad APIError base too
    (it descends APIStatusError -> APIError)."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.RateLimitError, mod.APIError)


def test_api_timeout_error_subclasses_connection_error(depcheck):
    """APITimeoutError is a *connection* error (descends APIConnectionError),
    not a status error — pin that so timeout handlers sit on the right branch
    and a `except APIConnectionError` catches timeouts."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.APITimeoutError, mod.APIConnectionError)
    assert issubclass(mod.APITimeoutError, mod.APIError)


def test_api_status_error_not_a_connection_error(depcheck):
    """APIStatusError and APIConnectionError are sibling branches: a 4xx/5xx
    is not a connection error. Pin the distinction so handlers that catch one
    don't silently catch the other."""
    mod = depcheck.load(IMPORT_NAME)
    assert not issubclass(mod.APIStatusError, mod.APIConnectionError)
    assert not issubclass(mod.APIConnectionError, mod.APIStatusError)


def test_all_used_exceptions_descend_from_anthropic_error(depcheck):
    """Every exception in USED_EXCEPTIONS descends from the AnthropicError
    root. Guards against a reorg that orphans a class we catch."""
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_EXCEPTIONS:
        exc = getattr(mod, name)
        assert issubclass(exc, mod.AnthropicError), (
            f"{name} unexpectedly no longer descends from AnthropicError"
        )


def test_api_status_error_carries_response_and_body(depcheck):
    """APIStatusError exposes `.response` and `.body` (set from the failing
    HTTP response) — error handlers read those for the status/body. Pin the
    constructor params that back them."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.APIStatusError.__init__)
    for p in ("message", "response", "body"):
        assert p in sig.parameters, f"APIStatusError.__init__ lost `{p}`"


def test_api_status_error_declares_status_code(depcheck):
    """Code reads `err.status_code` to branch on the HTTP status. The base
    APIStatusError declares it as a class annotation, and each concrete
    subclass pins its own HTTP status as a class attribute (404, 429, ...).
    Pin both so the attribute a handler reads stays present."""
    mod = depcheck.load(IMPORT_NAME)
    assert "status_code" in getattr(mod.APIStatusError, "__annotations__", {})
    expected = {
        "BadRequestError": 400,
        "AuthenticationError": 401,
        "PermissionDeniedError": 403,
        "NotFoundError": 404,
        "ConflictError": 409,
        "UnprocessableEntityError": 422,
        "RateLimitError": 429,
    }
    for name, code in expected.items():
        exc = getattr(mod, name)
        assert getattr(exc, "status_code", None) == code, (
            f"{name}.status_code is no longer the class attribute {code}"
        )


# --------------------------------------------------------------------------
# offline behavioural contracts — construction only, no network / no request
# --------------------------------------------------------------------------
def test_construct_sync_client_offline(depcheck):
    """Anthropic(api_key=, base_url=, timeout=, max_retries=) constructs fully
    offline (the SDK does not validate the key or open a connection at
    construction time) and stores the config on the client. We never call a
    method that would issue a request."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Anthropic(
        api_key="test-key-not-used",
        base_url="https://api.example.test",
        timeout=30.0,
        max_retries=0,
    )
    assert client.api_key == "test-key-not-used"
    # base_url is normalised to an httpx.URL; compare via str (trailing slash
    # is appended by httpx, so assert the prefix rather than exact equality).
    assert str(client.base_url).startswith("https://api.example.test")
    assert client.max_retries == 0
    # The messages resource is wired up and ready (still no request issued).
    assert client.messages is not None


def test_construct_async_client_offline(depcheck):
    """AsyncAnthropic constructs offline the same way and exposes .messages."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.AsyncAnthropic(api_key="test-key-not-used", max_retries=1)
    assert client.api_key == "test-key-not-used"
    assert client.max_retries == 1
    assert client.messages is not None


def test_construct_with_timeout_not_given_default(depcheck):
    """Omitting timeout leaves the SDK's NOT_GIVEN default in the signature;
    construction without a timeout still succeeds offline."""
    mod = depcheck.load(IMPORT_NAME)
    default = inspect.signature(mod.Anthropic.__init__).parameters["timeout"].default
    assert isinstance(default, mod.NotGiven)
    client = mod.Anthropic(api_key="test-key-not-used")
    assert client.messages is not None


def test_client_with_options_returns_client(depcheck):
    """`client.with_options(...)` is the documented copy-with-overrides hook;
    it returns a client of the same type (offline, no request)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    assert callable(client.with_options)
    copy = client.with_options(max_retries=5)
    assert isinstance(copy, mod.Anthropic)
    assert copy.max_retries == 5


def test_constructing_client_issues_no_request(depcheck, monkeypatch):
    """Hard guard that construction is offline: monkeypatch the underlying
    httpx clients' `send` to blow up, then construct both clients. If
    construction tried to issue any HTTP request this test would fail."""
    mod = depcheck.load(IMPORT_NAME)
    httpx = depcheck.try_load("httpx")
    if httpx is None:
        pytest.skip("httpx not importable; cannot install the no-request guard")

    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a real send
        raise AssertionError("anthropic client issued an HTTP request during construction")

    monkeypatch.setattr(httpx.Client, "send", _boom, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "send", _boom, raising=True)

    sync_client = mod.Anthropic(api_key="test-key-not-used")
    async_client = mod.AsyncAnthropic(api_key="test-key-not-used")
    assert sync_client.messages is not None
    assert async_client.messages is not None
