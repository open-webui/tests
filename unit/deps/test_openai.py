"""Dependency contract: openai (the openai-python SDK).

Role in Open WebUI: `openai==2.29.0` is a *declared, pinned* backend
requirement, but — importantly — the backend does NOT import or call the
SDK client at all. Every OpenAI / OpenAI-compatible interaction (chat
completions, the Responses API, embeddings, model listing, image
generation/edit, and audio STT/TTS, including Azure OpenAI) is performed
by hand-rolled `aiohttp` requests in `routers/openai.py`,
`routers/images.py`, `routers/audio.py`, `retrieval/utils.py`,
`utils/embeddings.py` and friends. The module named `open_webui.routers.openai`
is the application's *own* router, not this SDK. The only place the SDK is
even referenced is a comment in `utils/middleware.py`
(`_render_openai_tool_call_handler`) noting that the Responses API
server-side tool item schemas (`web_search_call`, `file_search_call`,
`computer_call`) it renders are "defined in the openai-python SDK
(generated from OpenAPI spec)".

Because the dep is pinned and shipped, two things must hold and are pinned
here so a bump fails loudly instead of surfacing later:

  1. Install/import contract — the pinned package must remain importable
     (a broken/yanked release in the resolved set is caught at collection,
     not at runtime).
  2. Public-surface contract — the clients (`OpenAI` / `AsyncOpenAI` /
     `AzureOpenAI`), their constructor kwargs (`api_key`, `base_url`,
     `timeout`, `max_retries`, `http_client`, ...), the request resources
     (`chat.completions.create`, `embeddings.create`, `models`,
     `responses`, `images`, `audio`) and the exception hierarchy
     (`OpenAIError` → `APIError` → `APIConnectionError` /
     `APIStatusError` → `RateLimitError`, ...) are pinned. The backend
     does not consume these *today*, but the SDK is fast-moving and a
     v2→v3-style surface change is exactly what this suite exists to catch;
     pinning the surface now means the contract is already in place the day
     any code path adopts the client, and the documented Responses tool-item
     type names stay anchored to a known SDK shape.

Pattern (see test_requests.py): symbol-existence checks (API surface) +
offline behavioural contracts. Nothing here touches the network — clients
are constructed against a throwaway localhost base_url but never used to
issue a request, exception classes are only introspected, and create()
methods are checked by signature only. Uses the `depcheck` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "openai"
DIST_NAME = "openai"

# Top-level symbols the SDK exposes that this contract pins: the sync/async
# (and Azure) clients, plus the exception classes a consumer would catch.
USED_SYMBOLS = [
    # clients
    "OpenAI",
    "AsyncOpenAI",
    "AzureOpenAI",
    "AsyncAzureOpenAI",
    # exception hierarchy (the ones the prompt + real consumers care about)
    "OpenAIError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "APIResponseValidationError",
    "RateLimitError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
    # version marker
    "__version__",
]

# Submodule paths the SDK ships, relative to the `openai` package (the
# depcheck helper resolves dotted names against the module it's given, and
# imports submodules as needed). The Responses tool-item schemas referenced
# by the middleware comment live under openai.types.responses.
USED_SUBMODULES = [
    "types",
    "types.chat",
    "types.responses",
    "resources",
    "_client",
    "_exceptions",
]

# Constructor kwargs the prompt enumerates (and any future client adoption
# would pass): these must remain accepted by OpenAI()/AsyncOpenAI().
CLIENT_INIT_KWARGS = ["api_key", "base_url", "timeout", "max_retries", "http_client"]

# Request-method kwargs a consumer would pass to the create() calls.
CHAT_CREATE_KWARGS = ["messages", "model", "temperature", "stream", "tools", "timeout"]
EMBEDDINGS_CREATE_KWARGS = ["input", "model", "timeout"]


def test_import(depcheck):
    """The pinned, shipped dependency must remain importable."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "openai"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_module_version_attr(depcheck):
    """`openai.__version__` is a public marker some integrations read."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str) and mod.__version__


def test_top_level_symbols_exist(depcheck):
    """Every top-level client/exception symbol the contract pins must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_submodules_importable(depcheck):
    """The SDK's documented submodule layout (types/resources/exceptions) is
    stable; the Responses tool-item schemas live under types.responses."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SUBMODULES)


def test_clients_are_classes(depcheck):
    """OpenAI / AsyncOpenAI / AzureOpenAI are constructible classes."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"):
        obj = getattr(mod, name)
        assert inspect.isclass(obj), f"openai.{name} is not a class (got {type(obj)!r})"


def test_openai_constructor_accepts_our_kwargs(depcheck):
    """OpenAI(api_key=, base_url=, timeout=, max_retries=, http_client=) — the
    kwargs the prompt enumerates and any client adoption would pass."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.OpenAI.__init__, CLIENT_INIT_KWARGS)


def test_async_openai_constructor_accepts_our_kwargs(depcheck):
    """AsyncOpenAI must accept the same construction kwargs as OpenAI."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncOpenAI.__init__, CLIENT_INIT_KWARGS)


def test_azure_openai_constructor_accepts_azure_kwargs(depcheck):
    """Azure OpenAI is reached today via raw aiohttp (api-version in the URL);
    if the SDK client is ever adopted for it, the Azure-specific kwargs must
    still exist. Pin api_version / azure_endpoint / azure_deployment plus the
    shared base kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.AzureOpenAI.__init__,
        ["api_key", "api_version", "azure_endpoint", "azure_deployment", "timeout"],
    )


def test_client_constructs_offline(depcheck):
    """Constructing a client must not perform any network I/O. Build one
    against a throwaway localhost base_url and never issue a request."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.OpenAI(api_key="x", base_url="http://localhost:9/v1")
    assert client is not None
    # base_url is normalised to a URL object; just assert it's exposed.
    assert client.base_url is not None
    assert client.max_retries is not None


def test_async_client_constructs_offline(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    client = mod.AsyncOpenAI(api_key="x", base_url="http://localhost:9/v1")
    assert client is not None
    assert client.base_url is not None


def test_client_exposes_request_resources(depcheck):
    """A constructed client must expose the request resources a consumer
    drives: chat.completions, embeddings, models, responses, images, audio.
    These are lazily-built resource objects (cached_property); accessing them
    builds the resource in-memory and does NOT connect."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.OpenAI(api_key="x", base_url="http://localhost:9/v1")

    assert client.chat is not None
    assert client.chat.completions is not None
    assert callable(client.chat.completions.create)

    assert client.embeddings is not None
    assert callable(client.embeddings.create)

    assert client.models is not None
    assert callable(client.models.list)
    assert callable(client.models.retrieve)

    # Responses API + image/audio resources the backend mirrors over HTTP.
    assert client.responses is not None
    assert client.images is not None
    assert client.audio is not None


def test_chat_completions_create_signature(depcheck):
    """chat.completions.create(messages=, model=, temperature=, stream=,
    tools=, timeout=, ...) — pin the params a consumer would pass."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.OpenAI(api_key="x", base_url="http://localhost:9/v1")
    depcheck.assert_params(client.chat.completions.create, CHAT_CREATE_KWARGS)


def test_embeddings_create_signature(depcheck):
    """embeddings.create(input=, model=, timeout=) — pin the embedding params."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.OpenAI(api_key="x", base_url="http://localhost:9/v1")
    depcheck.assert_params(client.embeddings.create, EMBEDDINGS_CREATE_KWARGS)


def test_exception_base_is_openai_error(depcheck):
    """All SDK API errors descend from OpenAIError, so a broad
    `except openai.OpenAIError` keeps catching everything."""
    mod = depcheck.load(IMPORT_NAME)
    base = mod.OpenAIError
    assert issubclass(mod.APIError, base)
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "APIResponseValidationError",
        "RateLimitError",
        "AuthenticationError",
        "BadRequestError",
        "NotFoundError",
        "InternalServerError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
        "ConflictError",
    ):
        assert issubclass(getattr(mod, name), base), f"{name} no longer subclasses OpenAIError"


def test_exception_subclass_of_api_error(depcheck):
    """The request-time errors all subclass APIError (the layer below
    OpenAIError), which is the class consumer code catches for HTTP failures.
    APITimeoutError is the connection-error specialisation."""
    mod = depcheck.load(IMPORT_NAME)
    api_error = mod.APIError
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "APIResponseValidationError",
        "RateLimitError",
        "AuthenticationError",
        "BadRequestError",
        "NotFoundError",
        "InternalServerError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
        "ConflictError",
    ):
        assert issubclass(getattr(mod, name), api_error), f"{name} no longer subclasses APIError"


def test_status_error_specialisations(depcheck):
    """RateLimitError / AuthenticationError / NotFoundError / BadRequestError
    etc. are the HTTP-status specialisations and must subclass APIStatusError,
    so status-specific `except` handlers keep working."""
    mod = depcheck.load(IMPORT_NAME)
    status_error = mod.APIStatusError
    for name in (
        "RateLimitError",
        "AuthenticationError",
        "BadRequestError",
        "NotFoundError",
        "InternalServerError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
        "ConflictError",
    ):
        assert issubclass(getattr(mod, name), status_error), (
            f"{name} no longer subclasses APIStatusError"
        )


def test_timeout_error_is_connection_error(depcheck):
    """APITimeoutError is a kind of APIConnectionError (a network-layer
    failure, not an HTTP-status one); pin that relationship explicitly."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.APITimeoutError, mod.APIConnectionError)
    # ...and APIConnectionError is NOT a status error (no HTTP response).
    assert not issubclass(mod.APIConnectionError, mod.APIStatusError)


def test_responses_tool_item_types_present(depcheck):
    """The middleware renders Responses API server-side tool items
    (web_search_call / file_search_call / computer_call) whose schemas the
    code comments attribute to the openai-python SDK. Pin that the
    types.responses namespace still ships the corresponding tool-call types,
    so the documented wire shape stays anchored to a known SDK version."""
    responses = depcheck.resolve(depcheck.load(IMPORT_NAME), "types.responses")
    names = set(dir(responses))
    # The SDK names these *ToolCall (e.g. ResponseFunctionWebSearch /
    # ResponseFileSearchToolCall / ResponseComputerToolCall). Assert the
    # namespace is populated rather than over-pinning exact class names that
    # the SDK renames between minors; require at least the web-search and
    # computer-call tool types to exist by substring.
    assert any("WebSearch" in n for n in names), (
        "openai.types.responses no longer exposes a web-search tool type"
    )
    assert any("ComputerToolCall" in n or "ComputerCall" in n for n in names), (
        "openai.types.responses no longer exposes a computer-call tool type"
    )
    assert any("FileSearch" in n for n in names), (
        "openai.types.responses no longer exposes a file-search tool type"
    )
