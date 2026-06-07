"""Dependency contract: mcp (Model Context Protocol SDK).

Open WebUI talks to external MCP "tool servers" through this SDK. The
backend opens a Streamable-HTTP transport, wraps it in a `ClientSession`,
and drives `initialize()` / `list_tools()` / `call_tool()` /
`list_resources()` / `read_resource()` to expose remote MCP tools to the
LLM (see `open_webui/utils/mcp/client.py`). It also reuses the SDK's
OAuth model classes (`mcp.shared.auth.*`) for the dynamic-client and
static-credential OAuth flows in `utils/oauth.py` and `routers/configs.py`,
subclassing `OAuthClientMetadata`/`OAuthClientInformationFull` and calling
`OAuthMetadata.model_validate(...)`.

This module pins the slice of the mcp API the backend actually relies on.
The import paths in particular MOVE between SDK versions
(`mcp.client.session`, `mcp.client.streamable_http`,
`mcp.client.stdio`, `mcp.client.sse`, `mcp.client.auth`,
`mcp.shared.auth`, `mcp.types`), so a bump that relocated or renamed any
of them fails loudly here instead of as an ImportError deep in a
tool-call path. Everything is offline: no MCP server is spawned and no
network transport is opened — we only construct objects and introspect
signatures.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API
surface) + offline behavioural contracts. Uses the `depcheck` fixture
from unit/deps/conftest.py. Skips cleanly when mcp is not importable.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "mcp"
DIST_NAME = "mcp"

# Top-level re-exports the backend imports as `from mcp import ...`.
TOP_LEVEL_SYMBOLS = [
    "ClientSession",
    "StdioServerParameters",
]

# Submodule + dotted symbols the backend references via explicit import
# paths. These relocate between SDK versions — the highest-value checks.
USED_SYMBOLS = [
    # client session (top-level re-export lives here)
    "client.session.ClientSession",
    # transports
    "client.streamable_http.streamable_http_client",
    "client.stdio.stdio_client",
    "client.sse.sse_client",
    # client-side OAuth provider + storage protocol
    "client.auth.OAuthClientProvider",
    "client.auth.TokenStorage",
    # OAuth data models reused by open_webui's oauth/configs
    "shared.auth.OAuthClientInformationFull",
    "shared.auth.OAuthClientMetadata",
    "shared.auth.OAuthToken",
    "shared.auth.OAuthMetadata",
    # protocol types the client surfaces
    "types.Tool",
    "types.CallToolResult",
    "types.TextContent",
    "types.ImageContent",
    "types.ListToolsResult",
    "types.ListResourcesResult",
    "types.ReadResourceResult",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "mcp"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_top_level_symbols_exist(depcheck):
    """`from mcp import ClientSession, StdioServerParameters` must resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_used_symbols_exist(depcheck):
    """Every dotted mcp symbol the codebase imports must still resolve at
    its current import path. A bump that moved/renamed any of them breaks
    open_webui's MCP tool-server and OAuth integration at import time."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_top_level_clientsession_is_session_module_class(depcheck):
    """`from mcp import ClientSession` must be the same class as
    `mcp.client.session.ClientSession` — the backend imports the former
    but the SDK defines it in the latter."""
    mod = depcheck.load(IMPORT_NAME)
    session_cls = depcheck.resolve(mod, "client.session.ClientSession")
    assert mod.ClientSession is session_cls


# --- ClientSession contract -------------------------------------------------


def test_clientsession_constructor_params(depcheck):
    """client.py constructs `ClientSession(read_stream, write_stream)`
    positionally from the transport's (read, write, _) tuple. Those two
    leading parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.ClientSession.__init__,
        ["read_stream", "write_stream"],
    )


def test_clientsession_is_async_context_manager(depcheck):
    """client.py does
    `await exit_stack.enter_async_context(ClientSession(...))`,
    so ClientSession must implement the async context-manager protocol."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.ClientSession))
    for dunder in ("__aenter__", "__aexit__"):
        assert dunder in names, f"ClientSession.{dunder} missing"
        assert callable(getattr(mod.ClientSession, dunder))


def test_clientsession_methods_exist_and_callable(depcheck):
    """The driver calls these five coroutine methods on the live session."""
    mod = depcheck.load(IMPORT_NAME)
    for m in (
        "initialize",
        "list_tools",
        "call_tool",
        "list_resources",
        "read_resource",
    ):
        meth = getattr(mod.ClientSession, m, None)
        assert callable(meth), f"ClientSession.{m} missing/not callable"
        assert inspect.iscoroutinefunction(meth), (
            f"ClientSession.{m} is no longer a coroutine function"
        )


def test_clientsession_call_tool_signature(depcheck):
    """client.py: `await self.session.call_tool(function_name, function_args)`
    — the (name, arguments) positional pair must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ClientSession.call_tool, ["name", "arguments"])


def test_clientsession_list_resources_accepts_cursor(depcheck):
    """client.py: `await self.session.list_resources(cursor=cursor)`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ClientSession.list_resources, ["cursor"])


def test_clientsession_read_resource_accepts_uri(depcheck):
    """client.py: `await self.session.read_resource(uri)`."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ClientSession.read_resource, ["uri"])


# --- Transport context managers ---------------------------------------------


def test_streamable_http_client_signature(depcheck):
    """client.py opens the Streamable-HTTP transport with
    `streamable_http_client(url, headers=..., httpx_client_factory=...)`.
    All three of those parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    fn = depcheck.resolve(mod, "client.streamable_http.streamable_http_client")
    assert callable(fn)
    depcheck.assert_params(fn, ["url", "headers", "httpx_client_factory"])


def test_streamable_http_client_is_context_manager_factory(depcheck):
    """`async with streamable_http_client(...)` is used via enter_async_context;
    calling it (without awaiting/connecting) must yield an async context
    manager object. This does NOT open any network connection.

    The backend uses the `streamable_http_client` spelling (the older
    `streamablehttp_client` remains as a deprecated alias in mcp >= 1.27)."""
    mod = depcheck.load(IMPORT_NAME)
    fn = depcheck.resolve(mod, "client.streamable_http.streamable_http_client")
    cm = fn("http://localhost:0/never-connected")
    try:
        assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__"), (
            "streamable_http_client() no longer returns an async context manager"
        )
    finally:
        aclose = getattr(cm, "aclose", None)
        if callable(aclose):
            gen = aclose()
            close = getattr(gen, "close", None)
            if callable(close):
                close()


def test_stdio_client_signature(depcheck):
    """The stdio transport is invoked as `stdio_client(server_params)`; the
    leading `server` parameter must remain (kept as a contract even though
    the HTTP transport is the primary path)."""
    mod = depcheck.load(IMPORT_NAME)
    fn = depcheck.resolve(mod, "client.stdio.stdio_client")
    assert callable(fn)
    depcheck.assert_params(fn, ["server"])


def test_sse_client_signature(depcheck):
    """The SSE transport is invoked as `sse_client(url, headers=...)`."""
    mod = depcheck.load(IMPORT_NAME)
    fn = depcheck.resolve(mod, "client.sse.sse_client")
    assert callable(fn)
    depcheck.assert_params(fn, ["url", "headers"])


# --- StdioServerParameters --------------------------------------------------


def test_stdio_server_parameters_fields(depcheck):
    """StdioServerParameters is the stdio-transport config model; the
    command/args/env trio must remain modellable."""
    mod = depcheck.load(IMPORT_NAME)
    fields = set(mod.StdioServerParameters.model_fields)
    for f in ("command", "args", "env"):
        assert f in fields, f"StdioServerParameters lost field {f!r}"


def test_stdio_server_parameters_constructible(depcheck):
    """Constructing it with the kwargs a launcher would pass must succeed
    and round-trip the values."""
    mod = depcheck.load(IMPORT_NAME)
    params = mod.StdioServerParameters(
        command="python",
        args=["-m", "server"],
        env={"FOO": "bar"},
    )
    assert params.command == "python"
    assert params.args == ["-m", "server"]
    assert params.env == {"FOO": "bar"}


# --- mcp.types: object field shapes the client reads ------------------------


def test_tool_field_shape(depcheck):
    """list_tool_specs() reads tool.name, tool.description, tool.inputSchema
    and getattr(tool, 'outputSchema', None) off each Tool. Pin that shape
    by constructing a Tool and reading those attributes."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    fields = set(types_mod.Tool.model_fields)
    for f in ("name", "description", "inputSchema", "outputSchema"):
        assert f in fields, f"Tool lost field {f!r}"

    tool = types_mod.Tool(
        name="echo",
        description="Echo a message",
        inputSchema={"type": "object", "properties": {}},
    )
    assert tool.name == "echo"
    assert tool.description == "Echo a message"
    assert tool.inputSchema == {"type": "object", "properties": {}}
    # outputSchema is optional; client.py reads it defensively via getattr.
    assert getattr(tool, "outputSchema", None) is None


def test_list_tools_result_shape(depcheck):
    """list_tool_specs() does `result = await session.list_tools()` then
    `result.tools`. ListToolsResult must expose `.tools` as a list of
    Tool, and survive model construction offline."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    assert "tools" in types_mod.ListToolsResult.model_fields
    tool = types_mod.Tool(name="t", inputSchema={"type": "object"})
    result = types_mod.ListToolsResult(tools=[tool])
    assert isinstance(result.tools, list)
    assert result.tools[0].name == "t"


def test_call_tool_result_shape(depcheck):
    """call_tool() reads result.isError and result.model_dump(mode='json')
    then `result_dict['content']`. CallToolResult must expose `content`
    and `isError`, dump to a dict with a 'content' key, and default
    isError to a falsy value."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    fields = set(types_mod.CallToolResult.model_fields)
    for f in ("content", "isError"):
        assert f in fields, f"CallToolResult lost field {f!r}"

    text = types_mod.TextContent(type="text", text="hello")
    result = types_mod.CallToolResult(content=[text])
    # isError defaults falsy (client.py: `if result.isError: raise`).
    assert not result.isError
    dumped = result.model_dump(mode="json")
    assert isinstance(dumped, dict)
    assert "content" in dumped
    assert dumped["content"][0]["text"] == "hello"


def test_call_tool_result_iserror_true_path(depcheck):
    """The error branch is driven by isError being truthy when set."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    text = types_mod.TextContent(type="text", text="boom")
    result = types_mod.CallToolResult(content=[text], isError=True)
    assert result.isError is True


def test_text_content_shape(depcheck):
    """TextContent carries type=='text' and a `text` string; it appears in
    CallToolResult.content for textual tool output."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    fields = set(types_mod.TextContent.model_fields)
    for f in ("type", "text"):
        assert f in fields, f"TextContent lost field {f!r}"
    tc = types_mod.TextContent(type="text", text="x")
    assert tc.type == "text"
    assert tc.text == "x"


def test_image_content_shape(depcheck):
    """ImageContent carries type=='image', base64 `data`, and `mimeType`;
    it appears in tool output content alongside TextContent."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    fields = set(types_mod.ImageContent.model_fields)
    for f in ("type", "data", "mimeType"):
        assert f in fields, f"ImageContent lost field {f!r}"
    ic = types_mod.ImageContent(type="image", data="aGk=", mimeType="image/png")
    assert ic.type == "image"
    assert ic.data == "aGk="
    assert ic.mimeType == "image/png"


def test_list_resources_result_shape(depcheck):
    """list_resources() does result.model_dump() then `['resources']`;
    ListResourcesResult must expose a `resources` field that dumps to a
    list."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    assert "resources" in types_mod.ListResourcesResult.model_fields
    result = types_mod.ListResourcesResult(resources=[])
    dumped = result.model_dump()
    assert isinstance(dumped, dict)
    assert dumped.get("resources") == []


# --- OAuth models reused by oauth.py / configs.py ---------------------------


def test_oauth_client_metadata_subclassable_with_kwargs(depcheck):
    """oauth.py subclasses OAuthClientMetadata and constructs it with
    client_name/redirect_uris/grant_types/response_types and mutates
    token_endpoint_auth_method/scope afterwards. Pin those fields and
    that the model accepts those kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientMetadata")
    fields = set(cls.model_fields)
    for f in (
        "redirect_uris",
        "grant_types",
        "response_types",
        "scope",
        "client_name",
        "token_endpoint_auth_method",
    ):
        assert f in fields, f"OAuthClientMetadata lost field {f!r}"

    meta = cls(
        client_name="Open WebUI",
        redirect_uris=["https://example.test/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    assert str(meta.redirect_uris[0]).startswith("https://example.test/callback")
    # Fields the code reassigns post-construction must be settable.
    meta.scope = "a b"
    assert meta.scope == "a b"
    meta.token_endpoint_auth_method = "client_secret_post"
    assert meta.token_endpoint_auth_method == "client_secret_post"


def test_oauth_client_information_full_fields(depcheck):
    """oauth.py subclasses OAuthClientInformationFull and constructs it
    with client_id/client_secret/redirect_uris/grant_types/response_types/
    scope/token_endpoint_auth_method, and the codebase also instantiates it
    via `OAuthClientInformationFull(**dict)` and `.model_validate(...)`.
    Pin the credential fields and that it inherits the metadata fields."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientInformationFull")
    fields = set(cls.model_fields)
    for f in (
        "client_id",
        "client_secret",
        "redirect_uris",
        "grant_types",
        "response_types",
        "scope",
        "token_endpoint_auth_method",
    ):
        assert f in fields, f"OAuthClientInformationFull lost field {f!r}"

    info = cls(
        client_id="abc",
        client_secret="shh",
        redirect_uris=["https://example.test/cb"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    assert info.client_id == "abc"
    assert info.client_secret == "shh"


def test_oauth_client_information_full_model_validate(depcheck):
    """configs.py / oauth.py call OAuthClientInformationFull.model_validate(...)
    on dynamic-registration responses and construct it via **dict; verify
    both round-trip the client_id."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientInformationFull")
    assert callable(cls.model_validate)
    payload = {
        "client_id": "xyz",
        "redirect_uris": ["https://example.test/cb"],
    }
    validated = cls.model_validate(payload)
    assert validated.client_id == "xyz"
    via_kwargs = cls(**payload)
    assert via_kwargs.client_id == "xyz"


def test_oauth_metadata_fields_and_validate(depcheck):
    """oauth.py / configs.py call OAuthMetadata.model_validate(server_json)
    and then read .scopes_supported and .token_endpoint_auth_methods_supported.
    Pin those two fields (plus the required `issuer`) and model_validate."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthMetadata")
    fields = set(cls.model_fields)
    for f in (
        "issuer",
        "scopes_supported",
        "token_endpoint_auth_methods_supported",
    ):
        assert f in fields, f"OAuthMetadata lost field {f!r}"

    assert callable(cls.model_validate)
    meta = cls.model_validate(
        {
            "issuer": "https://issuer.test",
            "authorization_endpoint": "https://issuer.test/authorize",
            "token_endpoint": "https://issuer.test/token",
            "scopes_supported": ["openid", "email"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
        }
    )
    assert meta.scopes_supported == ["openid", "email"]
    assert meta.token_endpoint_auth_methods_supported == ["client_secret_post"]


def test_oauth_token_fields(depcheck):
    """OAuthToken is imported by client.py for the token-storage flow; pin
    the standard token fields it carries."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthToken")
    fields = set(cls.model_fields)
    for f in ("access_token", "token_type"):
        assert f in fields, f"OAuthToken lost field {f!r}"
    tok = cls(access_token="tok", token_type="Bearer")
    assert tok.access_token == "tok"


def test_token_storage_protocol_methods(depcheck):
    """client.py imports TokenStorage as the auth provider's persistence
    contract. Whether ABC or Protocol, it must expose the four async
    accessors the SDK's OAuth provider drives."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "client.auth.TokenStorage")
    names = set(dir(cls))
    for m in ("get_tokens", "set_tokens", "get_client_info", "set_client_info"):
        assert m in names, f"TokenStorage lost method {m!r}"


def test_oauth_client_provider_exists(depcheck):
    """OAuthClientProvider is imported by client.py as the httpx auth flow
    for authenticated MCP servers; it must remain a class at this path."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "client.auth.OAuthClientProvider")
    assert inspect.isclass(cls)
