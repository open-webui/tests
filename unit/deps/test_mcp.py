"""Dependency contract: mcp (Model Context Protocol SDK).

Open WebUI talks to external MCP tool servers through this SDK: `utils/mcp/client.py` opens the
Streamable-HTTP transport, wraps it in a `ClientSession` and drives `initialize()`,
`list_tools()`, `call_tool()`, `list_resources()` and `read_resource()`, while `utils/oauth.py`
and `routers/configs.py` reuse its OAuth models.

What the backend imports from the SDK, and the arguments it passes when it calls those names, are
read from the backend with `ast`: an import or keyword upstream adds is checked the day it lands,
and one it drops stops being pinned, so no hand-kept list goes stale (a list here once tracked a
rename the backend never made). The session methods and the result fields the client reads are
pinned by hand, since `ast` cannot tell what `self.session` holds. The SDK connecting to a real
server end to end is integration/deps/test_outbound_stack.py.

Everything is offline: no server is spawned and no transport is opened. Uses the `depcheck`
fixture from unit/deps/conftest.py; skips when mcp is not importable.

Discriminates: in a backend copy importing a name the installed SDK lacks, the import inventory
fails; calling `streamablehttp_client` with a keyword it does not take fails the call check; an SDK
whose `call_tool` stops taking the arguments positionally fails the session check.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "mcp"
DIST_NAME = "mcp"

# How client.py calls the live session: (method, positional arguments, keyword arguments).
SESSION_CALLS = [
    ("initialize", (), {}),
    ("list_tools", (), {"cursor": None}),
    ("call_tool", ("echo", {"text": "hi"}), {}),
    ("list_resources", (), {"cursor": None}),
    ("read_resource", ("file:///notes.txt",), {}),
]


def _is_mcp(module: str | None) -> bool:
    return module == IMPORT_NAME or (module or "").startswith(f"{IMPORT_NAME}.")


def _mcp_imports(backend: Path) -> dict[Path, tuple[ast.Module, dict[str, tuple[str, str]]]]:
    """Per backend file that imports from mcp: its tree and each bound name's (module, name)."""
    found = {}
    for path in sorted((backend / "open_webui").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if IMPORT_NAME not in source:
            continue
        tree = ast.parse(source)
        bound = {
            alias.asname or alias.name: (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_mcp(node.module)
            for alias in node.names
        }
        if bound:
            found[path] = (tree, bound)
    assert found, f"no module under {backend} imports from mcp; retarget this contract"
    return found


def _resolve(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "mcp"


def test_version_reported(depcheck):
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_every_name_the_backend_imports_resolves(depcheck, open_webui_backend):
    depcheck.load(IMPORT_NAME)
    missing = [
        f"{path.relative_to(open_webui_backend)}: from {module} import {name}"
        for path, (_, bound) in _mcp_imports(open_webui_backend).items()
        for module, name in bound.values()
        if not depcheck.has(importlib.import_module(module), name)
    ]
    assert not missing, f"the installed mcp lacks names the backend imports: {missing}"


def _call_problem(target, call: ast.Call) -> str | None:
    """Why the call's arguments would not bind to `target`, or None when they do."""
    positional = [None for argument in call.args if not isinstance(argument, ast.Starred)]
    keywords = {keyword.arg: None for keyword in call.keywords if keyword.arg}
    try:
        inspect.signature(target).bind_partial(*positional, **keywords)
    except TypeError as error:
        return str(error)
    return None


def test_every_call_into_an_imported_name_binds(depcheck, open_webui_backend):
    """`streamablehttp_client(url, headers=..., httpx_client_factory=...)`, `ClientSession(read,
    write)`, `OAuthMetadata.model_validate(...)` and every other call the backend makes into a
    name it imported from mcp still accepts the arguments as passed."""
    depcheck.load(IMPORT_NAME)
    problems = []
    for path, (tree, bound) in _mcp_imports(open_webui_backend).items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id in bound:
                target = _resolve(*bound[function.id])
            elif (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in bound
            ):
                target = getattr(_resolve(*bound[function.value.id]), function.attr)
            else:
                continue
            problem = _call_problem(target, node)
            if problem:
                where = f"{path.relative_to(open_webui_backend)}:{node.lineno}"
                problems.append(f"{where} {ast.unparse(node.func)}(...): {problem}")
    assert not problems, f"backend calls the installed mcp no longer accepts: {problems}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_imported_transports_open_as_async_context_managers(depcheck, open_webui_backend):
    """client.py enters the transport with `exit_stack.enter_async_context(...)`. Calling it
    only builds the context manager; nothing connects until it is entered."""
    depcheck.load(IMPORT_NAME)
    transports = {
        (module, name)
        for _, bound in _mcp_imports(open_webui_backend).values()
        for module, name in bound.values()
        if module == "mcp.client.streamable_http"
    }
    assert transports, "the backend no longer imports a Streamable-HTTP transport; retarget"
    for module, name in sorted(transports):
        opened = _resolve(module, name)("http://127.0.0.1:0/never-connected")
        assert hasattr(opened, "__aenter__") and hasattr(opened, "__aexit__"), name
        close = getattr(getattr(opened, "aclose", lambda: None)(), "close", None)
        if callable(close):
            close()


def test_clientsession_is_async_context_manager(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for dunder in ("__aenter__", "__aexit__"):
        assert callable(getattr(mod.ClientSession, dunder, None)), f"ClientSession.{dunder}"


@pytest.mark.parametrize(
    ("method", "args", "kwargs"), SESSION_CALLS, ids=[call[0] for call in SESSION_CALLS]
)
def test_the_session_calls_client_py_makes_still_bind(depcheck, method, args, kwargs):
    mod = depcheck.load(IMPORT_NAME)
    session_method = getattr(mod.ClientSession, method, None)
    assert inspect.iscoroutinefunction(session_method), f"ClientSession.{method} is not async"
    try:
        inspect.signature(session_method).bind(None, *args, **kwargs)
    except TypeError as error:
        pytest.fail(f"client.py's ClientSession.{method}(...) call no longer binds: {error}")


def test_list_tools_result_shape(depcheck):
    """list_tool_specs() pages with `result.tools` and `result.nextCursor`, then reads each
    tool's name, description and inputSchema."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    tool = types_mod.Tool(name="t", description="d", inputSchema={"type": "object"})
    result = types_mod.ListToolsResult(tools=[tool])
    assert result.tools[0].name == "t"
    assert result.tools[0].description == "d"
    assert result.tools[0].inputSchema == {"type": "object"}
    assert result.nextCursor is None


def test_call_tool_result_shape(depcheck):
    """call_tool() reads result.isError and result.model_dump(mode='json')['content']."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    text = types_mod.TextContent(type="text", text="hello")
    result = types_mod.CallToolResult(content=[text])
    assert not result.isError
    assert result.model_dump(mode="json")["content"][0]["text"] == "hello"


def test_call_tool_result_iserror_true_path(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    text = types_mod.TextContent(type="text", text="boom")
    assert types_mod.CallToolResult(content=[text], isError=True).isError is True


def test_list_resources_result_shape(depcheck):
    """list_resources() does result.model_dump() then `['resources']`."""
    mod = depcheck.load(IMPORT_NAME)
    types_mod = depcheck.resolve(mod, "types")
    assert types_mod.ListResourcesResult(resources=[]).model_dump().get("resources") == []


def test_oauth_client_metadata_subclassable_with_kwargs(depcheck):
    """oauth.py subclasses OAuthClientMetadata, constructs it with client_name/redirect_uris/
    grant_types/response_types and sets token_endpoint_auth_method/scope afterwards."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientMetadata")
    meta = cls(
        client_name="Open WebUI",
        redirect_uris=["https://example.test/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    assert str(meta.redirect_uris[0]).startswith("https://example.test/callback")
    meta.scope = "a b"
    assert meta.scope == "a b"
    meta.token_endpoint_auth_method = "client_secret_post"
    assert meta.token_endpoint_auth_method == "client_secret_post"


def test_oauth_client_information_full_fields(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientInformationFull")
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
    """configs.py and oauth.py validate dynamic-registration answers and build it from a dict."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthClientInformationFull")
    payload = {"client_id": "xyz", "redirect_uris": ["https://example.test/cb"]}
    assert cls.model_validate(payload).client_id == "xyz"
    assert cls(**payload).client_id == "xyz"


def test_oauth_metadata_fields_and_validate(depcheck):
    """OAuthMetadata.model_validate(server_json), then .scopes_supported and
    .token_endpoint_auth_methods_supported are read."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthMetadata")
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
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "shared.auth.OAuthToken")
    assert cls(access_token="tok", token_type="Bearer").access_token == "tok"
