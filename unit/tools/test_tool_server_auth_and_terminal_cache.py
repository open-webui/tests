"""Tool server bearer headers and the terminal server cache, open-webui v0.11.2.

`26f37426b`: `build_tool_server_headers` formatted `Bearer {key}` in every bearer branch, so a
connection without a credential sent a bare `Authorization: Bearer `. All branches now go through
`bearer_auth_header`. `81b9afb73`: `get_terminal_servers` trusted an empty list another worker
cached, hiding configured terminals; it now rebuilds unless no enabled connection has a url.

integration/tools/test_tool_server_auth_and_terminal_cache.py pins both through the API. What
stays here cannot be seen from outside: the header audit covers the session, system OAuth and
OAuth 2.1 branches no test can reach without a keyless credential, and the cache test holds the
other half of `81b9afb73`, that an empty list is still a hit when no connection could serve.

Discriminates: passes on dev bbfa876af; the audit fails with the bearer branch formatting
`f'Bearer {key}'` into `headers['Authorization']` again, and the cache test fails when an empty
cached list is never trusted (every url-less or disabled terminal forces a rebuild and rewrite).
"""

from __future__ import annotations

import ast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.applications import Starlette
from starlette.requests import Request

pytestmark = pytest.mark.regression


def _is_formatted_bearer_header(node: ast.AST) -> bool:
    """`headers['Authorization'] = f'Bearer ...'`, which ships `Bearer ` for an empty token."""
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.JoinedStr):
        return False
    first_part = node.value.values[0] if node.value.values else None
    is_bearer = isinstance(first_part, ast.Constant) and str(first_part.value).startswith("Bearer")
    is_authorization = any(
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "Authorization"
        for target in node.targets
    )
    return is_bearer and is_authorization


def test_no_tool_server_header_formats_a_bearer_token_itself(open_webui_backend):
    tools_source = open_webui_backend / "open_webui" / "utils" / "tools.py"
    tree = ast.parse(tools_source.read_text(encoding="utf-8"))

    offenders = [node.lineno for node in ast.walk(tree) if _is_formatted_bearer_header(node)]

    assert offenders == [], (
        f"utils/tools.py lines {offenders} format a bearer header without `bearer_auth_header`, "
        "so an empty credential goes out as `Bearer `"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connections",
    [
        [],
        [{"id": "terminal-1", "enabled": True}],
        [{"id": "terminal-1", "url": "http://127.0.0.1:9", "enabled": False}],
    ],
    ids=["no-connection", "no-url", "disabled"],
)
async def test_an_empty_cached_list_is_a_hit_when_no_connection_could_serve(
    owui_module, connections
):
    tools = owui_module("open_webui.utils.tools")
    config = owui_module("open_webui.models.config")
    cache = Mock(spec=["get", "set"], get=AsyncMock(return_value="[]"), set=AsyncMock())
    app = Starlette()
    app.state.redis = cache

    async def read_setting(key: str, default=None):
        return connections if key == "terminal_server.connections" else default

    with patch.object(config.Config, "get", read_setting):
        servers = await tools.get_terminal_servers(Request({"type": "http", "app": app}))

    assert servers == []
    cache.set.assert_not_awaited()
