"""Regression test for MCP tools/list pagination.

Regression for open-webui/open-webui#29879.

Open WebUI previously fetched only the first page returned by MCP
``tools/list``, ignoring ``nextCursor`` and therefore hiding tools
from subsequent pages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]


def _tool(name: str) -> types.Tool:
    return types.Tool(
        name=name,
        description=f'Description for {name}',
        inputSchema={
            'type': 'object',
            'properties': {},
        },
    )


async def test_mcp_tool_specs_follow_tools_list_pagination(owui_module) -> None:
    """Regression for open-webui/open-webui#29879."""
    module = owui_module('open_webui.utils.mcp.client')

    client = module.MCPClient()

    first_page = types.ListToolsResult(
        tools=[_tool('tool-one')],
        nextCursor='page-2',
    )
    second_page = types.ListToolsResult(
        tools=[_tool('tool-two')],
        nextCursor=None,
    )

    session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=[first_page, second_page])
    )
    client.session = session

    result = await client.list_tool_specs()

    assert result == [
        {
            'name': 'tool-one',
            'description': 'Description for tool-one',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
        {
            'name': 'tool-two',
            'description': 'Description for tool-two',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    ]

    assert session.list_tools.await_count == 2
    session.list_tools.assert_any_await(cursor=None)
    session.list_tools.assert_any_await(cursor='page-2')
