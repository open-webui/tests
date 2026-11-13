"""Regression: an image inside an action's result reached the model as base64 text.

open-webui 0.11.4 fix `afda09454` (PR #29665, issue #29208): `process_tool_result` moved an image
out of the model's context only when the whole result was one image data URI. It now walks the
result after every tool type's own branch and attaches each string that is entirely one image
data URI, leaving an `[image]` marker in the text.

The Python tool, tool server, terminal and MCP paths are pinned over HTTP in
integration/security/test_v0114_tool_result_images.py. An action's `(data, headers)` result stays
here because nothing outside can see it: the action caller discards the processed text and files
unless the result carried embeds. The path involves no I/O, so nothing is mocked.

Discriminates: passes on ef67cc3fa, fails with `afda09454` reverted (the base64 image stays in the
text and nothing is attached).
"""

from __future__ import annotations

import base64
import inspect
import json

import pytest

pytestmark = pytest.mark.regression

PNG = "data:image/png;base64," + base64.b64encode(b"fake-png").decode()
SERVER_HEADERS = {"Content-Type": "application/json"}


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


async def settled(value):
    return await value if inspect.isawaitable(value) else value


@pytest.mark.asyncio
async def test_an_image_inside_an_actions_result_is_attached(middleware_module):
    tool_type, raw_result = "action", ({"graph": PNG}, SERVER_HEADERS)
    text, files, embeds = await settled(
        middleware_module.process_tool_result(
            request=None,
            tool_function_name="make_graph",
            tool_result=raw_result,
            tool_type=tool_type,
            metadata={"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"},
            user=None,
        )
    )

    assert PNG not in text, f"the {tool_type} tool's image reached the model as text"
    assert json.loads(text) == {"graph": "[image]"}
    assert files == [{"type": "image", "url": PNG}]
    assert embeds == []
