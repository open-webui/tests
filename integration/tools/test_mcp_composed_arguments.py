"""Regression: an MCP tool whose input schema offers its arguments in `anyOf` got none of them.

Discussion open-webui/open-webui#32898. A tool can describe its input as alternatives, a wiki page
by `id` or by `path` plus `locale`, with no top-level `properties`. Open WebUI kept only the
arguments named in the top-level `properties`, so the tool was called with `{}` and the server
rejected the call. Both the native tool calls and the legacy prompt-based ones did this.

Discriminates: red on dev 63eed95c8 (the tool receives `{}` on both paths), green with the
fix 0d72d76ca.
"""

from __future__ import annotations

import json
import re
import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def wiki(admin, make_user, preserve):
    """(person, server id): an MCP wiki server that `person` may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"wiki_{secrets.token_hex(4)}"
    with serving_mcp(composed=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


BRANCHES = [{"id": 169}, {"path": "guides/setup", "locale": "en"}]


@pytest.mark.parametrize("arguments", BRANCHES, ids=["id", "path"])
def test_arguments_from_an_any_of_branch_reach_the_tool(wiki, upstream, arguments):
    person, server_id = wiki
    upstream.queue(reply.tool_call(f"{server_id}_find_page", arguments), reply.text("Found it."))

    with person.client() as client:
        ask(client, "open the page", tool_ids=[f"server:mcp:{server_id}"])

    follow_up = reply.conversation(upstream.chat_requests()[-1])
    [content] = [entry["content"] for entry in follow_up if entry["role"] == "tool"]
    assert json.loads(content) == arguments


@pytest.mark.parametrize("arguments", BRANCHES, ids=["id", "path"])
def test_legacy_tool_calling_passes_the_branch_arguments_too(wiki, upstream, arguments):
    person, server_id = wiki
    chosen = {"tool_calls": [{"name": f"{server_id}_find_page", "parameters": arguments}]}
    upstream.queue(reply.text(json.dumps(chosen)), reply.text("Found it."))

    with person.client() as client:
        ask(
            client,
            "open the page",
            tool_ids=[f"server:mcp:{server_id}"],
            params={"function_calling": "legacy"},
        )

    prompt = upstream.chat_requests()[-1]["messages"][-1]["content"]
    [result] = re.findall(r"<source [^>]*find_page[^>]*>(.*?)</source>", prompt, re.DOTALL)
    assert json.loads(result) == arguments
