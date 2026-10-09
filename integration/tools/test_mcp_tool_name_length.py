"""Regression: a chat with an MCP tool whose server id and tool name run past 64 characters failed.

Issue open-webui/open-webui#31821, fix e1248e5cf (PR open-webui/open-webui#31822). The tool offered
to the model is named after its server id and its own name, and providers that cap a tool name at
64 characters (OpenAI, Bedrock) refused the whole chat when a long server id pushed it over. A name
over the limit is now cut to 64 and ends in a short hash of the full name, so it stays stable and
two long names that start alike stay apart, and the MCP server is still called by the tool's own
name.

An MCP connection with a 70-character id has its `echo` tool offered; the scripted model calls the
name it was offered and the result must come back. Nearby: a short id keeps the plain name.

Every test here is red on dev 1c010b438: a tool written with the official MCP SDK that returns plain
text also sends it as structured data, and the model gets the text twice, the second time wrapped as
`{"result": ...}` (open-webui/open-webui#32126).

Discriminates: passes on dev b859124f9, fails with e1248e5cf reverted (the offered name is over 64
characters).
"""

from __future__ import annotations

import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

NAME_LIMIT = 64
PHRASE = "the lamp is lit at the pier"


def long_id(tail: str = "") -> str:
    return ("harbour_lighthouse_signal_" + "x" * 45)[: 70 - len(tail)] + tail


def tool_ids(*server_ids: str) -> list[str]:
    return [f"server:mcp:{server_id}" for server_id in server_ids]


def mcp_tools(client, upstream, *server_ids: str) -> set[str]:
    """The names offered with these servers selected beyond those offered without them."""
    offered = offered_tools(client, upstream, tool_ids=tool_ids(*server_ids))
    longest = max(offered, key=len)
    assert len(longest) <= NAME_LIMIT, f"{longest} is {len(longest)} characters"
    return offered - offered_tools(client, upstream)


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def serve(admin, preserve, person):
    """Save MCP connections named by the ids given, all to one echo server; returns the ids."""
    preserve(TOOL_SERVERS)
    with serving_mcp() as url:

        def save(*server_ids: str) -> None:
            connections = [
                mcp_connection(url, server_id, [read_grant(person.id)]) for server_id in server_ids
            ]
            with admin.client() as client:
                saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": connections})
            assert saved.status_code == 200, saved.text

        yield save


def test_a_tool_name_over_the_provider_limit_is_shortened_and_still_runs(upstream, person, serve):
    server_id = long_id()
    serve(server_id)
    full_name = f"{server_id}_echo"
    assert len(full_name) > NAME_LIMIT

    with person.client() as client:
        [name] = mcp_tools(client, upstream, server_id)
        assert name != full_name
        assert name.startswith(server_id[:40])
        assert mcp_tools(client, upstream, server_id) == {name}
        result = run_tool(client, upstream, name, {"text": PHRASE}, tool_ids=tool_ids(server_id))
    assert result == PHRASE


def test_two_long_names_that_start_alike_stay_apart(upstream, person, serve):
    first, second = long_id("_a"), long_id("_b")
    serve(first, second)

    with person.client() as client:
        offered = mcp_tools(client, upstream, first, second)
        assert len(offered) == 2, offered
        for name in offered:
            upstream.queue(reply.tool_call(name, {"text": name}), reply.text("done"))
            ask(client, f"use {name}", tool_ids=tool_ids(first, second))
            results = [
                entry["content"]
                for entry in upstream.chat_requests()[-1]["messages"]
                if entry["role"] == "tool"
            ]
            assert results[-1] == name


def test_a_tool_name_within_the_limit_is_unchanged(upstream, person, serve):
    server_id = f"pier_{secrets.token_hex(4)}"
    serve(server_id)
    name = f"{server_id}_echo"

    with person.client() as client:
        assert mcp_tools(client, upstream, server_id) == {name}
        result = run_tool(client, upstream, name, {"text": PHRASE}, tool_ids=tool_ids(server_id))
    assert result == PHRASE
