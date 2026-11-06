"""Regression: an MCP tool call that hung kept the chat waiting past the tool server timeout.

PR open-webui/open-webui#31641 (c1f245845, issue #31640): a tool call to an MCP server had no time
limit of its own, so a server that never answered kept the chat waiting until a proxy or the
server closed the connection, even with `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` set. The call now
stops after that many seconds (falling back to `AIOHTTP_CLIENT_TIMEOUT`) and the model is told the
tool failed, the way an OpenAPI tool server's call already ends.

The MCP server's `ponder` tool waits as long as it is asked to. An instance with a two-second
limit, and one with only the general client timeout at two seconds, get a call that would take
far longer. Nearby: a call inside the limit still answers, a call on an instance with neither
variable set is left to finish, and an OpenAPI tool server that hangs is cut off at the same limit.

Discriminates: passes on dev a5bc78300, fails with c1f245845 reverted (the MCP call runs to the
end, so the model gets the tool's late answer and both limited cases go red).
"""

from __future__ import annotations

import json
import secrets
import time

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.listener import json_answer
from harness.mcp_server import PONDERED, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

LIMIT_SECONDS = 2
HUNG_SECONDS = 20
TOOL_SERVER_LIMIT = {"AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER": str(LIMIT_SECONDS)}
GENERAL_LIMIT = {"AIOHTTP_CLIENT_TIMEOUT": str(LIMIT_SECONDS)}
OPENAPI_PREFIX = "/slow-api"


def _save_tool_servers(instance, preserve, *connections: dict) -> None:
    preserve(TOOL_SERVERS, on=instance)
    with admin_of(instance).client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": list(connections)})
    assert saved.status_code == 200, saved.text


def _timed_tool_result(instance, tool_id: str, tool: str, arguments: dict) -> tuple[str, float]:
    """Have the model call `tool`; returns what the model was told and how long the chat took."""
    person = create_user(instance)
    prompt = f"use {tool}"
    this_test = reply.answering(prompt)
    instance.upstream.queue(
        reply.tool_call(tool, arguments, match=this_test), reply.text("Noted.", match=this_test)
    )
    started = time.monotonic()
    with person.client() as client:
        ask(client, prompt, tool_ids=[tool_id])
    elapsed = time.monotonic() - started
    sent = [body for body in instance.upstream.chat_requests() if this_test(body)]
    follow_up = reply.conversation(sent[-1])
    [tool_message] = [entry for entry in follow_up if entry["role"] == "tool"]
    return tool_message["content"], elapsed


def _ponder(instance, preserve, seconds: float) -> tuple[str, float]:
    server_id = f"thinker_{secrets.token_hex(4)}"
    with serving_mcp(slow=True) as url:
        _save_tool_servers(instance, preserve, mcp_connection(url, server_id, [read_grant("*")]))
        return _timed_tool_result(
            instance, f"server:mcp:{server_id}", f"{server_id}_ponder", {"seconds": seconds}
        )


def _assert_cut_off(answered: str, elapsed: float) -> None:
    assert PONDERED not in answered, (
        f"the MCP call ran past the {LIMIT_SECONDS}s limit to its late answer (#31640)"
    )
    assert "error" in json.loads(answered), f"the timeout was not reported as an error: {answered}"
    assert elapsed < HUNG_SECONDS, f"the chat waited {elapsed:.1f}s for a hung MCP tool"


@pytest.mark.slow
def test_a_hung_mcp_tool_call_stops_at_the_tool_server_timeout(instance_with, preserve):
    limited = instance_with(TOOL_SERVER_LIMIT)

    answered, elapsed = _ponder(limited, preserve, HUNG_SECONDS)

    _assert_cut_off(answered, elapsed)


@pytest.mark.slow
def test_a_hung_mcp_tool_call_stops_at_the_general_client_timeout(instance_with, preserve):
    limited = instance_with(GENERAL_LIMIT)

    answered, elapsed = _ponder(limited, preserve, HUNG_SECONDS)

    _assert_cut_off(answered, elapsed)


@pytest.mark.slow
def test_an_mcp_tool_call_inside_the_limit_answers(instance_with, preserve):
    limited = instance_with(TOOL_SERVER_LIMIT)

    answered, _ = _ponder(limited, preserve, 0.2)

    assert PONDERED in answered


def test_without_a_timeout_a_slow_mcp_tool_call_is_left_to_finish(instance, upstream, preserve):
    answered, elapsed = _ponder(instance, preserve, LIMIT_SECONDS + 2)

    assert PONDERED in answered
    assert elapsed >= LIMIT_SECONDS + 2


def _hung_operation(request):
    time.sleep(HUNG_SECONDS)
    return json_answer({"thought": PONDERED})(request)


@pytest.mark.slow
def test_a_hung_openapi_tool_call_stops_at_the_same_limit(instance_with, preserve, listener):
    limited = instance_with(TOOL_SERVER_LIMIT)
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Slow", "version": "1"},
        "paths": {
            "/ponder": {
                "get": {"operationId": "ponder", "responses": {"200": {"description": "ok"}}}
            }
        },
    }
    listener.route("GET", f"{OPENAPI_PREFIX}/openapi.json", json_answer(spec))
    listener.route("GET", f"{OPENAPI_PREFIX}/ponder", _hung_operation)
    connection = {
        "url": f"{listener.base_url}{OPENAPI_PREFIX}",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True, "access_grants": [read_grant("*")]},
        "info": {"id": "slow", "name": "Slow"},
    }
    _save_tool_servers(limited, preserve, connection)

    answered, elapsed = _timed_tool_result(limited, "server:slow", "ponder", {})

    _assert_cut_off(answered, elapsed)
