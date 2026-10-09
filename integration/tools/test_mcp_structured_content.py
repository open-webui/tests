"""Regression: the model never saw the structured data an MCP tool returned.

Fix 22102e4a2 (MCP client). An MCP tool can answer with a short human summary in `content` and
its real data in `structuredContent`. Only the `content` items reached the model, so a weather
tool's forecast arrived as the one line summary and the numbers were lost. The structured data
is now added to the tool result as JSON, unless a text item already carries the same JSON.

Discriminates: passes on dev 1c010b438, fails with 22102e4a2 reverted (the follow-up request
carries the summary but not the forecast).
"""

from __future__ import annotations

import json
import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import (
    CONDITIONS,
    FORECAST,
    FORECAST_SUMMARY,
    HEADLINE,
    TOOL_SERVERS,
    mcp_connection,
    serving_mcp,
)
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def weather(admin, make_user, preserve):
    """(person, server id): an MCP weather server that `person` may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"weather_{secrets.token_hex(4)}"
    with serving_mcp(structured=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


def tool_result_sent_to_the_model(person, upstream, server_id: str, tool: str) -> str:
    """Have the model call one MCP tool; returns the tool message of the follow-up request."""
    upstream.queue(reply.tool_call(f"{server_id}_{tool}", {"city": "Graz"}), reply.text("Noted."))
    with person.client() as client:
        ask(client, f"use {tool} for Graz", tool_ids=[f"server:mcp:{server_id}"])
    follow_up = upstream.chat_requests()[-1]["messages"]
    tool_messages = [entry["content"] for entry in follow_up if entry["role"] == "tool"]
    assert len(tool_messages) == 1, f"{tool} did not run once: {follow_up}"
    return tool_messages[0]


def values_in(content: str) -> list:
    """Every value in the tool message: the message itself and, when it is JSON, all it nests."""
    try:
        pending = [json.loads(content)]
    except json.JSONDecodeError:
        return [content]
    values = []
    while pending:
        value = pending.pop()
        values.append(value)
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return values


def test_the_structured_forecast_is_sent_to_the_model(weather, upstream):
    person, server_id = weather

    content = tool_result_sent_to_the_model(person, upstream, server_id, "forecast")

    values = values_in(content)
    assert FORECAST in values, content
    assert FORECAST_SUMMARY in values, content


def test_structured_data_already_in_the_text_is_sent_once(weather, upstream):
    person, server_id = weather

    content = tool_result_sent_to_the_model(person, upstream, server_id, "conditions")

    assert values_in(content).count(CONDITIONS) == 1, content


def test_a_plain_text_result_is_sent_unchanged(weather, upstream):
    person, server_id = weather

    content = tool_result_sent_to_the_model(person, upstream, server_id, "headline")

    assert content == HEADLINE
