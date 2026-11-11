"""Journey: a tool server call that fails or hangs, as the person in the chat sees it.

A person picks a tool server for the chat and the model calls one of its tools. When the MCP tool
raises, the call opens to the tool's error and the model, told of it, still finishes the reply.
An OpenAPI operation that answers 500 shows the status and the server's words. An MCP call that
would take a minute is cut off at `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` and the reply arrives
within it. integration/tools/test_mcp_tool_call_timeout.py pins the limits over HTTP.

Discriminates: on dev ebc6add67, in a backend copy whose MCP tool call turns the tool's error into
an empty result, the MCP failure test fails; with the status dropped from a failed OpenAPI call, the
OpenAPI test fails; with the MCP call's time limit removed, the hung call is still running when
the test gives up and the timeout test fails.

All three tests run in the main chat and in a side chat, which picks the tool server in its own
More menu. The side chat variants are green on dev c15f28649 and on head e791cbdde.
"""

from __future__ import annotations

import dataclasses
import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import text_answer
from harness.mcp_server import CAPSIZE_ERROR, PONDERED, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.openapi_server import openapi_connection, serve_openapi
from utils.chat_surface import MAIN_CHAT, ChatOpener
from utils.chat_ui import expect_reply, send
from utils.tool_servers import pick_tool, tool_output

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE = {"principal_type": "user", "principal_id": "*", "permission": "read"}
TOOL_TIMEOUT = {"AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER": "2"}


def save_connection(admin, connection: dict) -> None:
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text


def call_and_finish(chat, upstream, tool_name: str, arguments: dict, closing: str) -> list[str]:
    """Have the model call `tool_name` and then say `closing`; returns the tool results it got."""
    question = f"try the tool {secrets.token_hex(3)}"
    upstream.queue(
        reply.tool_call(tool_name, arguments, match=reply.answering(question)),
        reply.text(closing, match=reply.answering(question)),
    )
    send(chat, question)
    expect_reply(chat, closing)
    follow_up = upstream.chat_requests()[-1]
    return [entry["content"] for entry in follow_up["messages"] if entry["role"] == "tool"]


def test_an_mcp_tool_that_raises_shows_its_error_and_the_reply_goes_on(
    chat_for, admin, make_user, preserve, upstream
):
    preserve(TOOL_SERVERS)
    server_id = f"harbour_{secrets.token_hex(3)}"
    with serving_mcp(failing=True) as url:
        save_connection(admin, mcp_connection(url, server_id, [EVERYONE]))
        chat = chat_for(make_user())
        pick_tool(chat, server_id)
        results = call_and_finish(
            chat, upstream, f"{server_id}_capsize", {}, "The boat trip did not work out."
        )

    assert any(CAPSIZE_ERROR in result for result in results), results
    expect(tool_output(chat, f"{server_id}_capsize")).to_contain_text(CAPSIZE_ERROR)


def test_an_openapi_operation_that_fails_shows_the_status(
    chat_for, admin, make_user, preserve, upstream, listener
):
    preserve(TOOL_SERVERS)
    name = f"Lock Keeper {secrets.token_hex(3)}"
    serve_openapi(listener, name, {"open_lock": text_answer("lock gate jammed", "text/plain", 500)})
    save_connection(admin, openapi_connection(listener.base_url, name, [EVERYONE]))
    chat = chat_for(make_user())
    pick_tool(chat, name)

    call_and_finish(chat, upstream, "open_lock", {}, "The lock would not open.")

    output = tool_output(chat, "open_lock")
    expect(output).to_contain_text("500")
    expect(output).to_contain_text("lock gate jammed")


@pytest.fixture
def limited(instance_with, preserve):
    """An instance whose tool server calls stop after two seconds."""
    extra = instance_with(TOOL_TIMEOUT)
    if not extra.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve(TOOL_SERVERS, on=extra)
    return extra


@pytest.fixture
def limited_chat_for(chat_for, limited, preserve) -> ChatOpener:
    """`chat_for` on that instance, with side chats allowed there."""
    admin = admin_of(limited)
    if chat_for.place != MAIN_CHAT:
        preserve("permissions", on=limited)
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["chat"].update(side_chat=True)
            saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text
    return dataclasses.replace(chat_for, upstream=limited.upstream, admin=admin)


@pytest.mark.slow
def test_a_hung_mcp_call_is_cut_off_at_the_tool_server_timeout(limited_chat_for, limited):
    server_id = f"slow_{secrets.token_hex(3)}"
    with serving_mcp(slow=True) as url:
        save_connection(admin_of(limited), mcp_connection(url, server_id, [EVERYONE]))
        chat = limited_chat_for(create_user(limited))
        pick_tool(chat, server_id)
        results = call_and_finish(
            chat, limited.upstream, f"{server_id}_ponder", {"seconds": 60}, "It took too long."
        )

    assert PONDERED not in str(results), results
    output = tool_output(chat, f"{server_id}_ponder")
    expect(output).to_contain_text('"error"')
    expect(output).not_to_contain_text(PONDERED)
