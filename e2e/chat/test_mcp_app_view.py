"""Regression: an MCP app shown in a reply keeps its page while the reply around it goes on.

The chat shows an MCP app's page in a frame under the tool call that returned it. Every update of
the reply rebuilt the app's entry, and the page was loaded again from scratch, losing whatever the
person had done in it. Since b79a778c0 (dev, after 0.13.0) the page is loaded again only when the
app itself changes.

Discriminates: passes on the dev e8c0ad85e build and fails in a frontend copy with b79a778c0
reverted (the frame holds a fresh page once the reply's text arrives).

The test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`). A side chat showed no embed a tool returned
(open-webui/open-webui#34015, fix PR open-webui/open-webui#34064): on dev c15f28649 the side chat
variants are red.
"""

from __future__ import annotations

import secrets
import threading

import pytest
from playwright.sync_api import Frame, expect

from harness import upstream as reply
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant
from utils.cached_chat import turn_on_tool
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "show me the harbour dashboard"


@pytest.fixture
def harbour(admin, make_user, preserve):
    """(person, server id): an MCP app server the person may use."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"harbour_{secrets.token_hex(4)}"
    with serving_mcp(apps=True) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield person, server_id


def app_page(chat: ChatSurface) -> Frame:
    """The document the dashboard's frame holds now."""
    return chat.panel.locator('iframe[title="dashboard"]').element_handle().content_frame()


# the model count cannot matter: the app belongs to the one reply that called the tool
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_the_app_page_is_kept_while_the_reply_goes_on(chat_for, harbour, upstream):
    """In a side chat: open-webui/open-webui#34015."""
    person, server_id = harbour
    chat = chat_for(person)
    turn_on_tool(chat, server_id)
    rest_of_reply = threading.Event()
    upstream.queue(
        reply.tool_call(f"{server_id}_dashboard", {}, match=reply.answering(PROMPT)),
        reply.text("The dashboard is above.", hold=rest_of_reply),
    )
    send(chat, PROMPT)
    frame = chat.panel.frame_locator('iframe[title="dashboard"]')
    expect(frame.get_by_text("Three berths free.")).to_be_visible()
    # stands in for what a person does in the app: state that lives only in this page load
    app_page(chat).evaluate("window.touched = true")

    rest_of_reply.set()

    expect_reply(chat, "The dashboard is above.")
    expect(frame.get_by_text("Three berths free.")).to_be_visible()
    kept = app_page(chat).evaluate("window.touched")
    assert kept is True, "the app's page was loaded again when the reply went on"
