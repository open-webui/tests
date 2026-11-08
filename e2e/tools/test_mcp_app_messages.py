"""Journey: an MCP app shown in a chat asks to send a message, and the person confirms or declines.

New in 0.13.1 (#33907). An MCP app page could not send anything into its chat, so an app button
asking a follow-up question failed. Now the chat shows a "Confirm Prompt from Embed" dialog with
the app's text; on Confirm the text is sent as the next message of the chat and the model
answers it, and the app hears that it went through. On Cancel nothing is sent and the app hears
that it was declined. The app here is `SIGNAL_PAGE` of `harness/mcp_server.py`, which speaks the
MCP Apps handshake by hand.

Discriminates: passes on dev e8c0ad85e and fails on a build of 6113e15af (a branch off dev of
Nov 5, before #33907): no dialog shows when the app asks.
"""

from __future__ import annotations

import secrets

import pytest
from playwright.sync_api import FrameLocator, Page, expect

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import SIGNAL_MESSAGE, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant
from utils.chat_ui import conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def signals_chat(admin, make_user, preserve, upstream):
    """(person, chat id): a chat whose reply shows the signals app of a connected MCP server."""
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"signals_{secrets.token_hex(4)}"
    with serving_mcp(apps=True) as url:
        with admin.client() as client:
            saved = client.post(
                TOOL_SERVERS[1],
                json={
                    "TOOL_SERVER_CONNECTIONS": [
                        mcp_connection(url, server_id, [read_grant(person.id)])
                    ]
                },
            )
        assert saved.status_code == 200, saved.text
        prompt = f"show the harbour signals {secrets.token_hex(4)}"
        upstream.queue(
            reply.tool_call(f"{server_id}_signals", {}, match=reply.answering(prompt)),
            reply.text("Here are the signals.", match=reply.answering(prompt)),
        )
        with person.client() as client:
            turn, _ = ask(client, prompt, tool_ids=[f"server:mcp:{server_id}"])
        yield person, turn.chat_id


def signals_app(page: Page) -> FrameLocator:
    app = page.frame_locator("iframe[title='signals']")
    expect(app.get_by_text("ready", exact=True)).to_be_visible()
    return app


def test_a_message_the_app_asks_to_send_is_sent_once_confirmed(signals_chat, page_for, upstream):
    person, chat_id = signals_chat
    upstream.queue(
        reply.text("Berths B and D are free tonight.", match=reply.answering(SIGNAL_MESSAGE))
    )
    page = page_for(person, path=f"/c/{chat_id}")
    app = signals_app(page)

    app.get_by_role("button", name="Ask the harbour master").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("Confirm Prompt from Embed")
    expect(dialog).to_contain_text(SIGNAL_MESSAGE)
    dialog.get_by_role("button", name="Confirm").click()

    expect_reply(page, "Berths B and D are free tonight.")
    expect(conversation(page).get_by_text(SIGNAL_MESSAGE, exact=True)).to_be_visible()
    assert any(reply.answering(SIGNAL_MESSAGE)(body) for body in upstream.chat_requests())


def test_a_message_the_person_declines_is_not_sent(signals_chat, page_for, upstream):
    person, chat_id = signals_chat
    before = len(upstream.chat_requests())
    page = page_for(person, path=f"/c/{chat_id}")
    app = signals_app(page)

    app.get_by_role("button", name="Ask the harbour master").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text(SIGNAL_MESSAGE)
    dialog.get_by_role("button", name="Cancel").click()

    expect(app.get_by_text("cancelled", exact=True)).to_be_visible()
    expect(conversation(page).get_by_text(SIGNAL_MESSAGE, exact=True)).to_have_count(0)
    assert len(upstream.chat_requests()) == before, "a declined message reached the model"
