"""Regression: connecting an MCP server that asks for many scopes ended in a state error.

Issue open-webui/open-webui#26382, fix cd64930c0 (PR open-webui/open-webui#31894). Pressing a
sign-in-needing MCP server in the Integrations menu sends the browser through its authorization
server and back to the callback. The server here asks for the 42 scopes of Google Workspace's, so
the session cookie that carried the whole authorization URL went past the 4096 bytes Chromium
keeps, Chromium dropped it and the callback sent the person home with "OAuth callback state is
invalid or expired". The person now lands connected and the model's tool call carries their token.

Twin of integration/tools/test_mcp_oauth_long_scope_list.py.

`test_pressing_a_server_that_asks_for_many_scopes_connects_it` is red on dev 1c010b438: a tool
written with the official MCP SDK that returns plain text also sends it as structured data, and the
model gets the text twice, the second time wrapped as `{"result": ...}`
(open-webui/open-webui#32126).

Discriminates: passes on dev b859124f9, fails with cd64930c0 reverted (the browser comes back with
the state error in the address and the server is still not connected).
"""

from __future__ import annotations

import re
import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.mcp_oauth import GOOGLE_WORKSPACE_SCOPES, register_oauth_mcp, serving_protected_mcp
from harness.mcp_server import TOOL_SERVERS
from harness.terminal_server import read_grant
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

PHRASE = "the kettle sings at dawn"


def test_pressing_a_server_that_asks_for_many_scopes_connects_it(
    page_for, make_user, admin, preserve, upstream
):
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"wide_mcp_{secrets.token_hex(4)}"
    with serving_protected_mcp(scopes=GOOGLE_WORKSPACE_SCOPES) as mcp:
        register_oauth_mcp(admin, mcp, server_id, [read_grant(person.id)])
        page = page_for(person)
        expect(chat_input(page)).to_be_visible()

        page.get_by_label("Integrations").click()
        page.get_by_role("button", name=re.compile(r"^Tools")).click()
        page.get_by_role("button", name=server_id).click()

        # the round trip ends at the app: with a token issued, or sent home with an error
        page.wait_for_url(
            lambda url: "error=" in url or bool(mcp.auth_server.issued), timeout=30_000
        )
        expect(chat_input(page)).to_be_visible()
        assert "error=" not in page.url, page.url
        asked = mcp.auth_server.requests_to("/authorize")[-1].query["scope"].split()
        assert asked == GOOGLE_WORKSPACE_SCOPES

        # the return from the sign-in selects the server for the chat
        question = "say it back"
        upstream.queue(
            reply.tool_call(f"{server_id}_echo", {"text": PHRASE}, match=reply.answering(question)),
            reply.text("Done.", match=reply.answering(question)),
        )
        page.get_by_label("Integrations").click()
        page.get_by_role("button", name=re.compile(r"^Tools")).click()
        expect(page.get_by_role("button", name=server_id)).to_have_attribute("aria-pressed", "true")
        page.keyboard.press("Escape")
        send(page, question)
        expect_reply(page, "Done.")

    tool_results = [
        entry["content"]
        for entry in upstream.chat_requests()[-1]["messages"]
        if entry["role"] == "tool"
    ]
    assert tool_results == [PHRASE], tool_results
    assert mcp.presented[-1] == mcp.auth_server.issued[-1]["access_token"]
