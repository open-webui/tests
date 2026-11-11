"""Regression: a tool's inline HTML embed vanished, or changed, when the HTML held entities.

Fix PR open-webui/open-webui#31390 (issue open-webui/open-webui#28085) in the chat view. A tool
that returns an HTMLResponse with `Content-Disposition: inline` shows its page as an embed under
the reply. The view decoded HTML entities in the embeds, the arguments and the file links, which
is only right for chats saved by older versions. So an entity such as `&quot;` broke the embed
list and nothing was shown, and `&amp;` or `&lt;` in the page arrived as a real `&` or `<`. The
embed now shows exactly what the tool returned.

A workspace tool returns a page with entities, the model calls it, and the embed is read from
inside its iframe.

Discriminates: passes on the dev a5bc78300 build; with the encoding of the embeds reverted (the
mutation build) the `&quot;` page shows no embed and the `&lt;b&gt;` page gains a real bold element.

Every test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`). A side chat showed no embed a tool returned
(open-webui/open-webui#34015, fix PR open-webui/open-webui#34064): on dev c15f28649 the side chat
variants are red.

A prompt an embed submits to the chat is asked, once confirmed, in the chat the embed is in, with
that chat's history, and never in another chat on the page. The PR names it as a difference that
an embed in a side chat still sends its prompt to the main chat; the test pins what the main chat
does. It is red in every side chat on dev c15f28649 (#34015, no embed to click) and on dev
e791cbdde, which holds #34064: the confirmed prompt is asked in the main chat. The fix for
open-webui/open-webui#34152 (branch fix-embed-prompt-sent-twice) hands each embed's prompt to the
chat that shows it; every variant passes on its build and the side chat ones stay red on dev
262a8c6f7 until it merges.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import python_tool
from utils.cached_chat import turn_on_tool
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = '''
import json

from fastapi.responses import HTMLResponse


class Tools:
    def show_quoted(self) -> HTMLResponse:
        """Show the quoted page."""
        page = '<p id="probe" title="a &quot;b&quot;">He said &quot;hi&quot; &amp; left</p>'
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"})

    def show_escaped(self) -> HTMLResponse:
        """Show the escaped page."""
        page = '<p id="probe">Fish &amp;amp; chips &lt;b&gt;plain&lt;/b&gt;</p>'
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"})

    def show_asker(self, prompt: str) -> HTMLResponse:
        """Show a page that asks the chat a question."""
        message = json.dumps({"type": "input:prompt:submit", "text": prompt})
        page = (
            "<button id='ask'>Ask the chat</button><script>"
            "ask.onclick = () => parent.postMessage(" + message + ", '*');</script>"
        )
        return HTMLResponse(content=page, headers={"Content-Disposition": "inline"})
'''


@pytest.fixture(scope="module")
def embed_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Page embedder") as tool_id:
        yield tool_id


def call_tool(chat: ChatSurface, upstream, method: str, arguments: dict | None = None) -> None:
    turn_on_tool(chat, "Page embedder")
    upstream.queue(
        reply.tool_call(method, arguments or {}, match=reply.answering("show the page")),
        reply.text("Page shown.", match=reply.answering("show the page")),
    )
    send(chat, "show the page")
    expect_reply(chat, "Page shown.")


# the model count cannot matter: the embed belongs to the one reply that called the tool
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_an_embed_with_quote_entities_is_shown_as_the_tool_wrote_it(
    chat_for, make_user, upstream, embed_tool
):
    """In a side chat: open-webui/open-webui#34015."""
    chat = chat_for(make_user())

    call_tool(chat, upstream, "show_quoted")

    probe = last_reply(chat).frame_locator("iframe").locator("#probe")
    expect(probe, "the tool's embed vanished").to_be_visible()
    expect(probe).to_have_text('He said "hi" & left')
    expect(probe).to_have_attribute("title", 'a "b"')


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_an_embed_keeps_escaped_markup_as_text(chat_for, make_user, upstream, embed_tool):
    """In a side chat: open-webui/open-webui#34015."""
    chat = chat_for(make_user())

    call_tool(chat, upstream, "show_escaped")

    probe = last_reply(chat).frame_locator("iframe").locator("#probe")
    expect(probe, "the tool's embed vanished").to_be_visible()
    expect(probe).to_have_text("Fish &amp; chips <b>plain</b>")
    expect(probe.locator("b"), "escaped markup turned into a real element").to_have_count(0)


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_prompt_an_embed_submits_is_asked_in_the_chat_the_embed_is_in(
    chat_for, make_user, upstream, embed_tool
):
    """In a side chat: open-webui/open-webui#34015, then the prompt goes to the main chat
    (open-webui/open-webui#34152)."""
    chat = chat_for(make_user())
    prompt = f"How high is the tide at noon? ({uuid.uuid4().hex[:8]})"
    call_tool(chat, upstream, "show_asker", {"prompt": prompt})
    upstream.queue(reply.text("About four metres.", match=reply.answering(prompt)))

    asker = last_reply(chat).frame_locator("iframe").get_by_role("button", name="Ask the chat")
    expect(asker, "the tool's embed vanished").to_be_visible()
    asker.click()
    dialogs = chat.page.get_by_role("dialog").filter(has_text="Confirm Prompt from Embed")
    expect(dialogs, "not exactly one chat asked to confirm the embed's prompt").to_have_count(1)
    expect(dialogs).to_contain_text(prompt)
    dialogs.get_by_role("button", name="Confirm").click()

    expect_reply(chat, "About four metres.")
    expect(conversation(chat).locator(".chat-user").last).to_contain_text(prompt)
    asked = json.dumps(
        next(filter(reply.answering(prompt), reversed(upstream.chat_requests())))["messages"]
    )
    assert "show the page" in asked, f"the prompt was sent without its chat's history: {asked}"
    panel_id = chat.panel.get_attribute("id")
    other_chats = chat.page.locator(f"[role='tabpanel']:not([id='{panel_id}'])")
    expect(other_chats.get_by_text(prompt), "the prompt was asked in another chat").to_have_count(0)
