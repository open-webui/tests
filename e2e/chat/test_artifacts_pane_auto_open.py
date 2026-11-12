"""Regression: the artifacts pane closed again right after it opened on its own.

Fix `1058444d7` (PR #31652, issue #31643). When a filter wrote the start of a reply (some text
and an HTML block) before the model answered, the page showed it and opened the pane, but the
server never sent that text as part of the reply's output, so the model's first output replaced
it on the page, the page briefly saw no HTML block and closed the pane. The server now sends the
filter's text to the page before the model's output. API twin:
integration/chat/test_filter_written_reply_start.py.

Fix `f45332499` (PR #31653): independently of filters, the pane could look for artifacts before
the page had picked them up from the reply, find none and close. It now looks again as it opens.

`test_the_pane_a_filter_written_html_block_opens_stays_open` is red on dev 62f70a844: since
de73bb830 the pane closes again right after it opens on a filter-written HTML block
(open-webui/open-webui#32082).

Discriminates: passes on dev 015dbc861 with its build; with `1058444d7` reverted in a backend
copy the filter test goes red (the pane closed while the model's words streamed in, then came
back). The `f45332499` race depends on timing inside the page: with it reverted (the 015dbc861
mutation build) the plain reply test failed 3 of 16 runs and never on the clean build, so it
guards the pane staying open without reliably discriminating that fix.

Both tests run in the main chat and in a side chat (`utils/chat_surface.py`), the plain reply test
in every way a side chat starts and every state it can be in. The page has one artifacts pane,
which shows the main chat's artifacts (open-webui/open-webui#34026; PR open-webui/open-webui#34064
leaves it as it is): the side chat variants are red on dev c15f28649 and on dev e791cbdde. The
pane watcher starts in the open page, so a side chat keeps the tab it was opened in.
"""

from __future__ import annotations

import textwrap

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import artifacts_pane, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FILTER_TEXT = "A preview first.\n\n```html\n<h1>filter card</h1>\n```\n\n"

PREVIEW_FILTER = textwrap.dedent(
    f"""
    class Filter:
        async def inlet(self, body: dict, __event_emitter__) -> dict:
            await __event_emitter__({{"type": "message", "data": {{"content": {FILTER_TEXT!r}}}}})
            return body
    """
).strip()


# counts each time the pane leaves the page after it was shown
PANE_CLOSINGS = """
window.paneClosings = 0;
let paneShown = false;
new MutationObserver(() => {
    const shown = [...document.querySelectorAll('[role="tabpanel"][id^="chat-panel-artifacts"]')]
        .some((pane) => pane.checkVisibility());
    if (paneShown && !shown) window.paneClosings += 1;
    paneShown = shown;
}).observe(document, { childList: true, subtree: true });
"""


def watching_the_pane(chat: ChatSurface) -> ChatSurface:
    chat.page.evaluate(PANE_CLOSINGS)
    return chat


def shown_artifact(page: Page):
    return artifacts_pane(page).frame_locator("iframe").locator("h1")


def test_the_pane_a_filter_written_html_block_opens_stays_open(
    chat_for, admin, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34026."""
    prompt = "build the page behind a filter"
    # a slow stream gives the page time to redraw between the model's pieces
    words = ["The model's ", "own words ", "follow."]
    upstream.queue(reply.text(words, chunk_delay=0.5, match=reply.answering(prompt)))
    with installed_function(admin, PREVIEW_FILTER, is_global=True):
        chat = watching_the_pane(chat_for(make_user()))
        page = chat.page
        send(chat, prompt)
        expect_reply(chat, "The model's own words follow.")

        expect(
            artifacts_pane(page), "the pane closed once the model answered (#31652)"
        ).to_be_visible()
        expect(shown_artifact(page)).to_have_text("filter card")
        assert page.evaluate("window.paneClosings") == 0, "the pane closed after opening (#31652)"


# the model count cannot matter: the pane shows the blocks of the reply that holds them
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_the_pane_a_reply_html_block_opens_stays_open(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34026."""
    prompt = "build the page directly"
    answer = "Here it is.\n\n```html\n<h1>reply card</h1>\n```\n\nThat is the page."
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    chat = watching_the_pane(chat_for(make_user()))
    page = chat.page

    send(chat, prompt)
    expect_reply(chat, "That is the page.")

    expect(artifacts_pane(page)).to_be_visible()
    expect(shown_artifact(page)).to_have_text("reply card")
    assert page.evaluate("window.paneClosings") == 0, "the pane closed after opening (#31653)"
