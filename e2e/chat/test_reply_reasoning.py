"""Journey: a reply's reasoning stays folded under how long the model thought, also after a reload.

A model that reasons before it answers shows a folded block above the answer, labelled with how
long it thought. Opening the block shows the reasoning drawn as markdown (bold words, a list). A
reload shows the same label with the block folded again. An HTML block written in the reasoning is
part of the thinking only, in a reply streamed now and in one stored with its thoughts inside its
text as older versions kept them: the pane the answer's own HTML block opens holds that one page,
also once the reasoning is opened. Folding on the open chat and Always Expand Details are covered in
test_chat_journeys.py and test_settings_interface_effects.py.

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: the label's duration dropped) the reload test goes red, and in the `rendering-front2` copy
(reasoning left in the text the artifacts are read from) the stored reply case goes red, and in
the `rendering-front3` copy (artifacts read from every output item, thoughts included) the
streamed case goes red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

THOUGHTS = "Checking **two** things:\n\n- the tide table\n- the wind forecast"
# the provider pauses this long between the reasoning and the answer
THINKING_SECONDS = 2.2


def thought_label(page: Page) -> Locator:
    return last_reply(page).get_by_text(re.compile(r"^Thought for"))


def expect_folded_then_drawn(page: Page) -> None:
    expect(thought_label(page)).to_have_text(re.compile(r"^Thought for [23] seconds$"))
    tide = conversation(page).get_by_text("the tide table")
    expect(tide).to_be_hidden()

    thought_label(page).click()

    thoughts = last_reply(page).get_by_role("listitem")
    expect(thoughts).to_have_text(["the tide table", "the wind forecast"])
    expect(last_reply(page).get_by_role("strong")).to_have_text("two")
    expect(last_reply(page)).not_to_contain_text("**")


def test_the_reasoning_is_folded_under_how_long_it_took_and_again_after_a_reload(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    prompt = "should we sail today?"
    upstream.queue(
        reply.text(
            "Yes, sail before noon.",
            reasoning=THOUGHTS,
            chunk_delay=THINKING_SECONDS,
            match=reply.answering(prompt),
        )
    )
    send(page, prompt)
    expect(last_reply(page)).to_contain_text("Yes, sail before noon.", timeout=REPLY_TIMEOUT_MS)

    expect_folded_then_drawn(page)
    page.reload()
    expect_reply(page, "Yes, sail before noon.")
    expect_folded_then_drawn(page)


DRAFT_THOUGHT = "Maybe a page like this:\n\n```html\n<h1>draft in thought</h1>\n```"
FINAL_ANSWER = "Here is the page.\n\n```html\n<h1>final page</h1>\n```\n\nPage done."
# how a reply from before reasoning was kept apart from the answer stores its thoughts
STORED_WITH_DETAILS = (
    '<details type="reasoning" done="true" duration="2">\n'
    "<summary>Thought for 2 seconds</summary>\n"
    f"{DRAFT_THOUGHT}\n</details>\n{FINAL_ANSWER}"
)


def reply_streamed_now(page_for, account, upstream) -> Page:
    page = page_for(account)
    prompt = "think about a page first"
    upstream.queue(reply.text(FINAL_ANSWER, reasoning=DRAFT_THOUGHT, match=reply.answering(prompt)))
    send(page, prompt)
    return page


def reply_stored_earlier(page_for, account, upstream) -> Page:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "think about a page first"},
                {"role": "assistant", "content": STORED_WITH_DETAILS},
            ],
        )
    page = page_for(account)
    page.goto(f"/c/{chat_id}")
    return page


@pytest.mark.parametrize("open_reply", [reply_streamed_now, reply_stored_earlier])
def test_an_html_block_in_the_reasoning_is_no_artifact_of_its_own(
    page_for, make_user, upstream, open_reply
):
    page = open_reply(page_for, make_user(), upstream)

    expect_reply(page, "Page done.")
    pane = page.locator("#artifacts-container")
    expect(pane.frame_locator("iframe").locator("h1")).to_have_text("final page")
    expect(pane).to_contain_text("Version 1 of 1")
    last_reply(page).get_by_text(re.compile(r"^Thought")).click()
    expect(last_reply(page).get_by_text("draft in thought")).to_be_visible()
    expect(pane).to_contain_text("Version 1 of 1")
