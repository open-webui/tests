"""Journey: a reply's reasoning stays folded under how long the model thought, also after a reload.

A model that reasons before it answers shows a folded block above the answer, labelled with how
long it thought. Opening the block shows the reasoning drawn as markdown (bold words, a list). A
reload shows the same label with the block folded again. An HTML block written in the reasoning is
part of the thinking only, in a reply streamed now and in one stored with its thoughts inside its
text as older versions kept them: the pane the answer's own HTML block opens holds that one page,
also once the reasoning is opened. Folding on the open chat and Always Expand Details are covered in
test_chat_journeys.py and test_settings_interface_effects.py.

`test_an_html_block_in_the_reasoning_is_no_artifact_of_its_own` failed on dev 76ad6f97c in CI with
the reply left blank although the server answered it, and passed 3 of 3 locally: since de73bb830 the
reply in a new chat sometimes stays blank until a reload although the server saved it whole
(open-webui/open-webui#32091).

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: the label's duration dropped) the reload test goes red, and in the `rendering-front2` copy
(reasoning left in the text the artifacts are read from) the stored reply case goes red, and in
the `rendering-front3` copy (artifacts read from every output item, thoughts included) the
streamed case goes red. On dev f659c0d12 both cases of the reasoning artifact test go red: the
Artifacts view the answer's HTML block opened took the chat's place and hid the reply
(open-webui/open-webui#33878).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`; the stored reply
dragged into a tab beside a new chat). The page has one artifacts pane, which shows the main
chat's artifacts, so the reasoning artifact test is red in a side chat on dev c15f28649 and on
dev e791cbdde (open-webui/open-webui#34026, fix PR open-webui/open-webui#34166).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

THOUGHTS = "Checking **two** things:\n\n- the tide table\n- the wind forecast"
# the provider pauses this long between the reasoning and the answer
THINKING_SECONDS = 2.2


def thought_label(chat: ChatSurface) -> Locator:
    return last_reply(chat).get_by_text(re.compile(r"^Thought for"))


def expect_folded_then_drawn(chat: ChatSurface) -> None:
    expect(thought_label(chat)).to_have_text(re.compile(r"^Thought for [23] seconds$"))
    tide = conversation(chat).get_by_text("the tide table")
    expect(tide).to_be_hidden()

    thought_label(chat).click()

    thoughts = last_reply(chat).get_by_role("listitem")
    expect(thoughts).to_have_text(["the tide table", "the wind forecast"])
    expect(last_reply(chat).get_by_role("strong")).to_have_text("two")
    expect(last_reply(chat)).not_to_contain_text("**")


def test_the_reasoning_is_folded_under_how_long_it_took_and_again_after_a_reload(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    prompt = "should we sail today?"
    upstream.queue(
        reply.text(
            "Yes, sail before noon.",
            reasoning=THOUGHTS,
            chunk_delay=THINKING_SECONDS,
            match=reply.answering(prompt),
        )
    )
    send(chat, prompt)
    expect(last_reply(chat)).to_contain_text("Yes, sail before noon.", timeout=REPLY_TIMEOUT_MS)

    expect_folded_then_drawn(chat)
    chat = chat.reload()
    expect_reply(chat, "Yes, sail before noon.")
    expect_folded_then_drawn(chat)


DRAFT_THOUGHT = "Maybe a page like this:\n\n```html\n<h1>draft in thought</h1>\n```"
FINAL_ANSWER = "Here is the page.\n\n```html\n<h1>final page</h1>\n```\n\nPage done."
# how a reply from before reasoning was kept apart from the answer stores its thoughts
STORED_WITH_DETAILS = (
    '<details type="reasoning" done="true" duration="2">\n'
    "<summary>Thought for 2 seconds</summary>\n"
    f"{DRAFT_THOUGHT}\n</details>\n{FINAL_ANSWER}"
)


def reply_streamed_now(chat_for, account, upstream) -> ChatSurface:
    chat = chat_for(account)
    prompt = "think about a page first"
    upstream.queue(reply.text(FINAL_ANSWER, reasoning=DRAFT_THOUGHT, match=reply.answering(prompt)))
    send(chat, prompt)
    return chat


def reply_stored_earlier(chat_for, account, upstream) -> ChatSurface:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "think about a page first"},
                {"role": "assistant", "content": STORED_WITH_DETAILS},
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    return chat


@pytest.mark.parametrize("open_reply", [reply_streamed_now, reply_stored_earlier])
def test_an_html_block_in_the_reasoning_is_no_artifact_of_its_own(
    chat_for, make_user, upstream, open_reply
):
    chat = open_reply(chat_for, make_user(), upstream)

    expect_reply(chat, "Page done.")
    pane = chat.page.locator("#artifacts-container")
    expect(pane.frame_locator("iframe").locator("h1")).to_have_text("final page")
    expect(pane).to_contain_text("Version 1 of 1")
    last_reply(chat).get_by_text(re.compile(r"^Thought")).click()
    expect(last_reply(chat).get_by_text("draft in thought")).to_be_visible()
    expect(pane).to_contain_text("Version 1 of 1")
