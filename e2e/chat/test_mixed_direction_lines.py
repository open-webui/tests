"""Regression: an English line under an Arabic line took the Arabic line's direction.

Issue open-webui/open-webui#31827, fix 983fa6c97 (PR open-webui/open-webui#31868). A message that
mixed right-to-left and left-to-right lines gave every line the direction of the first one, in
the chat input and in the sent message, so "Hello, how are you?" under an Arabic line showed as
"?Hello, how are you" on the wrong side. Each line now picks its own direction.

The message is typed and sent in the same page, so the line breaks render whether or not a reload
would keep them.

Discriminates: passes on the dev b859124f9 build, fails on that build with 983fa6c97 reverted
(the sent English line shows as "?Hello, how are you" and the Arabic line typed under an English
one sits on the left edge).

Both tests run in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

ARABIC = "مرحبا بالعالم كيف حالك اليوم هل أنت بخير وهل كل شيء على ما يرام"
ENGLISH = "Hello, how are you?"

# where the text holding `needle` is drawn against its box, and the "?" against the "o" of "Hello"
MEASURE = """
(line, needle) => {
    const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node && !node.textContent.includes(needle)) node = walker.nextNode();
    const letter = (index) => {
        const range = document.createRange();
        range.setStart(node, index);
        range.setEnd(node, index + 1);
        return range.getBoundingClientRect();
    };
    const text = node.textContent;
    const whole = document.createRange();
    whole.selectNodeContents(node);
    const box = line.getBoundingClientRect();
    const drawn = whole.getBoundingClientRect();
    const question = text.indexOf('?');
    return {
        questionAfterHello: question < 0 || letter(question).left > letter(text.indexOf('o')).left,
        gapLeft: drawn.left - box.left,
        gapRight: box.right - drawn.right,
    };
}
"""


def line_with(scope: Locator, text: str) -> Locator:
    return scope.locator("p", has_text=text)


def assert_reads_left_to_right(scope: Locator) -> None:
    drawn = line_with(scope, ENGLISH).evaluate(MEASURE, ENGLISH)
    assert drawn["questionAfterHello"], "the question mark is drawn in front of Hello"
    assert drawn["gapLeft"] < drawn["gapRight"], "the English line sits against the right edge"


def assert_sits_on_the_right(scope: Locator) -> None:
    drawn = line_with(scope, ARABIC).evaluate(MEASURE, ARABIC)
    assert drawn["gapRight"] < drawn["gapLeft"], "the Arabic line sits against the left edge"


def type_lines(chat: ChatSurface, first: str, second: str) -> None:
    page = chat.page
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    page.keyboard.type(first)
    page.keyboard.press("Shift+Enter")
    page.keyboard.type(second)


def test_a_sent_english_line_under_an_arabic_one_reads_left_to_right(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    upstream.queue(reply.text("Fine, thank you.", match=reply.answering(ENGLISH)))
    type_lines(chat, ARABIC, ENGLISH)
    chat.page.keyboard.press("Enter")
    expect_reply(chat, "Fine, thank you.")

    assert_reads_left_to_right(conversation(chat).locator(".chat-user"))


def test_an_arabic_line_typed_under_an_english_one_sits_on_the_right(chat_for, make_user):
    chat = chat_for(make_user())
    type_lines(chat, ENGLISH, ARABIC)

    assert_sits_on_the_right(chat_input(chat))
