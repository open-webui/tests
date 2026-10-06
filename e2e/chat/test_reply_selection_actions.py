"""Journey: asking about, or for an explanation of, a passage selected in a reply.

Selecting text in a finished reply shows two buttons beside it. Explain puts the passage, quoted,
into the chat input with the word Explain under it; Ask opens a small question box and puts the
quoted passage with the typed question into the chat input. Nothing is sent until the person
sends it, and what is sent is the quote with the question. Escape hides the buttons again.

Discriminates: passes on the dev 30f3f6a8f build; in a frontend build whose Explain and Ask leave
the chat input empty and whose buttons stay up after Escape, every test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, expect_reply, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST_PARAGRAPH = "The west pier floods at spring tide."
SECOND_PARAGRAPH = "The east pier stays dry all year."


@pytest.fixture
def harbour_reply(page_for, make_user) -> Page:
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Where can I moor in Whitby?"},
                {"role": "assistant", "content": f"{FIRST_PARAGRAPH}\n\n{SECOND_PARAGRAPH}"},
            ],
        )
    page = page_for(owner)
    page.goto(f"/c/{chat_id}")
    expect_reply(page, SECOND_PARAGRAPH)
    return page


def _select(page: Page, text: str) -> Locator:
    """Drag across one paragraph of the reply to select it; returns the reply."""
    paragraph = last_reply(page).get_by_text(text, exact=True)
    box = paragraph.bounding_box()
    middle = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + 1, middle)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - 1, middle, steps=5)
    page.mouse.up()
    return last_reply(page)


def _send_and_read(page: Page, upstream, answer: str) -> str:
    upstream.queue(reply.text(answer, match=reply.answering(SECOND_PARAGRAPH)))
    chat_input(page).click()
    page.keyboard.press("Enter")
    expect_reply(page, answer)
    sent = next(filter(reply.answering(SECOND_PARAGRAPH), reversed(upstream.chat_requests())))
    return str(sent["messages"][-1]["content"])


def test_explain_quotes_the_selected_passage_into_the_message(harbour_reply, upstream):
    _select(harbour_reply, SECOND_PARAGRAPH).get_by_role("button", name="Explain").click()

    expect(chat_input(harbour_reply)).to_contain_text(SECOND_PARAGRAPH)
    expect(chat_input(harbour_reply)).to_contain_text("Explain")
    expect(chat_input(harbour_reply)).not_to_contain_text(FIRST_PARAGRAPH)
    assert [
        body for body in upstream.chat_requests() if reply.answering(SECOND_PARAGRAPH)(body)
    ] == []

    sent = _send_and_read(harbour_reply, upstream, "It sits above the high water line.")
    assert f"> {SECOND_PARAGRAPH}" in sent
    assert sent.rstrip().endswith("Explain")


def test_ask_sends_the_quoted_passage_with_the_typed_question(harbour_reply, upstream):
    selected = _select(harbour_reply, SECOND_PARAGRAPH)
    selected.get_by_role("button", name="Ask", exact=True).click()
    question = selected.get_by_role("textbox", name="Ask a question")
    expect(question).to_be_focused()
    question.fill("Even in a storm surge?")
    selected.get_by_role("button", name="Submit question").click()

    expect(chat_input(harbour_reply)).to_contain_text(SECOND_PARAGRAPH)
    expect(chat_input(harbour_reply)).to_contain_text("Even in a storm surge?")

    sent = _send_and_read(harbour_reply, upstream, "Mostly, yes.")
    assert f"> {SECOND_PARAGRAPH}" in sent
    assert sent.rstrip().endswith("Even in a storm surge?")


def test_escape_hides_the_buttons_again(harbour_reply):
    explain = _select(harbour_reply, SECOND_PARAGRAPH).get_by_role("button", name="Explain")
    expect(explain).to_be_visible()

    harbour_reply.keyboard.press("Escape")

    expect(explain).to_be_hidden()
    expect(chat_input(harbour_reply)).to_have_text("")
