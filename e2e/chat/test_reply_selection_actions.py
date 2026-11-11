"""Journey: asking about, or for an explanation of, a passage selected in a reply.

Selecting text in a finished reply shows two buttons beside it. Explain puts the passage, quoted,
into the chat input with the word Explain under it; Ask opens a small question box and puts the
quoted passage with the typed question into the chat input. Nothing is sent until the person
sends it, and what is sent is the quote with the question. Escape hides the buttons again.

Discriminates: passes on the dev 30f3f6a8f build; in a frontend build whose Explain and Ask leave
the chat input empty and whose buttons stay up after Escape, every test fails.

Retargeted for dev b1da0e8e8: the buttons now float at the end of the page (7129cc1d8, #32330), so
they are found by the reply they belong to and no longer inside it.

Every test runs on the saved chat in the main chat and dragged into a tab beside another chat
(`utils/chat_surface.py`), the only places a saved chat opens. A side chat answer offered no Ask
and no Explain on selected text (open-webui/open-webui#34014, fix PR open-webui/open-webui#34064):
on dev c15f28649 the side chat variants are red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT, ChatSurface
from utils.chat_ui import chat_input, expect_reply, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST_PARAGRAPH = "The west pier floods at spring tide."
SECOND_PARAGRAPH = "The east pier stays dry all year."
# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened;
# the model count cannot matter, as the buttons belong to the one reply holding the selection
SAVED_CHAT_PLACES = (MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)


@pytest.fixture
def harbour_reply(chat_for, make_user) -> tuple[ChatSurface, Locator]:
    """The chat open on the reply, and the box its Ask and Explain buttons show in."""
    owner = make_user()
    with owner.client() as client:
        chat_id, reply_id = seed_chat(
            client,
            [
                {"role": "user", "content": "Where can I moor in Whitby?"},
                {"role": "assistant", "content": f"{FIRST_PARAGRAPH}\n\n{SECOND_PARAGRAPH}"},
            ],
        )
    chat = chat_for(owner, chat_id=chat_id)
    expect_reply(chat, SECOND_PARAGRAPH)
    return chat, chat.page.locator(f"#floating-buttons-{chat_id}-{reply_id}")


def _select(chat: ChatSurface, text: str) -> None:
    """Drag across one paragraph of the reply to select it."""
    page = chat.page
    paragraph = last_reply(chat).get_by_text(text, exact=True)
    box = paragraph.bounding_box()
    middle = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + 1, middle)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - 1, middle, steps=5)
    page.mouse.up()


def _send_and_read(chat: ChatSurface, upstream, answer: str) -> str:
    upstream.queue(reply.text(answer, match=reply.answering(SECOND_PARAGRAPH)))
    chat_input(chat).click()
    chat.page.keyboard.press("Enter")
    expect_reply(chat, answer)
    sent = next(filter(reply.answering(SECOND_PARAGRAPH), reversed(upstream.chat_requests())))
    return str(reply.conversation(sent)[-1]["content"])


@pytest.mark.chat_surfaces(*SAVED_CHAT_PLACES)
def test_explain_quotes_the_selected_passage_into_the_message(harbour_reply, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat, buttons = harbour_reply
    _select(chat, SECOND_PARAGRAPH)
    buttons.get_by_role("button", name="Explain").click()

    expect(chat_input(chat)).to_contain_text(SECOND_PARAGRAPH)
    expect(chat_input(chat)).to_contain_text("Explain")
    expect(chat_input(chat)).not_to_contain_text(FIRST_PARAGRAPH)
    assert [
        body for body in upstream.chat_requests() if reply.answering(SECOND_PARAGRAPH)(body)
    ] == []

    sent = _send_and_read(chat, upstream, "It sits above the high water line.")
    assert f"> {SECOND_PARAGRAPH}" in sent
    assert sent.rstrip().endswith("Explain")


@pytest.mark.chat_surfaces(*SAVED_CHAT_PLACES)
def test_ask_sends_the_quoted_passage_with_the_typed_question(harbour_reply, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat, buttons = harbour_reply
    _select(chat, SECOND_PARAGRAPH)
    buttons.get_by_role("button", name="Ask", exact=True).click()
    question = buttons.get_by_role("textbox", name="Ask a question")
    expect(question).to_be_focused()
    question.fill("Even in a storm surge?")
    buttons.get_by_role("button", name="Submit question").click()

    expect(chat_input(chat)).to_contain_text(SECOND_PARAGRAPH)
    expect(chat_input(chat)).to_contain_text("Even in a storm surge?")

    sent = _send_and_read(chat, upstream, "Mostly, yes.")
    assert f"> {SECOND_PARAGRAPH}" in sent
    assert sent.rstrip().endswith("Even in a storm surge?")


@pytest.mark.chat_surfaces(*SAVED_CHAT_PLACES)
def test_escape_hides_the_buttons_again(harbour_reply):
    """In a side chat: open-webui/open-webui#34014."""
    chat, buttons = harbour_reply
    _select(chat, SECOND_PARAGRAPH)
    explain = buttons.get_by_role("button", name="Explain")
    expect(explain).to_be_visible()

    chat.page.keyboard.press("Escape")

    expect(explain).to_be_hidden()
    expect(chat_input(chat)).to_have_text("")
