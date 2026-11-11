"""Journey: forking a chat from a reply, deleting a question, steering a regenerate, saving an edit.

The buttons under a message in a two-turn chat: Fork chat on the first reply opens a new chat that
holds only the first turn and answers from it, the question's Delete (after its confirm dialog)
takes the question and its reply out of the stored chat, the Regenerate menu sends a typed change
or More Concise to the model after the question, and Save on an edited question keeps the new
text without asking the model again. A picture removed while editing a question is not sent with
it again, and a document removed while editing one and saved is gone from it after a reload. Each
result is read after a reload or in what the model was sent, as a fresh account against the
scripted model.

Discriminates: passes on dev 30f3f6a8f; in a backend copy the
fork test fails with the fork route copying the whole conversation past the chosen reply and the
delete test with the message delete route storing nothing; in a frontend build whose Save edits
nothing the save test fails, in one whose Remove file while editing keeps the file the picture
test fails, and in one whose Remove File on a document chip does nothing the document test fails.

Every test runs in the main chat and in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`); the ones that reload leave out a temporary side chat, which is gone
after a reload. A fork opens where it was asked, so in a side chat the fork is looked for in that
side chat's tab. A side chat message had no Fork chat, Delete, Edit or Regenerate
(open-webui/open-webui#34014, fix PR open-webui/open-webui#34064): on dev c15f28649 every side
chat variant is red. On dev e791cbdde, which holds that PR, all pass but the fork test: in a side
chat the fork is made and never opened.
"""

from __future__ import annotations

import base64
import re

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, ChatSurface, main_chat
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


# the model count cannot matter: each action acts on one message
# a temporary chat is gone after a reload
EVERY_LASTING_PLACE = tuple(place for place in EVERYWHERE if place != TEMPORARY_SIDE)


@pytest.fixture
def two_turns(chat_for, make_user, upstream) -> ChatSurface:
    """A fresh account's chat with two questions answered, open in the browser."""
    chat = chat_for(make_user())
    upstream.queue(
        reply.text("Lisbon is the capital.", match=reply.answering("capital of Portugal")),
        reply.text("About ten million people.", match=reply.answering("how many people")),
    )
    send(chat, "What is the capital of Portugal?")
    expect_reply(chat, "Lisbon is the capital.")
    send(chat, "And how many people live there?")
    expect_reply(chat, "About ten million people.")
    expect(chat.page).to_have_url(re.compile(r"/c/"))
    return chat


def _questions(chat: ChatSurface) -> Locator:
    return conversation(chat).locator(".chat-user")


def _questions_of_this_chat(chat: ChatSurface) -> Locator:
    """The questions from this test's first one on; a side chat can hold earlier turns."""
    first_question = _questions(chat).filter(has_text="What is the capital of Portugal?")
    return first_question.locator("xpath=self::* | following::*[contains(@class, 'chat-user')]")


def _turns_of_this_chat(chat: ChatSurface) -> Locator:
    """The replies from this test's first question on; a side chat can hold earlier turns."""
    first_question = _questions(chat).filter(has_text="What is the capital of Portugal?")
    return first_question.locator("xpath=following::*[contains(@class, 'chat-assistant')]")


def _shown(message: Locator) -> Locator:
    """The message hovered, so its hover-only buttons are shown."""
    message.hover()
    return message


def _whole_reply(reply_text: Locator) -> Locator:
    """The reply with its buttons, which sit beside the reply's text."""
    return reply_text.locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")


def _latest_request_messages(upstream) -> list[tuple[str, str]]:
    return [
        (entry["role"], str(entry["content"]))
        for entry in reply.conversation(upstream.chat_requests()[-1])
    ]


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_fork_from_the_first_reply_holds_only_the_first_turn(two_turns, upstream):
    """In a side chat: open-webui/open-webui#34014; on dev e791cbdde the fork is made but never
    opened."""
    chat = two_turns
    page = chat.page
    original_url = page.url
    first_reply = _shown(_whole_reply(_turns_of_this_chat(chat).first))
    first_reply.get_by_role("button", name="Fork chat").click()

    expect(page, "the fork did not open").not_to_have_url(original_url)
    expect_reply(chat, "Lisbon is the capital.")
    expect(_turns_of_this_chat(chat)).to_have_count(1)
    expect(conversation(chat).get_by_text("About ten million people.")).to_have_count(0)

    upstream.queue(reply.text("Porto is the second city.", match=reply.answering("second city")))
    send(chat, "What is the second city?")
    expect_reply(chat, "Porto is the second city.")
    sent = _latest_request_messages(upstream)
    assert ("assistant", "Lisbon is the capital.") in sent
    assert not any("ten million" in content for _, content in sent), sent

    page.goto(original_url)
    original = main_chat(page)
    expect_reply(original, "About ten million people.")
    expect(_turns_of_this_chat(original)).to_have_count(2)


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_deleted_question_and_its_reply_stay_gone(two_turns, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat = two_turns
    second_question = _shown(_questions(chat).last)
    second_question.get_by_role("button", name="Delete").click()
    chat.page.get_by_role("dialog", name="Delete message?").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(conversation(chat).get_by_text("About ten million people.")).to_have_count(0)
    chat = chat.reload()
    expect_reply(chat, "Lisbon is the capital.")
    expect(_questions_of_this_chat(chat)).to_have_count(1)
    expect(conversation(chat).get_by_text("how many people live there")).to_have_count(0)

    upstream.queue(reply.text("It sits on the Tagus.", match=reply.answering("which river")))
    send(chat, "On which river?")
    expect_reply(chat, "It sits on the Tagus.")
    sent = _latest_request_messages(upstream)
    assert not any("how many people" in content for _, content in sent), sent


def _regenerate_menu(chat: ChatSurface) -> Locator:
    regenerate = _shown(_whole_reply(last_reply(chat))).get_by_role("button", name="Regenerate")
    regenerate.last.click()  # the menu trigger wraps the button of the same name
    return chat.page.get_by_role("menu")


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_suggested_change_is_sent_after_the_question(two_turns, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat = two_turns
    upstream.queue(
        reply.text("Roughly 545,000 in the city.", match=reply.answering("just the city"))
    )
    menu = _regenerate_menu(chat)
    menu.get_by_placeholder("Suggest a change").fill("just the city, please")
    menu.get_by_role("menuitem", name="Submit suggestion").click()

    expect_reply(chat, "Roughly 545,000 in the city.")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()
    assert _latest_request_messages(upstream)[-1] == ("user", "just the city, please")


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_more_concise_asks_the_model_for_a_shorter_reply(two_turns, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat = two_turns
    upstream.queue(reply.text("Ten million.", match=reply.answering("More Concise")))
    _regenerate_menu(chat).get_by_role("menuitem", name="More Concise").click()

    expect_reply(chat, "Ten million.")
    assert _latest_request_messages(upstream)[-1] == ("user", "More Concise")


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_saved_question_edit_is_kept_without_a_new_reply(two_turns, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    chat = two_turns
    requests_before = len(upstream.chat_requests())
    second_question = _shown(_questions(chat).last)
    second_question.get_by_role("button", name="Edit").click()
    second_question.locator("textarea").fill("And how many people live in Lisbon?")
    second_question.get_by_role("button", name="Save", exact=True).click()

    expect(conversation(chat).get_by_text("And how many people live in Lisbon?")).to_be_visible()
    chat = chat.reload()
    expect_reply(chat, "About ten million people.")
    expect(conversation(chat).get_by_text("And how many people live in Lisbon?")).to_be_visible()
    expect(conversation(chat).get_by_text("2/2")).to_have_count(0)
    assert len(upstream.chat_requests()) == requests_before


# a 2x2 red PNG
RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5E"
    "rkJggg=="
)


def _images_sent(upstream, question: str) -> int:
    request = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    content = reply.conversation(request)[-1]["content"]
    parts = content if isinstance(content, list) else []
    return len([part for part in parts if part.get("type") == "image_url"])


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_picture_removed_while_editing_a_question_is_not_sent_again(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34014."""
    chat = chat_for(make_user())
    page = chat.page
    expect(chat_input(chat)).to_be_visible()
    chat.button("More").last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": "buoy.png", "mimeType": "image/png", "buffer": RED_PNG})
    expect(chat.button("Show image preview")).to_be_visible()
    upstream.queue(
        reply.text("A red buoy.", match=reply.answering("what colour is the buoy")),
        reply.text("I cannot see one now.", match=reply.answering("what colour is the buoy")),
    )
    send(chat, "what colour is the buoy?")
    expect_reply(chat, "A red buoy.")
    assert _images_sent(upstream, "what colour is the buoy") == 1

    question = _shown(_questions(chat).last)
    question.get_by_role("button", name="Edit").click()
    remove = question.get_by_role("button", name="Remove file")
    remove.focus()
    remove.click()
    expect(remove).to_have_count(0)
    question.get_by_role("button", name="Send").click()

    expect_reply(chat, "I cannot see one now.")
    assert _images_sent(upstream, "what colour is the buoy") == 0


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_document_removed_while_editing_a_question_stays_off_it_after_a_save(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34014."""
    chat = chat_for(make_user())
    page = chat.page
    expect(chat_input(chat)).to_be_visible()
    chat.button("More").last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files(
        {"name": "packing-list.txt", "mimeType": "text/plain", "buffer": b"rope, lamp, oilskin"}
    )
    expect(chat.button("Remove File")).to_be_visible()
    upstream.queue(reply.text("Rope and a lamp.", match=reply.answering("what do I pack")))
    send(chat, "what do I pack?")
    expect_reply(chat, "Rope and a lamp.")
    expect(_questions(chat).last.get_by_text("packing-list.txt")).to_be_visible()

    question = _shown(_questions(chat).last)
    question.get_by_role("button", name="Edit").click()
    question.get_by_role("button", name="Remove File", exact=True).click()
    expect(question.get_by_text("packing-list.txt")).to_have_count(0)
    question.get_by_role("button", name="Save", exact=True).click()

    chat = chat.reload()
    expect_reply(chat, "Rope and a lamp.")
    expect(_questions(chat).last).to_contain_text("what do I pack?")
    expect(_questions(chat).last.get_by_text("packing-list.txt")).to_have_count(0)
