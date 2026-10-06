"""Journey: forking a chat from a reply, deleting a question, steering a regenerate, saving an edit.

The buttons under a message in a two-turn chat: Fork chat on the first reply opens a new chat that
holds only the first turn and answers from it, the question's Delete (after its confirm dialog)
takes the question and its reply out of the stored chat, the Regenerate menu sends a typed change
or More Concise to the model after the earlier reply, and Save on an edited question keeps the new
text without asking the model again. Each result is read after a reload or in what the model was
sent, as a fresh account against the scripted model.

The two regenerate tests are red on dev: a saved chat sends the model only the stored history up
to the question, so a suggested change or More Concise reaches the model without the reply it is
about (with the regenerated reply appended from the database both pass).

Discriminates: passes on dev 30f3f6a8f apart from the two regenerate tests; in a backend copy the
fork test fails with the fork route copying the whole conversation past the chosen reply and the
delete test with the message delete route storing nothing; in a frontend build whose Save edits
nothing the save test fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, last_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def two_turns(page_for, make_user, upstream) -> Page:
    """A fresh account's chat with two questions answered, open in the browser."""
    page = page_for(make_user())
    upstream.queue(
        reply.text("Lisbon is the capital.", match=reply.answering("capital of Portugal")),
        reply.text("About ten million people.", match=reply.answering("how many people")),
    )
    send(page, "What is the capital of Portugal?")
    expect_reply(page, "Lisbon is the capital.")
    send(page, "And how many people live there?")
    expect_reply(page, "About ten million people.")
    expect(page).to_have_url(re.compile(r"/c/"))
    return page


def _questions(page: Page) -> Locator:
    return conversation(page).locator(".chat-user")


def _shown(message: Locator) -> Locator:
    """The message hovered, so its hover-only buttons are shown."""
    message.hover()
    return message


def _whole_reply(reply_text: Locator) -> Locator:
    """The reply with its buttons, which sit beside the reply's text."""
    return reply_text.locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")


def _latest_request_messages(upstream) -> list[tuple[str, str]]:
    return [
        (entry["role"], str(entry["content"])) for entry in upstream.chat_requests()[-1]["messages"]
    ]


def test_a_fork_from_the_first_reply_holds_only_the_first_turn(two_turns, upstream):
    page = two_turns
    original_url = page.url
    first_reply = _shown(_whole_reply(replies(page).first))
    first_reply.get_by_role("button", name="Fork chat").click()

    expect(page).not_to_have_url(original_url)
    expect_reply(page, "Lisbon is the capital.")
    expect(replies(page)).to_have_count(1)
    expect(conversation(page).get_by_text("About ten million people.")).to_have_count(0)

    upstream.queue(reply.text("Porto is the second city.", match=reply.answering("second city")))
    send(page, "What is the second city?")
    expect_reply(page, "Porto is the second city.")
    sent = _latest_request_messages(upstream)
    assert ("assistant", "Lisbon is the capital.") in sent
    assert not any("ten million" in content for _, content in sent), sent

    page.goto(original_url)
    expect_reply(page, "About ten million people.")
    expect(replies(page)).to_have_count(2)


def test_a_deleted_question_and_its_reply_stay_gone(two_turns, upstream):
    page = two_turns
    second_question = _shown(_questions(page).last)
    second_question.get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Delete message?").get_by_role("button", name="Confirm").click()

    expect(conversation(page).get_by_text("About ten million people.")).to_have_count(0)
    page.reload()
    expect_reply(page, "Lisbon is the capital.")
    expect(_questions(page)).to_have_count(1)
    expect(conversation(page).get_by_text("how many people live there")).to_have_count(0)

    upstream.queue(reply.text("It sits on the Tagus.", match=reply.answering("which river")))
    send(page, "On which river?")
    expect_reply(page, "It sits on the Tagus.")
    sent = _latest_request_messages(upstream)
    assert not any("how many people" in content for _, content in sent), sent


DROPPED_REPLY = (
    "the model was not sent the reply it is asked to change: a saved chat's regenerate reloads the "
    "history from the database only up to the question, so the suggestion follows the question"
)


def _regenerate_menu(page: Page) -> Locator:
    regenerate = _shown(_whole_reply(last_reply(page))).get_by_role("button", name="Regenerate")
    regenerate.last.click()  # the menu trigger wraps the button of the same name
    return page.get_by_role("menu")


def test_a_suggested_change_is_sent_after_the_reply_it_changes(two_turns, upstream):
    page = two_turns
    upstream.queue(
        reply.text("Roughly 545,000 in the city.", match=reply.answering("just the city"))
    )
    menu = _regenerate_menu(page)
    menu.get_by_placeholder("Suggest a change").fill("just the city, please")
    menu.get_by_role("button", name="Submit suggestion").click()

    expect_reply(page, "Roughly 545,000 in the city.")
    expect(conversation(page).get_by_text("2/2")).to_be_visible()
    assert _latest_request_messages(upstream)[-2:] == [
        ("assistant", "About ten million people."),
        ("user", "just the city, please"),
    ], DROPPED_REPLY


def test_more_concise_asks_the_model_for_a_shorter_reply(two_turns, upstream):
    page = two_turns
    upstream.queue(reply.text("Ten million.", match=reply.answering("More Concise")))
    _regenerate_menu(page).get_by_role("button", name="More Concise").click()

    expect_reply(page, "Ten million.")
    assert _latest_request_messages(upstream)[-2:] == [
        ("assistant", "About ten million people."),
        ("user", "More Concise"),
    ], DROPPED_REPLY


def test_a_saved_question_edit_is_kept_without_a_new_reply(two_turns, upstream):
    page = two_turns
    requests_before = len(upstream.chat_requests())
    second_question = _shown(_questions(page).last)
    second_question.get_by_role("button", name="Edit").click()
    second_question.locator("textarea").fill("And how many people live in Lisbon?")
    second_question.get_by_role("button", name="Save", exact=True).click()

    expect(conversation(page).get_by_text("And how many people live in Lisbon?")).to_be_visible()
    page.reload()
    expect_reply(page, "About ten million people.")
    expect(conversation(page).get_by_text("And how many people live in Lisbon?")).to_be_visible()
    expect(conversation(page).get_by_text("2/2")).to_have_count(0)
    assert len(upstream.chat_requests()) == requests_before
