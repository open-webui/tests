"""Journey: comparing a draft with a revision in the chat input's Compare text view.

More > Compare text opens the side-by-side compare and merge view with the message being typed on
the left as the original. Use in prompt puts the revision on the right into the message box and
sends nothing until the person sends it. In the comparison the view counts the changes ("Change 1
of 2") and steps through them with Previous change and Next change, copies the current change from
the original to the proposed revision or back, and says "No differences" for two identical texts.
Closing after an edit asks to discard the comparison first and leaves the message as it was;
closing with nothing edited asks nothing. Opening the view from the `/compare` command is
e2e/chat/test_slash_commands.py.

Discriminates: passes on the dev 3b8205505 build (backend at 3b8205505, frontend of 226dca18c,
which differs only in the chat input's selected-feature pills). In a build of it whose Use in prompt
also sends the message the Use in prompt test fails, in one whose copy buttons copy the other way
the copy test fails, in one whose Next change stays put the step-through test fails, in one that
counts identical texts as one change the no-differences test fails, and in one whose Cancel closes
without asking the discard test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, open_input_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ORIGINAL = "alpha\nbravo\ncharlie\ndelta\necho\nfoxtrot\ngolf"
REVISED = "alpha\nBRAVO\ncharlie\ndelta\necho\nFOXTROT\ngolf"


def _type_draft(page: Page, text: str) -> None:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(text)
    expect(chat_input(page)).to_have_text(text)


def _open_compare(page: Page) -> Locator:
    open_input_menu(page).get_by_role("menuitem", name="Compare text").click()
    compare = page.get_by_role("dialog", name="Compare text")
    expect(compare).to_be_visible()
    return compare


def _pane(compare: Locator, label: str) -> Locator:
    return compare.get_by_role("textbox", name=label, exact=True)


def _write(compare: Locator, label: str, text: str) -> None:
    _pane(compare, label).click()
    compare.page.keyboard.insert_text(text)


def _start_comparing(compare: Locator, original: str, revised: str) -> Locator:
    """Write both sides, press Compare and return the change counter."""
    _write(compare, "Original", original)
    _write(compare, "Proposed revision", revised)
    compare.get_by_role("button", name="Compare", exact=True).click()
    return compare.get_by_role("status")


def test_use_in_prompt_puts_the_revision_in_the_input_without_sending(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    draft = f"Tell me about the tides {uuid.uuid4().hex[:6]}"
    revised = f"{draft}, in one short sentence"
    _type_draft(page, draft)
    sent_before = len(upstream.chat_requests())

    compare = _open_compare(page)
    expect(_pane(compare, "Original")).to_have_text(draft)
    _pane(compare, "Proposed revision").click()
    page.keyboard.insert_text(revised)
    compare.get_by_role("button", name="Use in prompt").click()

    expect(compare).to_have_count(0)
    expect(chat_input(page)).to_have_text(revised)
    # a bounded look that nothing is sent on its own after the revision lands
    page.wait_for_timeout(1000)
    assert len(upstream.chat_requests()) == sent_before, "Use in prompt sent the message itself"
    expect(page).not_to_have_url(re.compile(r"/c/"))

    upstream.queue(reply.text("Noted.", match=reply.answering(revised)))
    chat_input(page).press("Enter")
    expect_reply(page, "Noted.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(revised)(body)]
    assert request["messages"][-1]["content"] == revised


def test_next_and_previous_change_step_through_the_changes(page_for, make_user):
    page = page_for(make_user())
    compare = _open_compare(page)
    counter = _start_comparing(compare, ORIGINAL, REVISED)
    previous = compare.get_by_role("button", name="Previous change")
    following = compare.get_by_role("button", name="Next change")

    expect(counter).to_have_text("Change 1 of 2")
    expect(previous).to_be_disabled()
    following.click()
    expect(counter).to_have_text("Change 2 of 2")
    expect(following).to_be_disabled()
    previous.click()
    expect(counter).to_have_text("Change 1 of 2")


def test_a_change_is_copied_to_the_proposal_or_back_to_the_original(page_for, make_user):
    page = page_for(make_user())
    compare = _open_compare(page)
    counter = _start_comparing(compare, ORIGINAL, REVISED)
    expect(counter).to_have_text("Change 1 of 2")

    compare.get_by_role("button", name="Copy change to proposal").click()
    expect(counter).to_have_text("Change 1 of 1")
    expect(_pane(compare, "Proposed revision")).to_contain_text("bravo")
    expect(_pane(compare, "Proposed revision")).not_to_contain_text("BRAVO")

    compare.get_by_role("button", name="Copy change to original").click()
    expect(counter).to_have_text("No differences")
    expect(_pane(compare, "Original")).to_contain_text("FOXTROT")
    expect(_pane(compare, "Original")).not_to_contain_text("foxtrot")

    compare.get_by_role("button", name="Use in prompt").click()
    expect(compare).to_have_count(0)
    expect(chat_input(page)).to_contain_text("bravo")
    expect(chat_input(page)).to_contain_text("FOXTROT")


def test_identical_texts_show_no_differences(page_for, make_user):
    page = page_for(make_user())
    same = "The harbour closes at dusk."
    _type_draft(page, same)
    compare = _open_compare(page)
    _write(compare, "Proposed revision", same)
    compare.get_by_role("button", name="Compare", exact=True).click()

    expect(compare.get_by_role("status")).to_have_text("No differences")
    expect(compare.get_by_role("button", name="Previous change")).to_be_disabled()
    expect(compare.get_by_role("button", name="Next change")).to_be_disabled()
    expect(compare.get_by_role("button", name="Copy change to proposal")).to_be_disabled()


def test_closing_after_an_edit_asks_to_discard_and_keeps_the_message(page_for, make_user):
    page = page_for(make_user())
    draft = f"Keep this draft {uuid.uuid4().hex[:6]}"
    _type_draft(page, draft)

    compare = _open_compare(page)
    compare.get_by_role("button", name="Cancel", exact=True).click()
    expect(compare).to_have_count(0)

    compare = _open_compare(page)
    _write(compare, "Proposed revision", "Something else entirely")
    compare.get_by_role("button", name="Cancel", exact=True).click()
    confirm = page.get_by_role("dialog", name="Discard comparison changes?")
    expect(confirm).to_contain_text("Your prompt will stay unchanged.")
    confirm.get_by_role("button", name="Cancel").click()
    expect(confirm).to_have_count(0)
    expect(_pane(compare, "Proposed revision")).to_have_text("Something else entirely")

    compare.get_by_role("button", name="Cancel", exact=True).click()
    confirm.get_by_role("button", name="Discard").click()
    expect(compare).to_have_count(0)
    expect(chat_input(page)).to_have_text(draft)
