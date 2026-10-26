"""A click outside an open menu also went through to what lay underneath, open-webui#29878.

Fix commit `8fc416ee7` (issue open-webui/open-webui#29878). The model selector and the shared
dropdown closed on `pointerdown` outside them and let the click go on, so a click meant only to
close the menu also landed on whatever was under it: a suggested prompt was sent as a message.
Both now close on the click itself, in the capture phase, and swallow it.

Discriminates: passes on the efe63bd34 build, fails on it with `8fc416ee7` reverted (the
suggestion clicked to close the menu is sent to the model).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input, conversation

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# a point on the element that no open popup covers, found by scanning across its middle
UNCOVERED_POINT = """(element) => {
    const box = element.getBoundingClientRect();
    const y = box.top + box.height / 2;
    for (let x = box.left + 4; x < box.right - 4; x += 8) {
        if (element.contains(document.elementFromPoint(x, y))) {
            return {x: x - box.left, y: y - box.top};
        }
    }
    return null;
}"""


def first_suggestion(page: Page) -> Locator:
    # the suggestion is the button; its listitem wrapper lays out as `contents` with no box
    return page.get_by_role("main").get_by_role("listitem").first.get_by_role("button")


def click_uncovered(target: Locator) -> None:
    # the popup grows in with a transition; measure it once it has settled. A transition the
    # page cancels (a tooltip leaving) rejects its promise, which counts as settled too
    target.page.evaluate("Promise.allSettled(document.getAnimations().map((a) => a.finished))")
    point = target.evaluate(UNCOVERED_POINT)
    assert point is not None, "the open menu covers the whole suggestion"
    target.click(position=point)


def expect_nothing_sent(page: Page, upstream) -> None:
    # bounded: the stray click would have sent the suggestion by now
    page.wait_for_timeout(1500)
    expect(conversation(page)).to_have_count(0)
    assert upstream.chat_requests() == [], "the click that closed the menu sent a suggestion"


@pytest.fixture
def page(page_for, make_user) -> Page:
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    expect(first_suggestion(page)).to_be_visible()
    return page


def test_a_click_that_closes_the_model_selector_leaves_the_suggestion_alone(page, upstream):
    page.get_by_role("button", name="Selected model: mock-model").click()
    models = page.get_by_role("listbox", name="Available models")
    expect(models).to_be_visible()

    click_uncovered(first_suggestion(page))

    expect(models).to_be_hidden()
    expect_nothing_sent(page, upstream)


def test_a_click_that_closes_the_attachment_menu_leaves_the_suggestion_alone(page, upstream):
    page.get_by_role("button", name="More", exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()

    click_uncovered(first_suggestion(page))

    expect(menu).to_be_hidden()
    expect_nothing_sent(page, upstream)


def test_the_next_click_on_a_suggestion_still_sends_it(page, upstream):
    page.get_by_role("button", name="Selected model: mock-model").click()
    click_uncovered(first_suggestion(page))
    expect(page.get_by_role("listbox", name="Available models")).to_be_hidden()

    first_suggestion(page).click()

    # a suggestion sends its full prompt, which differs from the words on its card
    expect(conversation(page).locator(".chat-user")).to_have_count(1)
    expect(conversation(page).locator(".chat-assistant")).to_have_count(1)
