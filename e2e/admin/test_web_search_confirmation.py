"""Journey: Web Search Confirmation in Admin Settings > Web Search asks before a search is used.

With the switch on and a text of the admin's own, a user who turns Web Search on in the chat
input's Integrations menu is asked first, with that text. Cancel leaves Web Search off, so the
next question reaches the model without the search tool; Continue turns it on, so the model is
offered the search. With the switch off nobody is asked.

Both answering tests used to fail: the dialog opened while the Integrations menu stayed open
behind it, and that menu's outside-click handler took the first click on Cancel or Continue to
close itself, so the dialog needed a second click (open-webui/open-webui#31963), fixed in dev
806644fcb.

Discriminates: passes on dev f6cbeb1a1; both answering tests fail on dev 6defd4a94, before
806644fcb. On a frontend build of dev 30f3f6a8f that closes the Integrations menu as Web Search is
picked and asks even with the switch off, the "unasked" test fails.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), the Cancel and
Continue tests in every way a side chat starts and every state it can be in. A side chat never
asked and ran the search at once (open-webui/open-webui#34021, fix PR
open-webui/open-webui#34064); for those the dev c15f28649 build is the mutant.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import expect_reply, open_integrations_menu, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTICE = "Searches leave the harbour network. Ask the office first."


@pytest.fixture
def search_engine(admin, preserve, listener):
    """Web search on for the instance, on a local engine nobody needs to reach here."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


def open_web_search_settings(page: Page) -> Locator:
    page.goto("/admin/settings/web")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Web Search Confirmation")).to_be_visible()
    return settings


def save_confirmation(admin_page: Page, turn_on: bool) -> None:
    settings = open_web_search_settings(admin_page)
    switch = settings.get_by_role("switch", name="Web Search Confirmation")
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    if turn_on:
        content = settings.get_by_text("Web Search Confirmation Content", exact=True)
        content.locator("xpath=following-sibling::div//textarea").fill(NOTICE)
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()


def turn_web_search_on(chat: ChatSurface) -> None:
    open_integrations_menu(chat).get_by_role("menuitemcheckbox", name="Web Search").click()


def offered_tools(upstream, question: str) -> set[str]:
    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    return {tool["function"]["name"] for tool in sent.get("tools") or []}


def ask(chat: ChatSurface, upstream, question: str) -> set[str]:
    upstream.queue(reply.text("Calm seas today.", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "Calm seas today.")
    return offered_tools(upstream, question)


def confirmation(page: Page) -> Locator:
    return page.get_by_role("dialog", name="Use Web Search?")


def answer_confirmation(page: Page, button: str) -> None:
    expect(confirmation(page)).to_contain_text(NOTICE)
    confirmation(page).get_by_role("button", name=button).click()
    expect(
        confirmation(page),
        f"one click on {button} left the dialog open: the Integrations menu, still open behind "
        "it, takes the click to close itself (#31963)",
    ).to_have_count(0)


# one model only: the question comes as the switch is turned on, before any model answers
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_cancelling_the_confirmation_leaves_web_search_off(
    page_for, chat_for, admin, make_user, upstream, search_engine
):
    """In a side chat: open-webui/open-webui#34021."""
    save_confirmation(page_for(admin), turn_on=True)
    chat = chat_for(make_user())

    turn_web_search_on(chat)
    answer_confirmation(chat.page, "Cancel")

    assert "search_web" not in ask(chat, upstream, "Is the sea calm at the pier?")


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_continuing_the_confirmation_turns_web_search_on(
    page_for, chat_for, admin, make_user, upstream, search_engine
):
    """In a side chat: open-webui/open-webui#34021."""
    save_confirmation(page_for(admin), turn_on=True)
    chat = chat_for(make_user())

    turn_web_search_on(chat)
    answer_confirmation(chat.page, "Continue")

    assert "search_web" in ask(chat, upstream, "Is the sea calm at the breakwater?")


def test_with_the_confirmation_off_web_search_turns_on_unasked(
    page_for, chat_for, admin, make_user, upstream, search_engine
):
    save_confirmation(page_for(admin), turn_on=False)
    chat = chat_for(make_user())

    turn_web_search_on(chat)
    chat.page.keyboard.press("Escape")

    assert "search_web" in ask(chat, upstream, "Is the sea calm at the lighthouse?")
    expect(confirmation(chat.page)).to_have_count(0)
