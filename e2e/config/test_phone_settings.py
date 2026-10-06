"""Journey: the Settings dialog on a phone opens from the drawer, fits the screen and saves.

On a 390 by 844 touch screen, Settings opens from the user menu at the foot of the sidebar drawer
and fills the screen without scrolling sideways, with Back, the search field and the tab strip on
it. The tab strip scrolls sideways, so every tab can be brought onto the screen and opens its own
panel. A switch flipped in Interface on the phone is saved and changes the chat, and Back closes
the dialog onto the chat again.

Discriminates: passes on the ebc6add67 build. In frontend builds of ebc6add67: with the tab strip
not scrolling, the every-tab test goes red at the first tab past the screen's edge; with Back doing
nothing, the switch and Back tests go red; in a frontend build whose phone layout is 480 pixels wide
(wider than the screen, so controls on the right fall off it), all four go red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, conversation, expect_reply
from utils.phone import (
    PHONE,
    SCREEN,
    expect_on_screen,
    expect_reachable,
    send_by_tapping,
    tap_on_screen,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TABS = [
    "General",
    "Interface",
    "Notifications",
    "Keyboard",
    "Personalization",
    "Audio",
    "Data Controls",
    "Usage",
    "Archived Chats",
    "Account",
    "About",
]


@pytest.fixture
def phone(page_for, make_user) -> Page:
    page = page_for(make_user(), **PHONE)
    expect_on_screen(chat_input(page))
    return page


def open_settings(page: Page) -> Locator:
    tap_on_screen(page.get_by_role("button", name="Open Sidebar").last)
    tap_on_screen(page.get_by_role("navigation", name="Chat history").get_by_label("User menu"))
    tap_on_screen(page.get_by_role("menu").get_by_role("button", name="Settings"))
    settings = page.get_by_role("dialog")
    expect_on_screen(settings.get_by_role("tab", name="General"))
    return settings


def expect_no_sideways_scroll(page: Page) -> None:
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= SCREEN["width"], f"the page scrolls sideways ({scroll_width}px wide)"


def expect_within_the_width(panel: Locator) -> None:
    expect(panel).to_be_visible()
    box = panel.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= SCREEN["width"] + 0.5, box


def test_settings_open_from_the_drawer_and_fit_the_screen(phone):
    settings = open_settings(phone)

    expect_within_the_width(settings)
    expect_on_screen(settings.get_by_role("button", name="Back", exact=True))
    expect_on_screen(settings.get_by_placeholder("Search"))
    expect_on_screen(settings.get_by_role("button", name="Save", exact=True))
    expect_no_sideways_scroll(phone)


def test_every_tab_can_be_scrolled_onto_the_screen_and_opens_its_panel(phone):
    settings = open_settings(phone)
    expect(settings.get_by_role("tab")).to_have_text(TABS)

    for name in TABS:
        tab = settings.get_by_role("tab", name=name, exact=True)
        expect_reachable(tab)
        tab.tap()
        expect(tab).to_have_attribute("aria-selected", "true")
        panel = settings.locator("nav + div")
        expect_within_the_width(panel)
        expect_on_screen(panel.get_by_text(re.compile(rf"^{name}")).first)
        expect_no_sideways_scroll(phone)


def test_a_switch_flipped_on_a_phone_changes_the_chat(phone, upstream):
    settings = open_settings(phone)
    interface = settings.get_by_role("tab", name="Interface", exact=True)
    expect_reachable(interface)
    interface.tap()
    switch = phone.locator("#tab-interface").get_by_role("switch", name="Chat Bubble UI")
    expect_reachable(switch)
    with phone.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.tap()
    expect(switch).to_have_attribute("aria-checked", "false")
    tap_on_screen(settings.get_by_role("button", name="Back", exact=True))
    expect(settings).to_be_hidden()

    upstream.queue(reply.text("Noted on the phone.", match=reply.answering("bubbles off")))
    send_by_tapping(phone, "bubbles off")
    expect_reply(phone, "Noted on the phone.")
    you = conversation(phone).locator(".user-message").get_by_text("You", exact=True)
    expect_on_screen(you)


def test_back_closes_settings_onto_the_chat(phone):
    settings = open_settings(phone)
    tap_on_screen(settings.get_by_role("tab", name="General", exact=True))

    tap_on_screen(settings.get_by_role("button", name="Back", exact=True))

    expect(settings).to_be_hidden()
    expect(phone).not_to_have_url(re.compile(r"settings="))
    expect_on_screen(chat_input(phone))
