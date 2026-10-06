"""Keyboard shortcuts follow the chords a user saved, and the default ones otherwise.

Ctrl+K opens the search by default (on Windows and Linux, Ctrl stands in for Cmd). A user can
rebind a shortcut in Settings, which saves it under `keybindings` in their settings; the app
loads those on start, so a moved shortcut answers to its new chord and no longer to its old one.
In Settings > Keyboard a chord is recorded by clicking the shortcut's keys and pressing the new
ones, Reset Defaults brings every default chord back, and Enable Keyboard Shortcuts switched off
leaves every chord inert, after a reload too.

Browser twin of frontend/shortcuts.test.ts, which drives `loadKeybindings` and
`matchKeybinding` directly.

Discriminates: passes on the bbfa876af build; with `loadKeybindings` ignoring what was saved,
Ctrl+Shift+P opens nothing and the rebinding test goes red. In a frontend build of dev 30f3f6a8f
whose recorder and Reset Defaults save nothing and whose app ignores the switch, the recorded,
reset and switched off tests fail.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.requires_browser, pytest.mark.requires_source]


def search(page: Page):
    return page.get_by_role("dialog").get_by_placeholder("Search")


def open_app(page_for, account) -> Page:
    """A signed-in page whose shortcuts are live: they are bound before the chat input renders."""
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    return page


def test_ctrl_k_opens_the_search_by_default(page_for, make_user):
    page = open_app(page_for, make_user())

    page.keyboard.press("Control+K")
    expect(search(page)).to_be_visible()


def test_a_saved_chord_moves_the_search_off_ctrl_k(page_for, make_user):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"keybindings": {"search": "Cmd+Shift+P"}}
        )
    saved.raise_for_status()
    page = open_app(page_for, account)

    page.keyboard.press("Control+Shift+P")
    expect(search(page)).to_be_visible()
    page.keyboard.press("Escape")
    expect(search(page)).to_be_hidden()

    # the search toggles, so a Ctrl+K that still opened it would be closed again by Ctrl+Shift+P
    page.keyboard.press("Control+K")
    page.keyboard.press("Control+Shift+P")
    expect(search(page)).to_be_visible()


def keyboard_settings(page: Page) -> Locator:
    page.goto("/?settings=shortcuts")
    tab = page.locator("#tab-shortcuts")
    expect(tab.get_by_role("switch", name="Enable Keyboard Shortcuts")).to_be_visible()
    return tab


def chord_button(tab: Locator, shortcut: str) -> Locator:
    name = tab.get_by_text(shortcut, exact=True)
    row = name.locator("xpath=ancestor::div[.//button[@title='Click to rebind']][1]")
    return row.get_by_title("Click to rebind")


def close_settings(page: Page) -> None:
    page.keyboard.press("Escape")
    expect(page.locator("#tab-shortcuts")).to_be_hidden()


def test_a_chord_recorded_in_settings_moves_the_search_off_ctrl_k(page_for, make_user):
    page = open_app(page_for, make_user())
    tab = keyboard_settings(page)

    chord_button(tab, "Search").click()
    expect(tab.get_by_text("Press keys...")).to_be_visible()
    page.keyboard.press("Control+Shift+Y")
    expect(tab.get_by_text("Press keys...")).to_have_count(0)
    page.goto("/")
    expect(chat_input(page)).to_be_visible()

    page.keyboard.press("Control+Shift+Y")
    expect(search(page)).to_be_visible()
    page.keyboard.press("Escape")
    expect(search(page)).to_be_hidden()
    page.keyboard.press("Control+K")
    page.keyboard.press("Control+Shift+Y")
    expect(search(page)).to_be_visible()


def test_reset_defaults_brings_ctrl_k_back(page_for, make_user):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"keybindings": {"search": "Cmd+Shift+P"}}
        )
    saved.raise_for_status()
    page = open_app(page_for, account)
    tab = keyboard_settings(page)

    tab.get_by_role("button", name="Reset Defaults").click()
    expect(chord_button(tab, "Search")).not_to_contain_text("P")
    page.goto("/")
    expect(chat_input(page)).to_be_visible()

    page.keyboard.press("Control+K")
    expect(search(page)).to_be_visible()


def test_with_shortcuts_switched_off_ctrl_k_opens_nothing_on_the_next_load(page_for, make_user):
    page = open_app(page_for, make_user())
    tab = keyboard_settings(page)
    switch = tab.get_by_role("switch", name="Enable Keyboard Shortcuts")

    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")
    page.goto("/")
    expect(chat_input(page)).to_be_visible()

    page.keyboard.press("Control+K")
    page.wait_for_timeout(1000)  # the search opens at once when the chord is live
    expect(search(page)).to_have_count(0)
