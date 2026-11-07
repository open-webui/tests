"""Keyboard shortcuts follow the chords a user saved, and the default ones otherwise.

Ctrl+K opens the search by default (on Windows and Linux, Ctrl stands in for Cmd). A user can
rebind a shortcut in Settings, which saves it under `keybindings` in their settings; the app
loads those on start, so a moved shortcut answers to its new chord and no longer to its old one.
In Settings > Keyboard a chord is recorded by clicking the shortcut's keys and pressing the new
ones, Reset Defaults brings every default chord back, and Enable Keyboard Shortcuts switched off
leaves every chord inert, after a reload too. The Tabs section lists the workspace shortcuts
(80ddf8e3b) with their default chords, and a rebound one (Toggle Split) answers to its new chord
and no longer to Ctrl+Backslash.

Browser twin of frontend/shortcuts.test.ts, which drives `loadKeybindings` and
`matchKeybinding` directly.

Discriminates: passes on the bbfa876af build; with `loadKeybindings` ignoring what was saved,
Ctrl+Shift+P opens nothing and the rebinding test goes red. In a frontend build of dev 30f3f6a8f
whose recorder and Reset Defaults save nothing and whose app ignores the switch, the recorded,
reset and switched off tests fail. In a frontend build of f659c0d12 listing Toggle Split under
Global and with its handler removed, the Tabs list and rebound tests fail.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
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


WORKSPACE_SHORTCUTS = {
    "Quick Open": "Ctrl+P",
    "Voice Note": "Ctrl+Alt+M",
    "Files": "Ctrl+Shift+E",
    "Desktop": "Ctrl+Alt+D",
    "New Chat Tab": "Ctrl+Alt+O",
    "Comments": "Ctrl+Shift+Y",
    "New File": "Ctrl+Alt+N",
    "New Terminal": "Ctrl+`",
    "New Browser": "Ctrl+Shift+B",
    "Close Tab": "Ctrl+W",
    "Next Tab": "Ctrl+Shift+]",
    "Previous Tab": "Ctrl+Shift+[",
    "Toggle Split": "Ctrl+\\",
}


def test_the_keyboard_settings_list_the_workspace_shortcuts_under_tabs(page_for, make_user):
    page = page_for(make_user(), path="/?settings=shortcuts")
    tab = page.locator("#tab-shortcuts")
    shortcuts = tab.get_by_role("list", name="Tabs", exact=True).get_by_role("listitem")

    expect(shortcuts).to_have_count(len(WORKSPACE_SHORTCUTS))
    listed = {}
    for item in shortcuts.all():
        name = item.inner_text().split("\n")[0].strip()
        listed[name] = item.get_by_title("Click to rebind").inner_text().strip()
    assert listed == WORKSPACE_SHORTCUTS, f"the Tabs section lists {listed}"


def test_a_rebound_workspace_shortcut_answers_to_its_new_chord(page_for, make_user, admin):
    account = make_user()
    # a side chat to open gives the workspace something to split off
    make_group(admin, [account], {"chat": {"side_chat": True}})
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"keybindings": {"toggleSplit": "Cmd+Shift+P"}},
        )
    saved.raise_for_status()
    page = open_app(page_for, account)
    panes = page.get_by_role("tablist", name="Tabs")
    expect(panes).to_have_count(1)

    # the split toggles, so a Ctrl+Backslash that still split would be undone by Ctrl+Shift+P
    page.keyboard.press("Control+Backslash")
    page.keyboard.press("Control+Shift+P")
    expect(panes).to_have_count(2)
    expect(page.get_by_role("separator", name="Resize panels")).to_have_attribute(
        "aria-valuenow", "50"
    )
