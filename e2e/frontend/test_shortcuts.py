"""Keyboard shortcuts follow the chords a user saved, and the default ones otherwise.

Ctrl+K opens the search by default (on Windows and Linux, Ctrl stands in for Cmd). A user can
rebind a shortcut in Settings, which saves it under `keybindings` in their settings; the app
loads those on start, so a moved shortcut answers to its new chord and no longer to its old one.

Browser twin of frontend/shortcuts.test.ts, which drives `loadKeybindings` and
`matchKeybinding` directly.

Discriminates: passes on the bbfa876af build; with `loadKeybindings` ignoring what was saved,
Ctrl+Shift+P opens nothing and the rebinding test goes red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

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
