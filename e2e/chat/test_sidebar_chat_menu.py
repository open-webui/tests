"""Journey: the sidebar's chat menu renames, pins, archives, deletes and moves a chat for good.

Each action is taken from the menu on the chat's sidebar row, and each is read back after a
reload, so what shows is what the server stored: the new title, the chat under Pinned, the chat
among Archived Chats in the settings, the chat gone, the chat inside its folder.

Discriminates: passes on dev ac00d40e3; in a backend copy each test fails when its route answers
without storing the change: the chat update dropping `title`, the pin and archive toggles not
flipping, the delete route not deleting and the folder route not moving the chat.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE = "Trip to the coast"


@pytest.fixture
def owner(make_user):
    """A fresh account with one chat and one expanded folder."""
    account = make_user()
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": TITLE}})
        assert created.status_code == 200, created.text
        folder = client.post("/api/v1/folders/", json={"name": "Holidays"})
        assert folder.status_code == 200, folder.text
        expanded = client.post(
            f"/api/v1/folders/{folder.json()['id']}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text
    return account


def _open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def _chat_row(sidebar: Locator, title: str) -> Locator:
    return sidebar.get_by_role("button", name=title)


def _chat_menu(sidebar: Locator, title: str) -> Locator:
    """Open the menu on the row of `title`, which shows its trigger while the row has focus."""
    row = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    row.hover()
    _chat_row(row, title).focus()
    # the row drops its age label ("1m") from its name once the menu trigger is shown
    expect(row.get_by_role("button", name=title, exact=True)).to_be_visible()
    row.get_by_role("button", name="Chat Menu").first.click()
    return sidebar.page.get_by_role("menu")


@pytest.fixture
def sidebar(page_for, owner) -> Locator:
    page = page_for(owner)
    side = _open_sidebar(page)
    expect(_chat_row(side, TITLE)).to_be_visible()
    return side


def _reloaded(sidebar: Locator) -> Locator:
    sidebar.page.reload()  # the sidebar stays open
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def test_a_renamed_chat_keeps_its_new_title(sidebar):
    _chat_menu(sidebar, TITLE).get_by_role("button", name="Rename").click()
    sidebar.page.keyboard.press("Control+A")
    sidebar.page.keyboard.type("Trip to the mountains")
    sidebar.page.keyboard.press("Enter")
    expect(_chat_row(sidebar, "Trip to the mountains")).to_be_visible()

    _reloaded(sidebar)
    expect(_chat_row(sidebar, "Trip to the mountains")).to_be_visible()
    expect(_chat_row(sidebar, TITLE)).to_have_count(0)


def test_a_pinned_chat_stays_pinned(sidebar):
    _chat_menu(sidebar, TITLE).get_by_role("button", name="Pin").click()
    expect(sidebar.get_by_role("button", name="Pinned")).to_be_visible()

    _reloaded(sidebar)
    expect(sidebar.get_by_role("button", name="Pinned")).to_be_visible()
    menu = _chat_menu(sidebar, TITLE)
    expect(menu.get_by_role("button", name="Unpin")).to_be_visible()


def test_an_archived_chat_leaves_the_sidebar_for_the_archive(sidebar):
    _chat_menu(sidebar, TITLE).get_by_role("button", name="Archive").click()
    expect(_chat_row(sidebar, TITLE)).to_have_count(0)

    page = _reloaded(sidebar).page
    expect(_chat_row(sidebar, TITLE)).to_have_count(0)
    page.get_by_role("button", name="User menu").first.click()
    page.get_by_role("menu").get_by_role("button", name="Settings").click()
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Archived Chats").click()
    expect(settings.get_by_text(TITLE)).to_be_visible()


def test_a_deleted_chat_stays_gone(sidebar):
    _chat_menu(sidebar, TITLE).get_by_role("button", name="Delete").click()
    confirm = sidebar.page.get_by_role("dialog", name="Delete chat?")
    confirm.get_by_role("button", name="Confirm").click()
    expect(_chat_row(sidebar, TITLE)).to_have_count(0)

    _reloaded(sidebar)
    expect(_chat_row(sidebar, TITLE)).to_have_count(0)


def test_a_chat_moved_into_a_folder_stays_there(sidebar):
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    folders.click()
    folder = sidebar.get_by_role("button", name="Holidays")
    expect(folder).to_be_visible()
    expect(sidebar.get_by_text("No chats")).to_be_visible()

    menu = _chat_menu(sidebar, TITLE)
    menu.get_by_role("button", name="Move").hover()
    sidebar.page.get_by_role("menu").get_by_role("button", name="Holidays").click()
    expect(sidebar.get_by_text("No chats")).to_have_count(0)

    _reloaded(sidebar)
    expect(folder).to_be_visible()
    expect(sidebar.get_by_text("No chats")).to_have_count(0)
    expect(_chat_row(sidebar, TITLE)).to_have_count(1)
