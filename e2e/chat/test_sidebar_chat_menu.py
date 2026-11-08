"""Journey: the sidebar chat menu renames, pins, archives, deletes, moves, clones and marks a chat.

Each action is taken from the menu on the chat's sidebar row, and each is read back after a
reload, so what shows is what the server stored: the new title, the chat under Pinned, the chat
among Archived Chats in the settings, the chat gone, the chat inside its folder, a copy holding the
conversation beside the original, and the unread dot until the chat is opened again. A rename to
only spaces is refused with an error and the stored title stays.

Discriminates: passes on dev 30f3f6a8f; in a backend copy each test fails when its route answers
without storing the change: the chat update dropping `title`, the pin and archive toggles not
flipping, the delete route not deleting, the folder route not moving the chat, the clone route
storing an empty conversation and the unread route leaving the read time alone. In the pre-fix
build of 0.12.4 (9cbd27011 reverted) the rename to spaces shows no error and saves the blank title.
Retargeted for dc45e3960, whose rows add how long ago the chat was updated to their name: passes on
dev a915df8ce, 5 of 5.

Leaving a rename with Escape gives the focus back to the chat's menu button (Tim's
75f8ae558). Passes on dev e8c0ad85e and fails on that build without the focus return.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.sidebar_chats import chat_menu, chat_row
from utils.sidebar_folders import folder_row
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE = "Trip to the coast"
QUESTION = {"id": "q1", "parentId": None, "childrenIds": ["a1"], "role": "user"}
ANSWER = {"id": "a1", "parentId": "q1", "childrenIds": [], "role": "assistant", "done": True}
HISTORY = {
    "currentId": "a1",
    "messages": {
        "q1": {**QUESTION, "content": "Where should we stay?"},
        "a1": {**ANSWER, "content": "A cottage above the harbour."},
    },
}


@pytest.fixture
def owner(make_user):
    """A fresh account with one chat and one expanded folder."""
    account = make_user()
    with account.client() as client:
        created = client.post(
            "/api/v1/chats/new", json={"chat": {"title": TITLE, "history": HISTORY}}
        )
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


@pytest.fixture
def sidebar(page_for, owner) -> Locator:
    page = page_for(owner)
    side = _open_sidebar(page)
    expect(chat_row(side, TITLE)).to_be_visible()
    return side


def _reloaded(sidebar: Locator) -> Locator:
    sidebar.page.reload()  # the sidebar stays open
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def test_a_renamed_chat_keeps_its_new_title(sidebar):
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Rename").click()
    sidebar.page.keyboard.press("Control+A")
    sidebar.page.keyboard.type("Trip to the mountains")
    sidebar.page.keyboard.press("Enter")
    expect(chat_row(sidebar, "Trip to the mountains")).to_be_visible()

    _reloaded(sidebar)
    expect(chat_row(sidebar, "Trip to the mountains")).to_be_visible()
    expect(chat_row(sidebar, TITLE)).to_have_count(0)


def test_a_chat_renamed_to_only_spaces_keeps_its_title(sidebar, owner):
    """#32770 (fix 9cbd27011): a title of only spaces was saved and the chat showed blank."""
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Rename").click()
    sidebar.page.keyboard.press("Control+A")
    sidebar.page.keyboard.type("   ")
    sidebar.page.keyboard.press("Enter")

    expect(sidebar.page.get_by_text("Title cannot be an empty string.")).to_be_visible()
    expect(chat_row(sidebar, TITLE)).to_be_visible()
    with owner.client() as client:
        listed = client.get("/api/v1/chats/")
    assert listed.status_code == 200, listed.text
    assert [chat["title"] for chat in listed.json()] == [TITLE]


def test_leaving_a_rename_with_escape_gives_the_focus_back_to_the_chat_menu(sidebar):
    # the title field went away with the focus in it, so the focus fell to the page
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Rename").click()
    title_field = sidebar.get_by_role("textbox")
    expect(title_field).to_be_focused()

    sidebar.page.keyboard.press("Escape")

    expect(title_field).to_have_count(0)
    row = sidebar.locator("#sidebar-chat-group").filter(has=chat_row(sidebar.page, TITLE))
    expect(
        row.get_by_role("button", name="Chat Menu").first,
        "after leaving the rename the focus is not on the chat's menu button",
    ).to_be_focused()
    expect(chat_row(sidebar, TITLE)).to_be_visible()


def test_a_pinned_chat_stays_pinned(sidebar):
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Pin").click()
    expect(sidebar.get_by_role("button", name="Pinned")).to_be_visible()

    _reloaded(sidebar)
    expect(sidebar.get_by_role("button", name="Pinned")).to_be_visible()
    menu = chat_menu(sidebar, TITLE)
    expect(menu.get_by_role("menuitem", name="Unpin")).to_be_visible()


def test_an_archived_chat_leaves_the_sidebar_for_the_archive(sidebar):
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Archive").click()
    expect(chat_row(sidebar, TITLE)).to_have_count(0)

    page = _reloaded(sidebar).page
    expect(chat_row(sidebar, TITLE)).to_have_count(0)
    open_settings(page)
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Archived Chats").click()
    expect(settings.get_by_text(TITLE)).to_be_visible()


def test_a_deleted_chat_stays_gone(sidebar):
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Delete").click()
    confirm = sidebar.page.get_by_role("dialog", name="Delete chat?")
    confirm.get_by_role("button", name="Confirm").click()
    expect(chat_row(sidebar, TITLE)).to_have_count(0)

    _reloaded(sidebar)
    expect(chat_row(sidebar, TITLE)).to_have_count(0)


def test_a_chat_moved_into_a_folder_stays_there(sidebar):
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    folders.click()
    folder = folder_row(sidebar, "Holidays")
    expect(folder).to_be_visible()
    expect(sidebar.get_by_text("No chats")).to_be_visible()

    menu = chat_menu(sidebar, TITLE)
    menu.get_by_role("menuitem", name="Move").hover()
    sidebar.page.get_by_role("menu").get_by_role("menuitem", name="Holidays").click()
    expect(sidebar.get_by_text("No chats")).to_have_count(0)

    _reloaded(sidebar)
    expect(folder).to_be_visible()
    expect(sidebar.get_by_text("No chats")).to_have_count(0)
    expect(chat_row(sidebar, TITLE)).to_have_count(1)


def test_a_clone_holds_the_conversation_and_leaves_the_original(sidebar):
    page = sidebar.page
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Clone").click()
    clone_title = f"Clone of {TITLE}"
    expect(chat_row(sidebar, clone_title)).to_be_visible()
    expect(page.get_by_text("A cottage above the harbour.")).to_be_visible()

    _reloaded(sidebar)
    expect(chat_row(sidebar, clone_title)).to_be_visible()
    expect(page.get_by_text("A cottage above the harbour.")).to_be_visible()
    chat_row(sidebar, TITLE).first.click()
    expect(page.get_by_text("A cottage above the harbour.")).to_be_visible()


def _unread_dot(sidebar: Locator, title: str) -> Locator:
    # the dot has no text or role of its own
    row = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    return row.locator(".bg-sky-500.rounded-full")


def test_a_chat_marked_unread_keeps_its_dot_until_it_is_opened(sidebar):
    expect(_unread_dot(sidebar, TITLE)).to_have_count(0)
    chat_menu(sidebar, TITLE).get_by_role("menuitem", name="Mark as unread").click()
    expect(_unread_dot(sidebar, TITLE)).to_be_visible()

    _reloaded(sidebar)
    expect(_unread_dot(sidebar, TITLE)).to_be_visible()
    chat_row(sidebar, TITLE).first.click()
    expect(sidebar.page.get_by_text("A cottage above the harbour.")).to_be_visible()
    sidebar.page.goto("/")
    expect(chat_row(sidebar, TITLE).first).to_be_visible()
    expect(_unread_dot(sidebar, TITLE)).to_have_count(0)
