"""Journey: pages open as tabs beside the chat, and a new chat keeps them.

Since 0.13.4 a page picked from the sidebar (Notes) or from the user menu (Workspace) opens in a tab
of its own beside the chat that is open, which stays in its tab; picking the page again selects
its tab instead of adding one. A note linked in a reply opens the same way. Starting a new chat,
from New Chat or by opening a folder in the sidebar, swaps the chat in its tab and leaves the page
tabs beside it.

Discriminates: passes on dev 485103cde. In a frontend copy of it whose sidebar, user menu and note
links navigate in place of opening a tab, all three fail (the chat's tab is gone, no Notes tab
opens); in one whose new chat clears the open tabs, the third fails while the note test passes.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.workspace_search import create_note
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, send
from utils.sidebar_folders import folder_link
from utils.user_menu import user_menu_trigger

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT = [
    {"role": "user", "content": "Which ferry crosses to the island?"},
    {"role": "assistant", "content": "The morning ferry leaves from the east pier."},
]


def _tabs(page: Page) -> Locator:
    return page.get_by_role("tablist", name="Tabs").first


def _tab(page: Page, name: str) -> Locator:
    return _tabs(page).get_by_role("tab", name=name, exact=True)


def _saved_chat(account, title: str) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(client, CHAT)
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert renamed.status_code == 200, renamed.text
    return chat_id


def _open_chat(page_for, account) -> tuple[Page, str, str]:
    title = f"Island ferry {uuid.uuid4().hex[:8]}"
    chat_id = _saved_chat(account, title)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, CHAT[1]["content"])
    return page, chat_id, title


def _expect_selected(page: Page, name: str, path: str) -> None:
    """The tab `name` is selected and the address shows its page (the workspace opens a section)."""
    expect(_tab(page, name)).to_have_attribute("aria-selected", "true")
    expect(page).to_have_url(re.compile(rf"{re.escape(path)}(?:/[a-z]+)?$"))


def _back_to_the_chat(page: Page, chat_id: str, title: str) -> None:
    chat_tab = _tab(page, title)
    expect(chat_tab, "the chat's tab is gone once a page opened").to_be_visible()
    chat_tab.click()
    _expect_selected(page, title, f"/c/{chat_id}")
    expect(page.get_by_role("log").get_by_text(CHAT[1]["content"])).to_be_visible()


def test_a_page_from_the_sidebar_or_the_user_menu_opens_in_a_tab_beside_the_chat(
    page_for, make_user
):
    # an admin's user menu offers the workspace whatever the default permissions say
    account = make_user(role="admin")
    page, chat_id, title = _open_chat(page_for, account)

    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    _back_to_the_chat(page, chat_id, title)

    user_menu_trigger(page).click()
    page.get_by_role("menu").get_by_role("menuitem", name="Workspace").click()
    _expect_selected(page, "Workspace", "/workspace")
    _back_to_the_chat(page, chat_id, title)

    # a page already in a tab is selected again, not opened twice
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    # the chat, Notes and the workspace
    expect(_tabs(page).get_by_role("tab")).to_have_count(3)


def test_a_note_linked_in_a_reply_opens_in_a_tab_beside_the_chat(page_for, make_user, upstream):
    account = make_user()
    note_title = f"Packing list {uuid.uuid4().hex[:8]}"
    with account.client() as client:
        note_id = create_note(client, note_title, "Rain jacket, ferry tickets.")
    page, chat_id, title = _open_chat(page_for, account)
    question = f"Where is my packing list? ({uuid.uuid4().hex[:8]})"
    upstream.queue(
        reply.text(
            f"It is in your note [packing list](/notes/{note_id}).",
            match=reply.answering(question),
        )
    )
    send(page, question)
    expect_reply(page, "It is in your note")

    page.get_by_role("log").get_by_role("button", name=re.compile(re.escape(note_title))).click()

    _expect_selected(page, note_title, f"/notes/{note_id}")
    _back_to_the_chat(page, chat_id, title)


def test_a_new_chat_keeps_the_page_tabs_open_beside_the_chat(page_for, make_user):
    account = make_user()
    folder_name = f"Islands {uuid.uuid4().hex[:8]}"
    with account.client() as client:
        created = client.post("/api/v1/folders/", json={"name": folder_name})
    assert created.status_code == 200, created.text
    folder_id = created.json()["id"]
    page, _, title = _open_chat(page_for, account)
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    _tab(page, title).click()

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).to_have_url(re.compile(r"/$"))
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(_tab(page, title)).to_have_count(0)
    expect(_tab(page, "Notes"), "a new chat closed the Notes tab").to_be_visible()

    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    folder_link(sidebar, folder_name).click()
    expect(page).to_have_url(re.compile(rf"/folders/{folder_id}$"))
    expect(_tab(page, folder_name)).to_have_attribute("aria-selected", "true")
    expect(_tab(page, "Notes"), "a new chat in a folder closed the Notes tab").to_be_visible()
