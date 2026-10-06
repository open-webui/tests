"""Journey: a person tags a chat from the chat actions menu, finds it by tag and untags it.

The menu in the chat header ends in a tag input ("Add a tag...") above the chat's tag chips: Enter
adds the typed tag, a tag the account already uses is offered as a suggestion that adds it to
another chat, and a click on a chip removes it. Each change is read back after a reload in the same
menu and through the search dialog's `tag:` filter, which lists the chats carrying a tag; once no
chat carries a tag the input stops suggesting it.

The suggestion test is red on dev: the suggestion list sits outside the menu, so the menu's
outside-click handler closes the menu on the pick and the tag is never added.

Discriminates: the other three pass on dev ebc6add67; in a frontend copy whose tag input skips
its add request and whose tag chip skips its delete request, every test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def unique_tag() -> str:
    return f"harbour{uuid.uuid4().hex[:8]}"


def stored_chat(client, title: str) -> str:
    question = {"id": "q1", "parentId": None, "childrenIds": ["a1"], "role": "user"}
    answer = {"id": "a1", "parentId": "q1", "childrenIds": [], "role": "assistant", "done": True}
    history = {
        "currentId": "a1",
        "messages": {
            "q1": {**question, "content": "Where should we stay?"},
            "a1": {**answer, "content": "A cottage above the harbour."},
        },
    }
    created = client.post("/api/v1/chats/new", json={"chat": {"title": title, "history": history}})
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture
def two_chats(make_user) -> tuple:
    """An account with two untagged chats: (account, id of the first, id of the second)."""
    account = make_user()
    with account.client() as client:
        first_id = stored_chat(client, f"Coast {uuid.uuid4().hex[:8]}")
        second_id = stored_chat(client, f"Hills {uuid.uuid4().hex[:8]}")
    return account, first_id, second_id


def tag_chat(account, chat_id: str, tag: str) -> None:
    with account.client() as client:
        added = client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": tag})
    assert added.status_code == 200, added.text


def chat_title(account, chat_id: str) -> str:
    with account.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}").json()["title"]


def open_actions_menu(page: Page, chat_id: str) -> Locator:
    page.goto(f"/c/{chat_id}")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Chat actions").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_placeholder("Add a tag...")).to_be_visible()
    return menu


def tag_chip(menu: Locator, tag: str) -> Locator:
    return menu.get_by_role("button", name=tag, exact=True)


def add_tag_by_typing(menu: Locator, tag: str) -> None:
    menu.get_by_placeholder("Add a tag...").fill(tag)
    menu.page.keyboard.press("Enter")
    expect(tag_chip(menu, tag)).to_be_visible()


def suggestion(page: Page, tag: str) -> Locator:
    return page.get_by_role("option", name=tag, exact=True)


def search_tag(page: Page, tag: str) -> Locator:
    """Opens the search dialog on a fresh page load and types the `tag:` filter."""
    page.reload()
    expect(chat_input(page)).to_be_visible()
    page.keyboard.press("Control+K")
    dialog = page.get_by_role("dialog")
    search = dialog.get_by_placeholder("Search")
    expect(search).to_be_visible()
    search.fill(f"tag:{tag}")
    return dialog


def listed(dialog: Locator, title: str) -> Locator:
    return dialog.get_by_role("link").filter(has_text=title)


def test_a_tag_added_from_the_chat_actions_menu_is_still_on_the_chat_after_a_reload(
    page_for, two_chats
):
    account, chat_id, _other_id = two_chats
    tag = unique_tag()
    page = page_for(account)
    add_tag_by_typing(open_actions_menu(page, chat_id), tag)

    page.reload()
    menu = open_actions_menu(page, chat_id)

    expect(tag_chip(menu, tag)).to_be_visible()


def test_searching_by_tag_lists_the_tagged_chat_and_not_the_other(page_for, two_chats):
    account, tagged_id, other_id = two_chats
    tag = unique_tag()
    page = page_for(account)
    add_tag_by_typing(open_actions_menu(page, tagged_id), tag)

    dialog = search_tag(page, tag)

    expect(listed(dialog, chat_title(account, tagged_id))).to_be_visible()
    expect(dialog.get_by_role("link")).to_have_count(1)
    expect(listed(dialog, chat_title(account, other_id))).to_have_count(0)


def test_a_tag_the_account_already_uses_is_suggested_and_added_to_a_second_chat(
    page_for, two_chats
):
    account, tagged_id, second_id = two_chats
    tag = unique_tag()
    tag_chat(account, tagged_id, tag)
    page = page_for(account)
    menu = open_actions_menu(page, second_id)
    expect(tag_chip(menu, tag)).to_have_count(0)

    menu.get_by_placeholder("Add a tag...").click()
    suggestion(page, tag).click()

    menu = open_actions_menu(page, second_id)
    expect(
        tag_chip(menu, tag),
        "picking the suggestion closed the menu before the tag was added",
    ).to_be_visible()

    dialog = search_tag(page, tag)
    expect(listed(dialog, chat_title(account, second_id))).to_be_visible()
    expect(listed(dialog, chat_title(account, tagged_id))).to_be_visible()
    expect(dialog.get_by_role("link")).to_have_count(2)


def test_a_removed_tag_is_gone_from_the_chat_the_search_and_the_suggestions(page_for, two_chats):
    account, tagged_id, other_id = two_chats
    tag = unique_tag()
    tag_chat(account, tagged_id, tag)
    page = page_for(account)
    menu = open_actions_menu(page, tagged_id)
    tag_chip(menu, tag).click()
    expect(tag_chip(menu, tag)).to_have_count(0)

    page.reload()
    menu = open_actions_menu(page, tagged_id)
    expect(tag_chip(menu, tag)).to_have_count(0)

    dialog = search_tag(page, tag)
    expect(dialog.get_by_text("No results found")).to_be_visible()
    expect(dialog.get_by_role("link")).to_have_count(0)
    page.keyboard.press("Escape")

    menu = open_actions_menu(page, other_id)
    menu.get_by_placeholder("Add a tag...").click()
    expect(page.get_by_role("listbox")).to_have_count(0)
    expect(suggestion(page, tag)).to_have_count(0)
