"""Journey: a person tags a chat from the chat actions menu and untags it.

The menu in the chat header ends in Tags, which opens a tag picker (`utils/tag_picker.py`): the
chat's tags ticked, the account's other tags unticked, and a search box where Enter adds the typed
tag. Ticking a tag the account already uses adds it to another chat, and unticking removes it. Each
change is read back after a reload in the same menu; once no chat carries a tag the picker stops
offering it.

The tag picker replaced the tag input and chips in 24b8bb0e6. Before it, picking a suggested tag
closed the menu before the tag was added (open-webui/open-webui#32014, fixed by 62f70a844); the
second test guards against a pick that is lost again.

Discriminates: the tests pass on dev 4acf39ec2; in a frontend copy whose tag input skips its add
request and whose tag chip skips its delete request (dev ebc6add67), every test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from utils.tag_picker import add_tag_by_typing, open_tag_picker, tag_entry

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


def stored_tags(account, chat_id: str) -> list[str]:
    with account.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}/tags")
    assert found.status_code == 200, found.text
    return [tag["name"] for tag in found.json()]


def open_tags_of(page: Page, chat_id: str):
    page.goto(f"/c/{chat_id}")
    return open_tag_picker(page)


def test_a_tag_added_from_the_chat_actions_menu_is_still_on_the_chat_after_a_reload(
    page_for, two_chats
):
    account, chat_id, _other_id = two_chats
    tag = unique_tag()
    page = page_for(account)
    add_tag_by_typing(open_tags_of(page, chat_id), tag)
    assert stored_tags(account, chat_id) == [tag]

    page.reload()
    picker = open_tags_of(page, chat_id)

    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "true")


def test_a_tag_the_account_already_uses_is_suggested_and_added_to_a_second_chat(
    page_for, two_chats
):
    account, tagged_id, second_id = two_chats
    tag = unique_tag()
    tag_chat(account, tagged_id, tag)
    page = page_for(account)
    picker = open_tags_of(page, second_id)
    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "false")

    tag_entry(picker, tag).click()
    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "true")

    assert stored_tags(account, second_id) == [tag], (
        "picking the account's tag never added it to the chat (open-webui/open-webui#32014)"
    )
    picker = open_tags_of(page, second_id)
    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "true")


def test_a_removed_tag_is_gone_from_the_chat_and_the_suggestions(page_for, two_chats):
    account, tagged_id, other_id = two_chats
    tag = unique_tag()
    tag_chat(account, tagged_id, tag)
    page = page_for(account)
    picker = open_tags_of(page, tagged_id)
    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "true")

    tag_entry(picker, tag).click()
    # no chat carries it any more, so the account's tags drop it too
    expect(tag_entry(picker, tag)).to_have_count(0)
    assert stored_tags(account, tagged_id) == []

    page.reload()
    picker = open_tags_of(page, tagged_id)
    expect(picker.get_by_role("menuitemcheckbox")).to_have_count(0)

    picker = open_tags_of(page, other_id)
    picker.get_by_label("Find or create a tag").fill(tag)
    expect(picker.get_by_text(f"Create “{tag}”")).to_be_visible()
    expect(tag_entry(picker, tag)).to_have_count(0)
