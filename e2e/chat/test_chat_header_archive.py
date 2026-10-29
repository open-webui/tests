"""Regression: the chat header menu offered Archive for an archived chat and said it was archived.

Issue open-webui/open-webui#31473, fix `ce94c41d8` (PR open-webui/open-webui#31857). The archive
button always read Archive, and the toast always said "Chat archived.", but the route toggles:
choosing it on an open archived chat unarchived it while telling the user it was archived. The
menu now reads Unarchive for an archived chat and the toast follows what the route did. A chat
that is not archived keeps Archive and "Chat archived.".

Discriminates: passes on the dev b859124f9 build, fails on that build with ce94c41d8 reverted
(the menu of an archived chat offers Archive and the toast says "Chat archived." for a chat that
was just unarchived).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.chat_history import seed_chat

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    return make_user()


def _seed_chat(account, *, archived: bool) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "where to go?"},
                {"role": "assistant", "content": "the coast"},
            ],
        )
        if archived:
            toggled = client.post(f"/api/v1/chats/{chat_id}/archive")
            assert toggled.status_code == 200, toggled.text
    return chat_id


def _is_archived(account, chat_id: str) -> bool:
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["archived"]


def _open_header_menu(page: Page, chat_id: str):
    page.goto(f"/c/{chat_id}")
    page.get_by_label("Chat actions").click()
    return page.get_by_role("menu")


def test_an_archived_chat_is_offered_unarchive_and_is_unarchived(owner, page_for):
    chat_id = _seed_chat(owner, archived=True)
    page = page_for(owner)
    menu = _open_header_menu(page, chat_id)
    expect(menu.get_by_role("menuitem", name="Unarchive")).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Archive", exact=True)).to_have_count(0)

    menu.get_by_role("menuitem", name="Unarchive").click()

    expect(page.get_by_text("Chat unarchived.")).to_be_visible()
    expect(page.get_by_text("Chat archived.")).to_have_count(0)
    assert not _is_archived(owner, chat_id)


def test_a_normal_chat_keeps_archive_and_is_archived(owner, page_for):
    chat_id = _seed_chat(owner, archived=False)
    page = page_for(owner)
    menu = _open_header_menu(page, chat_id)
    expect(menu.get_by_role("menuitem", name="Unarchive")).to_have_count(0)

    menu.get_by_role("menuitem", name="Archive", exact=True).click()

    expect(page.get_by_text("Chat archived.")).to_be_visible()
    assert _is_archived(owner, chat_id)
