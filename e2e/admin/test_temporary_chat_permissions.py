"""The default permissions decide whether accounts can use Temporary Chat or must.

Admin Panel > Users > Groups > Default permissions holds Allow Temporary Chat and, once that is
on, Enforce Temporary Chat. With the first off an account's header has no Temporary Chat button
and its chats are saved. With the second on the button is gone as well and every new chat is
temporary, so nothing it sends is stored. An admin keeps the button whatever the defaults say.
Enforce Temporary Chat switched on and then hidden by switching Allow Temporary Chat off has no
effect on the main chat: its chats are saved (the side chat is covered for open-webui#33928 in
test_default_chat_permissions.py).

Discriminates: passes on dev 176d31d1d, fails with `is_saved_chat_id` treating a `temporary:` id
as saved (an enforced temporary chat is stored). The hidden Enforce test passes on dev b6534c58f
and fails in a build of it whose new chat applies Enforce without Allow Temporary Chat.

A side chat is a new chat as well, so under Enforce Temporary Chat nothing it sends may be stored
either; that test opens it beside the new chat (`utils/chat_surface.py`). A side chat has no
Temporary Chat label of its own (the label belongs to a new main chat's empty page), so it checks
what is stored. The other tests pin the header's button and the address of the page's own chat.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_surface import SIDE_CHAT
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Which river runs through Villach?"
ANSWER = "The Drau runs through Villach."


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def stored_chats(account) -> list[dict]:
    with account.client() as client:
        found = client.get("/api/v1/chats/search", params={"text": "Villach"})
    found.raise_for_status()
    return found.json()


def toggle(page: Page):
    return page.get_by_role("button", name="Temporary Chat")


def test_switching_allow_off_removes_the_button_and_chats_are_saved(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Temporary Chat", turn_on=False)
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()
    expect(toggle(page)).to_have_count(0)

    send(page, PROMPT)

    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    assert len(stored_chats(account)) == 1


@pytest.mark.regression
def test_enforce_left_on_under_a_switched_off_allow_does_not_make_chats_temporary(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    admin_page = page_for(admin)
    save_default_permission(admin_page, "Enforce Temporary Chat", turn_on=True)
    save_default_permission(admin_page, "Allow Temporary Chat", turn_on=False)
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()
    expect(toggle(page)).to_have_count(0)

    send(page, PROMPT)

    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    assert len(stored_chats(account)) == 1, "the chat was not saved, it ran as a temporary chat"


def test_enforcing_temporary_chat_makes_every_new_chat_temporary(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Enforce Temporary Chat", turn_on=True)
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()
    expect(page.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)).to_be_visible()
    expect(toggle(page)).to_have_count(0)

    send(page, PROMPT)

    expect_reply(page, ANSWER)
    expect(page.locator('a[href^="/c/"]')).to_have_count(0)
    assert stored_chats(account) == []


def test_an_admin_keeps_the_temporary_chat_button_when_it_is_enforced(page_for, admin, preserve):
    preserve("permissions")
    page = page_for(admin)
    save_default_permission(page, "Enforce Temporary Chat", turn_on=True)

    page.goto("/")

    expect(toggle(page)).to_be_visible()


@pytest.mark.chat_surfaces(SIDE_CHAT)
def test_enforcing_temporary_chat_keeps_a_side_chat_out_of_storage(
    page_for, chat_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Enforce Temporary Chat", turn_on=True)
    account = make_user()
    side = chat_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))

    send(side, PROMPT)

    expect_reply(side, ANSWER)
    expect(side.page.locator('a[href^="/c/"]')).to_have_count(0)
    assert stored_chats(account) == [], "the side chat was stored under Enforce Temporary Chat"
