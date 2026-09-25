"""Journey: sharing a chat by link, opening it as someone else, and taking the link back.

The owner opens the chat's Share dialog, creates the link and grants one account read access.
That account opens `/s/{id}` in its own browser and reads the conversation; an account left off
the access list is sent back home. Once the owner deletes the link, the granted account is sent
home too.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `DELETE /api/v1/chats/{id}/share`
answering true without removing the share the deletion test fails (the old link still opens),
and with `can_read_shared_chat` granting any signed-in account the stranger test fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "which birds winter by the lake?"
ANSWER = "herons and a few grebes"


def _share_dialog(page: Page) -> Locator:
    page.get_by_role("button", name="Chat actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Share").click()
    return page.get_by_role("dialog").filter(has_text="Share Chat")


@pytest.fixture
def shared_chat(page_for, make_user, upstream):
    """A chat the owner has shared by link: the owner's page, its Share dialog and the link."""
    owner = make_user()
    page = page_for(owner)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)

    dialog = _share_dialog(page)
    dialog.get_by_role("button", name="Copy Link").click()
    link = dialog.get_by_role("link", name="You have shared this chat before")
    expect(link).to_be_visible()
    return page, dialog, link.get_attribute("href")


def _grant(dialog: Locator, account) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
    # the picker lists one page of users; searching reaches any account
    picker.get_by_placeholder("Search").fill(account.name)
    picker.get_by_role("button", name=account.name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(dialog.page.get_by_text("Access updated")).to_be_visible()


def _sent_home(page: Page) -> None:
    expect(page).to_have_url(re.compile(r"/$"))
    expect(page.get_by_text(ANSWER)).to_have_count(0)


def test_a_granted_account_reads_the_link_until_the_owner_deletes_it(
    shared_chat, page_for, make_user
):
    owner_page, dialog, share_path = shared_chat
    viewer = make_user()
    _grant(dialog, viewer)

    viewer_page = page_for(viewer)
    viewer_page.goto(share_path)
    expect(conversation(viewer_page).get_by_text(QUESTION)).to_be_visible()
    expect(conversation(viewer_page).get_by_text(ANSWER)).to_be_visible()

    dialog.get_by_role("button", name="delete this link").click()
    expect(dialog.get_by_role("button", name="Copy Link")).to_be_visible()

    viewer_page.goto(share_path)
    _sent_home(viewer_page)


def test_an_account_left_off_the_access_list_is_sent_home(shared_chat, page_for, make_user):
    _, _, share_path = shared_chat
    stranger_page = page_for(make_user())
    stranger_page.goto(share_path)
    _sent_home(stranger_page)
