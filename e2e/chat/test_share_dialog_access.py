"""Regression: the Share dialog misreported who can view a link.

Two fixes to the chat's Share dialog, both in the frontend:

- Relinking (893251ed1, open-webui/open-webui#31037, issue open-webui/open-webui#31029). After
  "delete this link" and a new link in the same open dialog, the dialog kept showing the old
  link's access, such as Public or a granted account, while the new link was private. It now
  reloads the access of the new link.
- Wording (e26b2edaa, open-webui/open-webui#31420, issue open-webui/open-webui#31417). Before a
  link exists the dialog said anyone with the URL could view the chat, but a new link starts
  private. It now says the link is private until you choose who can view it.
- First click (issue open-webui/open-webui#31486). After 8fc416ee7 the first click inside a
  dialog opened from a menu was swallowed, so Copy Link needed two clicks. 0082c153f
  (open-webui/open-webui#31488) fixed the sidebar chat menu, and ef67cc3fa
  (open-webui/open-webui#31496, issue open-webui/open-webui#31493) the chat header menu these
  tests open Share from, clicking once.

Discriminates: fails on dev 00a245b9f (the first Copy Link click creates no link) and passes on dev
176d31d1d and a5bc78300; it fails on a build with the relinking or wording fix reverted (the
relinked dialog still shows Public and the granted account, or the old sentence is shown).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.access_control import choose_visibility, visibility
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
QUESTION = "when does the ferry leave?"
ANSWER = "the ferry leaves at nine"
PRIVATE_NOTE = "Your link is private until you choose who can view it."
OLD_NOTE = "Users with the URL will be able to view the shared chat."
PRIVATE_HINT = "Only select users and groups with permission can access"
PUBLIC_HINT = "Accessible to all users"


@pytest.fixture
def public_sharing(preserve, admin):
    """Accounts may share chats with every signed-in user, as the admin can allow."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        permissions.setdefault("sharing", {})["public_chats"] = True
        saved = client.post(PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def owner_page(public_sharing, page_for, make_user, upstream) -> Page:
    """A page on a chat with one answered question, owned by a fresh account."""
    page = page_for(make_user())
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    return page


def share_dialog(page: Page) -> Locator:
    page.get_by_role("button", name="Chat actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Share").click()
    dialog = page.get_by_role("dialog").filter(has_text="Share Chat")
    expect(dialog).to_be_visible()
    return dialog


def create_link(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Copy Link").click()
    expect(dialog.get_by_role("link", name="You have shared this chat before")).to_be_visible()


def make_public(dialog: Locator) -> None:
    choose_visibility(dialog, "Public")
    expect(dialog.page.get_by_text("Access updated").first).to_be_visible()
    expect(dialog.get_by_text(PUBLIC_HINT)).to_be_visible()


def grant(dialog: Locator, account) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(account.name)
    picker.get_by_role("button", name=account.name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(dialog.get_by_text(account.name)).to_be_visible()


def relink(dialog: Locator) -> None:
    dialog.get_by_role("button", name="delete this link").click()
    expect(dialog.get_by_role("button", name="Copy Link")).to_be_visible()
    create_link(dialog)


def test_a_new_link_after_deleting_a_public_one_shows_private(owner_page):
    dialog = share_dialog(owner_page)
    create_link(dialog)
    make_public(dialog)

    relink(dialog)

    expect(visibility(dialog)).to_have_text("Private")
    expect(dialog.get_by_text(PRIVATE_HINT)).to_be_visible()
    expect(dialog.get_by_text(PUBLIC_HINT)).to_have_count(0)


def test_a_new_link_does_not_list_the_old_links_accounts(owner_page, make_user):
    viewer = make_user()
    dialog = share_dialog(owner_page)
    create_link(dialog)
    grant(dialog, viewer)

    relink(dialog)

    expect(dialog.get_by_text(viewer.name)).to_have_count(0)
    expect(dialog.get_by_text("No access grants. Private to you.")).to_be_visible()


def test_a_reopened_dialog_still_shows_the_links_real_access(owner_page):
    dialog = share_dialog(owner_page)
    create_link(dialog)
    make_public(dialog)
    dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_be_hidden()

    dialog = share_dialog(owner_page)

    expect(visibility(dialog)).to_have_text("Public")


def test_the_dialog_says_a_new_link_starts_private(owner_page):
    dialog = share_dialog(owner_page)

    expect(dialog.get_by_text(PRIVATE_NOTE)).to_be_visible()
    expect(dialog.get_by_text(OLD_NOTE)).to_have_count(0)

    create_link(dialog)
    expect(visibility(dialog)).to_have_text("Private")
