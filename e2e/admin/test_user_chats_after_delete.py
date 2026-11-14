"""Deleting a chat from a user's chat list in the admin panel left the list empty, #30601.

Fix commit `a7d81ebdc` (open-webui/open-webui#30604). The admin's view of a user's chats loads
more pages as its end scrolls into view, and a short list asks for page 2 at once, gets nothing
and stops. After a delete the view reloaded from the page it had reached, page 2, so the chats
that were left vanished and it read "No results found". It now reloads from page 1.

Deleting the last chat then shows the empty list, a closing step of the same test.

Discriminates: passes on dev efe63bd34; with a7d81ebdc reverted the list is empty after the delete.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _seed_chats(owner: Actor, titles: list[str]) -> None:
    with owner.client() as client:
        for title in titles:
            created = client.post("/api/v1/chats/new", json={"chat": {"title": title}})
            assert created.status_code == 200, created.text


def _open_user_chats(page: Page, owner: Actor):
    page.goto("/admin/users")
    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(owner.email)
    users.get_by_role("row").filter(has_text=owner.email).get_by_role(
        "button", name="Chats"
    ).click()
    chats = page.get_by_role("dialog").filter(has_text=f"{owner.name}'s Chats")
    # the list has asked for its next page and found the end
    expect(chats.get_by_text("Loading...")).to_have_count(0)
    return chats


def _delete_from_list(page: Page, chats, title: str) -> None:
    row = chats.get_by_role("link", name=title).locator("..")
    row.hover()
    tooltip_button(row, "Delete Chat").click()
    page.get_by_role("button", name="Confirm").click()


def test_the_remaining_chats_stay_listed_after_a_delete(page_for, make_user):
    owner = make_user()
    _seed_chats(owner, ["Kept chat alpha", "Deleted chat beta"])
    page = page_for(make_user(role="admin"))
    chats = _open_user_chats(page, owner)
    expect(chats.get_by_role("link", name="Kept chat alpha")).to_be_visible()

    _delete_from_list(page, chats, "Deleted chat beta")

    expect(chats.get_by_role("link", name="Deleted chat beta")).to_have_count(0)
    expect(chats.get_by_role("link", name="Kept chat alpha")).to_be_visible()
    expect(chats.get_by_text("No results found")).to_have_count(0)

    _delete_from_list(page, chats, "Kept chat alpha")

    expect(chats.get_by_text("No results found")).to_be_visible()
    with owner.client() as client:
        assert client.get("/api/v1/chats/").json() == []
