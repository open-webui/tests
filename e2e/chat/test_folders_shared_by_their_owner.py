"""Regression: the owner's sidebar never marked the chats of a folder they had shared.

Commit ce18eca34 lists a folder its owner shared in the owner's own shared folders, and the
sidebar then shows the owner's picture on every chat in that folder and its sub-folders, with the
owner's name on hover, as members see it. A folder the owner keeps to themselves, or stops
sharing, shows plain chats.

Discriminates: passes on the dev 87a937459 build. All three go red with the sidebar change of
ce18eca34 reverted in a frontend copy, and with the dev build on a backend copy with ce18eca34
reverted: no picture on the owner's chats, for the unshare test on its still shared folder.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}


@pytest.fixture
def crew(make_user, admin):
    """An owner and a member in one group allowed to share folders, and that group's id."""
    owner, member = make_user(), make_user()
    return owner, member, make_group(admin, [owner, member], FOLDER_SHARING)


def create_folder(owner, parent_id: str | None = None) -> str:
    form = {"name": f"Harbour {uuid.uuid4().hex[:6]}"}
    if parent_id:
        form["parent_id"] = parent_id
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json=form)
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text
    return folder_id


def filed_chat(owner, folder_id: str) -> str:
    """A chat of the owner's filed in the folder; returns its title."""
    title = f"Tide log {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "when is high tide?"},
                {"role": "assistant", "content": "High tide at noon."},
            ],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert titled.status_code == 200, titled.text
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
    return title


def share(owner, folder_id: str, group_id: str | None) -> None:
    """Share the folder with the group for reading, as the Share dialog saves it, or stop."""
    grants = []
    if group_id:
        grants = [{"principal_type": "group", "principal_id": group_id, "permission": "read"}]
    with owner.client() as client:
        updated = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert updated.status_code == 200, updated.text


def open_sidebar(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    return sidebar


def owner_picture(sidebar: Locator, title: str, owner) -> Locator:
    chat = sidebar.get_by_role("button", name=title)
    expect(chat).to_be_visible()
    return chat.locator(f'img[src$="/users/{owner.id}/profile/image"]')


def test_the_owner_sees_their_picture_on_chats_in_a_folder_they_shared(page_for, crew):
    owner, _, group_id = crew
    shared_id, kept_id = create_folder(owner), create_folder(owner)
    shared_title, kept_title = filed_chat(owner, shared_id), filed_chat(owner, kept_id)
    share(owner, shared_id, group_id)
    page = page_for(owner)

    sidebar = open_sidebar(page)

    picture = owner_picture(sidebar, shared_title, owner)
    expect(picture).to_be_visible()
    picture.hover()
    expect(page.get_by_role("tooltip").filter(has_text=owner.name)).to_be_visible()
    expect(owner_picture(sidebar, kept_title, owner)).to_have_count(0)


def test_the_owner_sees_their_picture_on_chats_in_a_sub_folder_of_it(page_for, crew):
    owner, _, group_id = crew
    shared_id = create_folder(owner)
    child_id = create_folder(owner, parent_id=shared_id)
    child_title = filed_chat(owner, child_id)
    share(owner, shared_id, group_id)
    page = page_for(owner)

    sidebar = open_sidebar(page)

    expect(owner_picture(sidebar, child_title, owner)).to_be_visible()


def test_a_folder_the_owner_stops_sharing_shows_plain_chats_again(page_for, crew):
    owner, _, group_id = crew
    stopped_id, still_id = create_folder(owner), create_folder(owner)
    stopped_title, still_title = filed_chat(owner, stopped_id), filed_chat(owner, still_id)
    share(owner, stopped_id, group_id)
    share(owner, still_id, group_id)
    share(owner, stopped_id, None)
    page = page_for(owner)

    sidebar = open_sidebar(page)

    expect(owner_picture(sidebar, still_title, owner)).to_be_visible()
    expect(owner_picture(sidebar, stopped_title, owner)).to_have_count(0)
