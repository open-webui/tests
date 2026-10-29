"""Journey: the folder menu and the chat header menu mark, export, move, rename, delete and scroll.

The "More" menu on a folder's sidebar row marks every chat in the folder as read, the unread dot
of a chat outside it stays, and the chats stay read after a reload. Its Export downloads a file
holding the folder's chats and no others, read back from disk. The "Chat actions" menu on an open
chat moves the chat into a folder, listed there after a reload, and deletes it after the confirm
dialog: the page goes home, the sidebar and the API no longer know the chat. In a chat long enough
to scroll, left scrolled to its end, "Scroll to Top" brings the first message back into view. The
unread dot has no text or role of its own, so it is found by its colour class.

Since 0.12.5 (551b03ff9) Rename in the header menu turns the chat's title into a text box: Enter
saves the new title, shown in the header, the sidebar and the stored chat, while a blank title is
refused with a toast and nothing is sent. The owner of a shared folder sees a member's chat in it
read-only yet gets Rename there and saves it (f517e528d); a member reading the owner's chat in the
folder gets no Rename.

Discriminates: passes on dev 30f3f6a8f; in a backend copy each test fails when its route answers
without storing the change: the folder read route marking nothing read, the folder chat list
answering empty, the chat folder route not moving the chat and the chat delete route not deleting.
In a frontend copy of dev 30f3f6a8f the Scroll to Top entry closes the menu and leaves the chat at
its end. The rename tests pass on dev c5981aaea (frontend build of 05e115366, no header change
since); in a frontend copy whose header rename never sends the title, whose blank check is dropped
and whose menu shows Rename only on writable views, each of the three goes red.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.chat_history import seed_chat
from utils.sidebar_folders import folder_link, folder_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER_NAME = "Harbour notes"


def _create_folder(account, name: str) -> str:
    with account.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text
    return folder_id


def _create_chat(account, title: str, folder_id: str | None = None, unread: bool = False) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "where to?"}, {"role": "assistant", "content": "ashore"}],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert titled.status_code == 200, titled.text
        if folder_id:
            moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
            assert moved.status_code == 200, moved.text
        if unread:
            marked = client.post(f"/api/v1/chats/{chat_id}/unread")
            assert marked.status_code == 200, marked.text
    return chat_id


def _open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    _show_folders(sidebar)
    return sidebar


def _show_folders(sidebar: Locator) -> None:
    section = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(section).to_be_visible()
    if section.get_attribute("aria-expanded") != "true":
        section.click()


def _reloaded(sidebar: Locator) -> Locator:
    sidebar.page.reload()  # the sidebar stays open
    _show_folders(sidebar)
    return sidebar


def _folder_menu(sidebar: Locator, name: str) -> Locator:
    """Open the "More" menu on the folder's row, a button shown while the row has focus."""
    row = folder_row(sidebar, name)
    folder_link(sidebar, name).focus()
    row.get_by_role("button").last.click()
    return sidebar.page.get_by_role("menu")


def _entry(sidebar: Locator, title: str) -> Locator:
    entry = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    expect(entry).to_be_visible()
    return entry


def _unread_dot(entry: Locator) -> Locator:
    return entry.locator(".bg-sky-500.rounded-full")


def _stored_chat_ids(account, folder_id: str) -> set[str]:
    with account.client() as client:
        listed = client.get(f"/api/v1/chats/folder/{folder_id}/list")
    assert listed.status_code == 200, listed.text
    return {chat["id"] for chat in listed.json()}


def _open_header_menu(page: Page, chat_id: str) -> Locator:
    page.goto(f"/c/{chat_id}")
    page.get_by_label("Chat actions").click()
    return page.get_by_role("menu")


def test_mark_all_as_read_clears_the_dots_of_the_folders_chats_only(make_user, page_for):
    account = make_user()
    suffix = uuid.uuid4().hex[:6]
    inside = [f"Tides {suffix}", f"Moorings {suffix}"]
    outside = f"Weather {suffix}"
    folder_id = _create_folder(account, FOLDER_NAME)
    for title in inside:
        _create_chat(account, title, folder_id, unread=True)
    _create_chat(account, outside, unread=True)
    sidebar = _open_sidebar(page_for(account))

    # every dot shows first, so their absence below is meaningful
    for title in [*inside, outside]:
        expect(_unread_dot(_entry(sidebar, title))).to_be_visible()

    _folder_menu(sidebar, FOLDER_NAME).get_by_role("button", name="Mark all as read").click()

    for title in inside:
        expect(_unread_dot(_entry(sidebar, title))).to_have_count(0)
    expect(_unread_dot(_entry(sidebar, outside))).to_be_visible()

    _reloaded(sidebar)
    for title in inside:
        expect(_unread_dot(_entry(sidebar, title))).to_have_count(0)
    expect(_unread_dot(_entry(sidebar, outside))).to_be_visible()


def test_folder_export_downloads_the_folders_chats_and_no_others(make_user, page_for, tmp_path):
    account = make_user()
    suffix = uuid.uuid4().hex[:6]
    inside = {f"Tides {suffix}", f"Moorings {suffix}"}
    folder_id = _create_folder(account, FOLDER_NAME)
    other_folder_id = _create_folder(account, "Elsewhere")
    inside_ids = {_create_chat(account, title, folder_id) for title in sorted(inside)}
    others = {f"Other folder {suffix}": other_folder_id, f"Loose {suffix}": None}
    for title, other_folder in others.items():
        _create_chat(account, title, other_folder)
    sidebar = _open_sidebar(page_for(account))
    for title in [*inside, *others]:
        _entry(sidebar, title)

    menu = _folder_menu(sidebar, FOLDER_NAME)
    with sidebar.page.expect_download() as download_info:
        menu.get_by_role("button", name="Export").click()
    download = download_info.value
    path = tmp_path / download.suggested_filename
    download.save_as(path)
    exported = json.loads(path.read_text())

    assert download.suggested_filename.startswith(f"folder-{FOLDER_NAME}-export-")
    assert download.suggested_filename.endswith(".json")
    assert {chat["title"] for chat in exported} == inside
    assert {chat["id"] for chat in exported} == inside_ids


def test_header_menu_move_files_the_open_chat_into_the_chosen_folder(make_user, page_for):
    account = make_user()
    title = f"Anchorage {uuid.uuid4().hex[:6]}"
    folder_id = _create_folder(account, FOLDER_NAME)
    chat_id = _create_chat(account, title)
    page = page_for(account)

    menu = _open_header_menu(page, chat_id)
    menu.get_by_role("button", name="Move").hover()
    page.get_by_role("menu").get_by_role("button", name=FOLDER_NAME).click()

    expect(page.get_by_text("Chat moved successfully")).to_be_visible()
    assert _stored_chat_ids(account, folder_id) == {chat_id}

    page.reload()
    sidebar = _open_sidebar(page)
    expect(folder_row(sidebar, FOLDER_NAME)).to_be_visible()
    expect(sidebar.get_by_text("No chats")).to_have_count(0)
    expect(_entry(sidebar, title)).to_be_visible()


def test_header_menu_delete_removes_the_open_chat_for_good(make_user, page_for):
    account = make_user()
    title = f"Wreck {uuid.uuid4().hex[:6]}"
    kept = f"Lighthouse {uuid.uuid4().hex[:6]}"
    chat_id = _create_chat(account, title)
    _create_chat(account, kept)
    page = page_for(account)

    menu = _open_header_menu(page, chat_id)
    menu.get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Delete chat?").get_by_role("button", name="Confirm").click()

    expect(page.get_by_text("Chat deleted.")).to_be_visible()
    expect(page).not_to_have_url(f"**/c/{chat_id}")
    with account.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").status_code != 200, "the chat is still stored"

    page.reload()
    sidebar = _open_sidebar(page)
    expect(_entry(sidebar, kept)).to_be_visible()
    expect(sidebar.locator("#sidebar-chat-group").filter(has_text=title)).to_have_count(0)


def test_header_menu_scroll_to_top_brings_the_first_message_into_view(make_user, page_for):
    account = make_user()
    opening = f"Opening line {uuid.uuid4().hex[:6]}"
    filler = "The tide table runs on and on. " * 40
    messages = [{"role": "user", "content": opening}]
    for turn in range(12):
        messages.append({"role": "assistant", "content": f"{filler} Entry {turn}"})
        messages.append({"role": "user", "content": f"Question {turn}"})
    with account.client() as client:
        chat_id, _ = seed_chat(client, messages)
    page = page_for(account, path=f"/c/{chat_id}")
    first = page.get_by_text(opening, exact=True)
    expect(page.get_by_text("Question 11", exact=True)).to_be_in_viewport()
    expect(first).not_to_be_in_viewport()

    page.get_by_label("Chat actions").click()
    page.get_by_role("menu").get_by_role("button", name="Scroll to Top").click()

    expect(first).to_be_in_viewport()


def _header(page: Page) -> Locator:
    """The bar over the open chat, holding its title and the "Chat actions" menu."""
    return page.locator("nav").filter(has=page.get_by_label("Chat actions"))


def _start_rename(page: Page, title: str) -> Locator:
    """Pick Rename in the open chat's header menu; returns the text box that replaced the title."""
    page.get_by_label("Chat actions").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Rename").click()
    box = _header(page).get_by_role("textbox", name="Title")
    expect(box).to_be_focused()
    expect(box).to_have_value(title)
    expect(_header(page).get_by_role("heading", level=1)).to_have_count(0)
    return box


def _save_title(page: Page, box: Locator, chat_id: str, new_title: str) -> int:
    box.fill(new_title)
    with page.expect_response(
        lambda response: (
            response.request.method == "POST" and response.url.endswith(f"/api/v1/chats/{chat_id}")
        )
    ) as answer:
        box.press("Enter")
    return answer.value.status


def _stored_title(account, chat_id: str) -> str:
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["title"]


def test_header_menu_rename_saves_the_new_title_on_enter(make_user, page_for):
    account = make_user()
    suffix = uuid.uuid4().hex[:6]
    title, renamed = f"Driftwood {suffix}", f"Salvage {suffix}"
    chat_id = _create_chat(account, title)
    page = page_for(account, path=f"/c/{chat_id}")
    sidebar = _open_sidebar(page)
    _entry(sidebar, title)

    box = _start_rename(page, title)
    status = _save_title(page, box, chat_id, renamed)

    assert status == 200, f"saving the new title got {status}"
    expect(_header(page).get_by_role("heading", level=1)).to_have_text(renamed)
    expect(box).to_have_count(0)
    assert _stored_title(account, chat_id) == renamed
    expect(_entry(sidebar, renamed)).to_be_visible()
    expect(sidebar.locator("#sidebar-chat-group").filter(has_text=title)).to_have_count(0)


def test_header_menu_rename_refuses_a_blank_title_and_keeps_the_old_one(make_user, page_for):
    account = make_user()
    title = f"Lantern {uuid.uuid4().hex[:6]}"
    chat_id = _create_chat(account, title)
    page = page_for(account, path=f"/c/{chat_id}")
    saves = []
    page.on(
        "request",
        lambda request: (
            saves.append(request.url)
            if request.method == "POST" and request.url.endswith(f"/api/v1/chats/{chat_id}")
            else None
        ),
    )

    box = _start_rename(page, title)
    box.fill("   ")
    box.press("Enter")

    expect(page.get_by_text("Title cannot be an empty string.")).to_be_visible()
    expect(_header(page).get_by_role("heading", level=1)).to_have_text(title)
    assert saves == [], f"a blank title was sent to the server: {saves}"
    assert _stored_title(account, chat_id) == title


def test_shared_folder_owner_renames_a_members_chat_from_its_header_but_not_the_member(
    make_user, admin, page_for
):
    owner, member = make_user(), make_user()
    group_id = make_group(admin, [owner, member], {"sharing": {"folders": True}})
    suffix = uuid.uuid4().hex[:6]
    folder_id = _create_folder(owner, f"Harbour {suffix}")
    grants = [
        {"principal_type": "group", "principal_id": group_id, "permission": permission}
        for permission in ("read", "write")
    ]
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text
    title, renamed = f"My mooring {suffix}", f"Berth 4 {suffix}"
    members_chat = _create_chat(member, title, folder_id)
    owners_chat = _create_chat(owner, f"Harbour log {suffix}", folder_id)
    owner_page = page_for(owner, path=f"/c/{members_chat}")

    # the owner sees the member's chat read-only, yet may rename it as the folder's owner
    expect(owner_page.get_by_role("button", name="Clone Chat")).to_be_visible()
    box = _start_rename(owner_page, title)
    status = _save_title(owner_page, box, members_chat, renamed)

    assert status == 200, f"the owner's rename got {status}"
    expect(_header(owner_page).get_by_role("heading", level=1)).to_have_text(renamed)
    assert _stored_title(member, members_chat) == renamed

    # a member reading the owner's chat in the folder gets no Rename
    member_page = page_for(member, path=f"/c/{owners_chat}")
    expect(member_page.get_by_role("button", name="Clone Chat")).to_be_visible()
    member_page.get_by_label("Chat actions").click()
    menu = member_page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Copy")).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Rename")).to_have_count(0)
