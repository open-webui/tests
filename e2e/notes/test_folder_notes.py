"""Journey: notes filed in a chat folder, on the folder's page, the Notes page and the note itself.

The section switch in the folder page's header offers Notes beside Chats and Knowledge, with the
number of notes filed in the folder. New Note there, and New Note in the folder tab's menu, make a
note filed in the folder and open it; back on the folder page it is listed, the note edited last
first, and still after a reload. A row's menu renames a note in place and deletes it after a
confirmation, which also takes it off the Notes page. Move in a note's menu, on the Notes page and
in the note itself, files the note in a folder or takes it out with No folder; the Notes page names
the note's folder on its row.
A member of a folder shared for reading finds every note in it, opens one read-only and gets no
New Note and no Rename, Move or Delete; a member who may write gets New Note, and the note made
there is filed in the shared folder. On a phone the Notes view is reachable and a tap opens a note,
and the keyboard reaches a note with Tab and opens it with Enter. Without the Notes permission the
switch offers no Notes. The values behind these steps are pinned by the integration twin,
integration/notes/test_folder_notes.py.

Discriminates: journey, no fix. On dev d599f7a08 all pass, five runs in a
row. In backend copies of it: the folder's note list always empty turns the create, menu, move,
shared, phone and keyboard tests red; the list ignoring the folder turns the menu, move and shared
tests red; the list sorted oldest first turns the create test red; a note no longer readable
through its folder turns the shared test red; the session always granting Notes turns the
permission test red.
"""

from __future__ import annotations

import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from utils.notes_page import NOTE_URL, note_row
from utils.phone import PHONE, expect_on_screen

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}
PERMISSIONS = "/api/v1/users/default/permissions"


def _ok(response) -> dict:
    assert response.status_code == 200, (
        f"{response.request.url}: {response.status_code} {response.text}"
    )
    return response.json()


def _folder(owner: Actor, name: str, parent_id: str | None = None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name, "parent_id": parent_id})
    return _ok(created)["id"]


def _note(author: Actor, title: str, folder_id: str | None = None) -> str:
    body = {"title": title, "data": {"content": {"md": f"text of {title}"}}, "access_grants": []}
    if folder_id:
        body["folder_id"] = folder_id
    with author.client() as client:
        return _ok(client.post("/api/v1/notes/create", json=body))["id"]


def _stored(actor: Actor, note_id: str) -> dict:
    with actor.client() as client:
        return _ok(client.get(f"/api/v1/notes/{note_id}"))


def _folder_titles(actor: Actor, folder_id: str) -> list[str]:
    with actor.client() as client:
        listed = _ok(client.get(f"/api/v1/folders/{folder_id}/notes"))
    return [note["title"] for note in listed["items"]]


def folder_region(page: Page) -> Locator:
    return page.get_by_role("region", name="Folder", exact=True)


def section_switch(page: Page) -> Locator:
    return folder_region(page).get_by_role("button", name="Switch section", exact=True)


def section_item(page: Page, section: str) -> Locator:
    """A section in the open section switch, named with its count when it has one."""
    return page.get_by_role("menuitemradio", name=re.compile(rf"^{section}( \d+)?$"))


def show_notes(page: Page, press=Locator.click) -> None:
    """Pick Notes in the folder page's section switch."""
    press(section_switch(page))
    press(section_item(page, "Notes"))
    expect(section_switch(page)).to_contain_text("Notes")


def note_link(page: Page, title: str) -> Locator:
    return folder_region(page).get_by_role("link", name=title, exact=True)


def note_links(page: Page) -> Locator:
    return folder_region(page).get_by_role("listitem").get_by_role("link")


def row_menu(page: Page, title: str, scope: Locator | None = None) -> Locator:
    """Open the Note Menu of the note row titled `title`, in the folder page or in `scope`."""
    link = page.get_by_role("link", name=title, exact=True)
    row = (scope or folder_region(page)).get_by_role("listitem").filter(has=link)
    # the button takes the date's place only while the pointer is on the row
    row.hover()
    row.get_by_role("button", name="Note Menu", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem").first).to_be_visible()
    return menu


def open_folder(page: Page, folder_id: str, name: str) -> None:
    page.goto(f"/folders/{folder_id}")
    # since 1fbb42be9 a folder opens on the tab it was left on, here the note
    page.get_by_role("tab", name=name, exact=True).click()
    expect(section_switch(page)).to_be_visible()


def title_note(page: Page, title: str) -> str:
    """Type `title` into the open note's Title box and leave it; the note's id."""
    expect(page).to_have_url(NOTE_URL)
    note_id = page.url.rsplit("/", 1)[1]
    # the folder keeps each note it opened in a tab of its own; only the selected one shows
    box = page.get_by_role("textbox", name="Title", exact=True).filter(visible=True)
    expect(box).to_be_editable()
    box.fill(title)
    page.get_by_label("Write something...").filter(visible=True).click()
    return note_id


def wait_stored_title(actor: Actor, note_id: str, title: str) -> None:
    deadline = time.monotonic() + 10
    stored = _stored(actor, note_id)["title"]
    while stored != title and time.monotonic() < deadline:
        time.sleep(0.2)
        stored = _stored(actor, note_id)["title"]
    assert stored == title, f"the note's title was saved as {stored!r}, not {title!r}"


def test_new_notes_from_the_folder_page_are_filed_and_listed_newest_first(page_for, make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_name = f"Harbour {tag}"
    folder_id = _folder(owner, folder_name)
    page = page_for(owner, path=f"/folders/{folder_id}")
    show_notes(page)
    expect(folder_region(page).get_by_text("No Notes", exact=True)).to_be_visible()

    folder_region(page).get_by_role("button", name="New Note", exact=True).first.click()
    first = title_note(page, f"Tide table {tag}")
    wait_stored_title(owner, first, f"Tide table {tag}")
    assert _stored(owner, first)["folder_id"] == folder_id, (
        "New Note on the folder page did not file the note"
    )

    open_folder(page, folder_id, folder_name)
    page.get_by_role("button", name="Folder options", exact=True).click()
    page.get_by_role("menuitem", name="New Note", exact=True).click()
    second = title_note(page, f"Berth list {tag}")
    wait_stored_title(owner, second, f"Berth list {tag}")
    assert _stored(owner, second)["folder_id"] == folder_id, (
        "New Note in the folder menu did not file the note"
    )

    open_folder(page, folder_id, folder_name)
    show_notes(page)
    expect(section_switch(page)).to_have_text(re.compile(r"Notes\s*2"))
    expect(note_links(page)).to_have_text([f"Berth list {tag}", f"Tide table {tag}"])

    note_link(page, f"Tide table {tag}").click()
    expect(page).to_have_url(re.compile(rf"/notes/{first}$"))
    title_note(page, f"Tide table, spring {tag}")
    wait_stored_title(owner, first, f"Tide table, spring {tag}")
    open_folder(page, folder_id, folder_name)
    show_notes(page)
    expect(note_links(page)).to_have_text([f"Tide table, spring {tag}", f"Berth list {tag}"])

    page.reload()
    show_notes(page)
    expect(note_links(page)).to_have_text([f"Tide table, spring {tag}", f"Berth list {tag}"])


def test_a_rows_menu_renames_a_note_and_deletes_it_after_confirming(page_for, make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner, f"Harbour {tag}")
    kept = _note(owner, f"Kept {tag}", folder_id)
    doomed = _note(owner, f"Doomed {tag}", folder_id)
    page = page_for(owner, path=f"/folders/{folder_id}")
    show_notes(page)
    expect(note_links(page)).to_have_count(2)

    row_menu(page, f"Kept {tag}").get_by_role("menuitem", name="Rename", exact=True).click()
    rename = folder_region(page).get_by_role("textbox", name="Rename", exact=True)
    expect(rename).to_be_focused()
    rename.fill(f"Renamed {tag}")
    rename.press("Enter")
    expect(note_link(page, f"Renamed {tag}")).to_be_visible()
    wait_stored_title(owner, kept, f"Renamed {tag}")

    row_menu(page, f"Doomed {tag}").get_by_role("menuitem", name="Delete", exact=True).click()
    dialog = page.get_by_role("dialog", name="Delete note?")
    expect(dialog).to_be_visible()
    expect(note_link(page, f"Doomed {tag}")).to_be_visible()
    dialog.get_by_role("button", name="Confirm", exact=True).click()
    expect(note_link(page, f"Doomed {tag}")).to_have_count(0)
    expect(note_links(page)).to_have_text([f"Renamed {tag}"])
    expect(section_switch(page)).to_have_text(re.compile(r"Notes\s*1"))
    with owner.client() as client:
        assert client.get(f"/api/v1/notes/{doomed}").status_code == 404, (
            "the deleted note is still stored"
        )

    page.goto("/notes")
    expect(note_row(page, f"Renamed {tag}")).to_be_visible()
    expect(note_row(page, f"Doomed {tag}")).to_have_count(0)


def _pick_move(page: Page, menu: Locator, target: str) -> None:
    menu.get_by_role("menuitem", name="Move", exact=True).hover()
    page.get_by_role("menuitem", name=target, exact=True).click()


def test_move_files_a_note_in_a_folder_and_takes_it_out_again(page_for, make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_name = f"Harbour {tag}"
    folder_id = _folder(owner, folder_name)
    note_id = _note(owner, f"Drifter {tag}")
    page = page_for(owner, path="/notes")
    row = note_row(page, f"Drifter {tag}")
    expect(row).to_be_visible()
    expect(row.get_by_text(folder_name)).to_have_count(0)

    _pick_move(page, row_menu(page, f"Drifter {tag}", page.get_by_role("main")), folder_name)
    expect(row.get_by_text(folder_name, exact=True)).to_be_visible()
    assert _folder_titles(owner, folder_id) == [f"Drifter {tag}"]

    page.goto(f"/notes/{note_id}")
    page.get_by_role("button", name="Note Menu", exact=True).click()
    menu = page.get_by_role("menu").first
    _pick_move(page, menu, "No folder")
    expect(page.get_by_role("menuitem", name="No folder", exact=True)).to_have_count(0)
    open_folder(page, folder_id, folder_name)
    show_notes(page)
    expect(folder_region(page).get_by_text("No Notes", exact=True)).to_be_visible()
    page.goto("/notes")
    row = note_row(page, f"Drifter {tag}")
    expect(row).to_be_visible()
    expect(row.get_by_text(folder_name)).to_have_count(0)


def test_shared_folder_members_find_its_notes_and_only_writers_add_one(page_for, make_user, admin):
    owner, reader, writer = make_user(), make_user(), make_user()
    make_group(admin, [owner], FOLDER_SHARING)
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner, f"Harbour {tag}")
    with owner.client() as client:
        grants = [
            grant("user", reader.id, "read"),
            grant("user", writer.id, "read"),
            grant("user", writer.id, "write"),
        ]
        _ok(
            client.post(
                f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
            )
        )
    _note(owner, f"Owner's plan {tag}", folder_id)
    _note(writer, f"Writer's draft {tag}", folder_id)

    page = page_for(reader, path=f"/folders/{folder_id}")
    show_notes(page)
    expect(note_links(page)).to_have_count(2)
    expect(note_link(page, f"Owner's plan {tag}")).to_be_visible()
    expect(note_link(page, f"Writer's draft {tag}")).to_be_visible()
    expect(folder_region(page).get_by_role("button", name="New Note")).to_have_count(0)
    menu = row_menu(page, f"Owner's plan {tag}")
    for entry in ("Rename", "Move", "Delete"):
        expect(menu.get_by_role("menuitem", name=entry, exact=True)).to_have_count(0)
    page.keyboard.press("Escape")

    note_link(page, f"Owner's plan {tag}").click()
    expect(page).to_have_url(NOTE_URL)
    expect(page.get_by_role("textbox", name="Title", exact=True)).to_be_disabled()
    expect(page.get_by_text("Read-Only Access")).to_be_visible()

    writer_page = page_for(writer, path=f"/folders/{folder_id}")
    show_notes(writer_page)
    expect(note_links(writer_page)).to_have_count(2)
    folder_region(writer_page).get_by_role("button", name="New Note", exact=True).first.click()
    created = title_note(writer_page, f"Writer's second {tag}")
    wait_stored_title(writer, created, f"Writer's second {tag}")
    assert f"Writer's second {tag}" in _folder_titles(owner, folder_id), (
        "the writing member's new note is not in the shared folder"
    )


def test_on_a_phone_the_notes_view_opens_a_note_with_a_tap(page_for, make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner, f"Harbour {tag}")
    note_id = _note(owner, f"Pocket note {tag}", folder_id)
    page = page_for(owner, path=f"/folders/{folder_id}", **PHONE)
    show_notes(page, press=Locator.tap)
    link = note_link(page, f"Pocket note {tag}")
    expect_on_screen(link)
    expect_on_screen(folder_region(page).get_by_role("button", name="New Note", exact=True).first)
    link.tap()
    expect(page).to_have_url(re.compile(rf"/notes/{note_id}$"))
    expect(page.get_by_role("textbox", name="Title", exact=True)).to_have_value(
        f"Pocket note {tag}"
    )


def test_the_keyboard_reaches_a_note_and_opens_it(page_for, make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner, f"Harbour {tag}")
    note_id = _note(owner, f"Keyed note {tag}", folder_id)
    page = page_for(owner, path=f"/folders/{folder_id}")
    show_notes(page)
    link = note_link(page, f"Keyed note {tag}")
    expect(link).to_be_visible()

    folder_region(page).get_by_role("button", name="New Note", exact=True).first.focus()
    for _ in range(8):
        page.keyboard.press("Tab")
        if link.evaluate("element => element === document.activeElement"):
            break
    expect(link, "Tab never reached the note in the folder's list").to_be_focused()
    page.keyboard.press("Enter")
    expect(page).to_have_url(re.compile(rf"/notes/{note_id}$"))


def test_without_the_notes_permission_the_folder_page_offers_no_notes(
    page_for, make_user, admin, preserve
):
    owner = make_user()
    folder_id = _folder(owner, f"Harbour {uuid.uuid4().hex[:6]}")
    allowed = page_for(owner, path=f"/folders/{folder_id}")
    section_switch(allowed).click()
    expect(section_item(allowed, "Notes")).to_be_visible()

    preserve("permissions")
    with admin.client() as client:
        permissions = _ok(client.get(PERMISSIONS))
        permissions["features"] = {**permissions["features"], "notes": False}
        _ok(client.post(PERMISSIONS, json=permissions))
    page = page_for(owner, path=f"/folders/{folder_id}")
    section_switch(page).click()
    expect(section_item(page, "Chats")).to_be_visible()
    expect(section_item(page, "Notes")).to_have_count(0)
