"""Journey: the note's menu and its Access button, as the owner uses them in the editor.

Copy link puts the note's address on the clipboard, from the editor and from the Notes page, and
the address opens the note. Copy to clipboard puts the note's Markdown there as plain text and
its formatting as HTML. Download as plain text saves what the editor holds under the note's
title, and the Markdown download keeps a table cell of two lines inside its row
(open-webui/open-webui#31539, issue #31538). Upload files attaches a file to the note as a chip
above the text, stored with the note and still there after a reload, and the chip's close
button detaches it again. Delete asks for a confirmation, then goes to the Notes page where the
note is no longer listed, and the API no longer finds it. The Access button opens Access Control,
where an owner allowed to share notes adds a person or a group from the access list at Read or
Write, changes the level or removes them, or makes the note public; each change is saved at once,
and the other account then meets the note read-only, editable or not at all. Without the sharing
permission the panel has no access list.

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the copied link pointing at the Notes page (editor and list), the clipboard's plain text
getting the HTML, the plain-text download writing the HTML, the upload not saving the note's
files, the chip's close button not saving, the access panel never sending the grants, the editor's
Delete not calling the API, removing a person dropping only their write grant and the access list
shown without the sharing permission.
In the a5bc78300 build with #31539 reverted, the second line of a table cell starts a broken row.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]
READ_CLIPBOARD_HTML = """async () => {
    const [item] = await navigator.clipboard.read();
    return await (await item.getType('text/html')).text();
}"""


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


def _create_note(owner: Actor, title: str, markdown: str) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": markdown}}, "access_grants": []},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_note(owner: Actor, note_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def _open_note(page: Page, note_id: str, text: str) -> Locator:
    page.goto(f"/notes/{note_id}")
    editor = _note_editor(page)
    expect(editor).to_contain_text(text)
    return editor


def _open_note_menu(page: Page) -> Locator:
    # the "..." button has no name; it is the last menu button in the editor's header
    page.get_by_role("main").locator("[aria-haspopup=true]:visible").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Download")).to_be_visible()
    return menu


def _share_submenu(page: Page, menu: Locator) -> None:
    menu.get_by_role("button", name="Share").hover()
    expect(page.get_by_role("button", name="Copy link")).to_be_visible()


def _eventually(read: Callable[[], object], expected: object, timeout: float = 10.0) -> None:
    """Wait until `read()` returns `expected`; the editor saves a moment after a change."""
    deadline = time.monotonic() + timeout
    current = read()
    while current != expected and time.monotonic() < deadline:
        time.sleep(0.2)
        current = read()
    assert current == expected, f"expected {expected!r}, still {current!r}"


def _clipboard_text(page: Page) -> str:
    return page.evaluate("navigator.clipboard.readText()")


# ---------------------------------------------------------------- copy


def test_copy_link_in_the_editor_copies_an_address_that_opens_the_note(page_for, make_user):
    author = make_user()
    note_id = _create_note(author, _unique("Link me"), "reachable by link")
    page = page_for(author, permissions=CLIPBOARD)
    _open_note(page, note_id, "reachable by link")

    _share_submenu(page, _open_note_menu(page))
    page.get_by_role("button", name="Copy link").click()

    expect(page.get_by_text("Copied link to clipboard")).to_be_visible()
    link = _clipboard_text(page)
    assert link == f"{author.base_url}/notes/{note_id}"
    page.goto("/notes")
    page.goto(link)
    expect(_note_editor(page)).to_have_text("reachable by link")


def test_copy_link_on_the_notes_page_copies_the_notes_address(page_for, make_user):
    author = make_user()
    title = _unique("Listed")
    note_id = _create_note(author, title, "in the list")
    page = page_for(author, permissions=CLIPBOARD)
    page.goto("/notes")
    row = page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)

    row.get_by_label("Note Menu").click()
    _share_submenu(page, page.get_by_role("menu"))
    expect(page.get_by_role("button", name="Copy to clipboard")).to_have_count(0)
    page.get_by_role("button", name="Copy link").click()

    expect(page.get_by_text("Copied link to clipboard")).to_be_visible()
    assert _clipboard_text(page) == f"{author.base_url}/notes/{note_id}"


def test_copy_to_clipboard_copies_the_markdown_and_the_formatting(page_for, make_user):
    author = make_user()
    note_id = _create_note(author, _unique("Recipe"), "# Soup\n\nuse **fresh** leeks")
    page = page_for(author, permissions=CLIPBOARD)
    _open_note(page, note_id, "fresh")

    _share_submenu(page, _open_note_menu(page))
    page.get_by_role("button", name="Copy to clipboard").click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    assert _clipboard_text(page) == "# Soup\n\nuse **fresh** leeks"
    html = page.evaluate(READ_CLIPBOARD_HTML)
    assert "<strong>fresh</strong>" in html, html
    assert re.search(r"<h1[^>]*>Soup</h1>", html), html


# ---------------------------------------------------------------- download


def test_download_as_plain_text_saves_what_the_editor_holds(page_for, make_user):
    author = make_user()
    title = _unique("Shopping")
    note_id = _create_note(author, title, "eggs")
    page = page_for(author)
    editor = _open_note(page, note_id, "eggs")
    editor.click()
    page.keyboard.press("End")
    page.keyboard.type(" and flour")
    expect(editor).to_have_text("eggs and flour")

    menu = _open_note_menu(page)
    menu.get_by_role("button", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Plain text (.txt)").click()

    download = download_info.value
    assert download.suggested_filename == f"{title}.txt"
    assert Path(download.path()).read_text() == "eggs and flour"


TABLE_HTML = (
    "<table><tbody>"
    "<tr><th><p>Item</p></th><th><p>Qty</p></th></tr>"
    "<tr><td><p>Tea</p><p>Green</p></td><td><p>3</p></td></tr>"
    "</tbody></table><p>end</p>"
)


@pytest.mark.regression
def test_download_keeps_a_two_line_table_cell_inside_its_row(page_for, make_user):
    author = make_user()
    title = _unique("Pantry")
    with author.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": title,
                "data": {"content": {"html": TABLE_HTML, "md": ""}},
                "access_grants": [],
            },
        )
    assert created.status_code == 200, created.text
    note_id = created.json()["id"]
    page = page_for(author)
    editor = _open_note(page, note_id, "Green")
    editor.get_by_text("end", exact=True).click()
    page.keyboard.press("End")
    page.keyboard.type(" of list")
    _eventually(
        lambda: "end of list" in str(_stored_note(author, note_id)["data"]["content"]["md"]), True
    )

    menu = _open_note_menu(page)
    menu.get_by_role("button", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Plain text (.md)").click()

    markdown = Path(download_info.value.path()).read_text()
    assert re.search(r"^\| Tea(<br>)+Green \| 3 \|$", markdown, re.M), (
        f"a two-line table cell broke its row (open-webui/open-webui#31538): {markdown!r}"
    )


# ---------------------------------------------------------------- files


def _attach_file(page: Page, name: str, text: bytes) -> None:
    menu = _open_note_menu(page)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("button", name="Upload files").click()
    chooser.value.set_files([{"name": name, "mimeType": "text/plain", "buffer": text}])


def _stored_file_names(owner: Actor, note_id: str) -> list[str]:
    files = (_stored_note(owner, note_id).get("data") or {}).get("files") or []
    return [file.get("name") for file in files]


def test_a_file_uploaded_into_a_note_is_attached_and_kept(page_for, make_user):
    author = make_user()
    note_id = _create_note(author, _unique("Trip"), "see the itinerary")
    page = page_for(author)
    _open_note(page, note_id, "see the itinerary")

    _attach_file(page, "itinerary.txt", b"Monday: ferry to the island")

    chip = page.get_by_role("main").get_by_role("button", name=re.compile("itinerary.txt"))
    expect(chip).to_be_visible()
    _eventually(lambda: _stored_file_names(author, note_id), ["itinerary.txt"])
    page.reload()
    expect(chip).to_be_visible()


def test_the_close_button_on_a_notes_file_detaches_it(page_for, make_user):
    author = make_user()
    note_id = _create_note(author, _unique("Trip"), "see the itinerary")
    page = page_for(author)
    _open_note(page, note_id, "see the itinerary")
    _attach_file(page, "itinerary.txt", b"Monday: ferry to the island")
    _eventually(lambda: _stored_file_names(author, note_id), ["itinerary.txt"])

    chip = page.get_by_role("main").get_by_role("button", name=re.compile("itinerary.txt"))
    chip.hover()
    chip.get_by_role("button", name="Remove File").click()

    expect(chip).to_have_count(0)
    _eventually(lambda: _stored_file_names(author, note_id), [])
    page.reload()
    expect(_note_editor(page)).to_contain_text("see the itinerary")
    expect(chip).to_have_count(0)


# ---------------------------------------------------------------- delete


def test_delete_in_the_editor_removes_the_note_from_the_list_and_the_server(page_for, make_user):
    author = make_user()
    doomed = _unique("Doomed")
    kept = _unique("Kept")
    doomed_id = _create_note(author, doomed, "to be thrown out")
    _create_note(author, kept, "to be kept")
    page = page_for(author)
    _open_note(page, doomed_id, "to be thrown out")

    _open_note_menu(page).get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Delete note?").get_by_role("button", name="Confirm").click()

    expect(page.get_by_text("Note deleted successfully")).to_be_visible()
    expect(page).to_have_url(re.compile(r"/notes$"))
    cards = page.get_by_role("main").get_by_role("button", name="Open note")
    expect(cards.filter(has_text=kept)).to_have_count(1)
    expect(cards.filter(has_text=doomed)).to_have_count(0)
    with author.client() as client:
        assert client.get(f"/api/v1/notes/{doomed_id}").status_code != 200, "the note is stored"
    page.reload()
    expect(cards.filter(has_text=kept)).to_have_count(1)
    expect(cards.filter(has_text=doomed)).to_have_count(0)


# ---------------------------------------------------------------- the Access panel


@pytest.fixture
def sharer(make_user, admin) -> Actor:
    """A fresh account in a group allowed to share notes, with public sharing too."""
    owner = make_user()
    make_group(admin, [owner], {"sharing": {"notes": True, "public_notes": True}})
    return owner


def _open_access_panel(page: Page) -> Locator:
    page.get_by_role("main").get_by_role("button", name="Access").click()
    panel = page.get_by_role("dialog").filter(has_text="Access Control")
    expect(panel).to_be_visible()
    return panel


def _add_access(page: Page, panel: Locator, name: str) -> None:
    panel.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has=page.get_by_role("button", name="Add"))
    picker.get_by_placeholder("Search").fill(name)
    picker.get_by_role("button", name=name).last.click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()


def _stored_grants(owner: Actor, note_id: str) -> set[tuple[str, str, str]]:
    grants = _stored_note(owner, note_id).get("access_grants") or []
    return {(item["principal_type"], item["principal_id"], item["permission"]) for item in grants}


def _expect_read_only(page: Page, note_id: str, text: str) -> None:
    editor = _open_note(page, note_id, text)
    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_be_visible()
    expect(editor).to_have_attribute("contenteditable", "false")


def _expect_editable(page: Page, note_id: str, text: str) -> None:
    editor = _open_note(page, note_id, text)
    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_have_count(0)
    expect(editor).to_have_attribute("contenteditable", "true")


def test_adding_a_person_in_the_access_panel_shares_the_note_to_read(page_for, sharer, make_user):
    reader = make_user()
    note_id = _create_note(sharer, _unique("Minutes"), "we agreed on Friday")
    page = page_for(sharer)
    _open_note(page, note_id, "we agreed on Friday")

    panel = _open_access_panel(page)
    _add_access(page, panel, reader.name)

    expect(panel.get_by_role("combobox", name="Access level")).to_have_value("read")
    assert _stored_grants(sharer, note_id) == {("user", reader.id, "read")}
    _expect_read_only(page_for(reader), note_id, "we agreed on Friday")


def test_raising_a_person_to_write_lets_them_edit(page_for, sharer, make_user):
    writer = make_user()
    note_id = _create_note(sharer, _unique("Plan"), "draft plan")
    page = page_for(sharer)
    _open_note(page, note_id, "draft plan")
    panel = _open_access_panel(page)
    _add_access(page, panel, writer.name)

    panel.get_by_role("combobox", name="Access level").select_option("write")

    expected = {("user", writer.id, "read"), ("user", writer.id, "write")}
    _eventually(lambda: _stored_grants(sharer, note_id), expected)
    _expect_editable(page_for(writer), note_id, "draft plan")


def test_a_group_added_for_writing_lets_its_members_edit(page_for, sharer, make_user, admin):
    member = make_user()
    # an account may share with the groups it belongs to
    group_id = make_group(admin, [sharer, member])
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    note_id = _create_note(sharer, _unique("Rota"), "who cooks when")
    page = page_for(sharer)
    _open_note(page, note_id, "who cooks when")
    panel = _open_access_panel(page)
    _add_access(page, panel, group_name)
    expect(panel.get_by_text(group_name)).to_be_visible()

    panel.get_by_role("combobox", name="Access level").select_option("write")

    expected = {("group", group_id, "read"), ("group", group_id, "write")}
    _eventually(lambda: _stored_grants(sharer, note_id), expected)
    _expect_editable(page_for(member), note_id, "who cooks when")


def test_removing_a_person_from_the_access_list_shuts_them_out(page_for, sharer, make_user):
    former = make_user()
    note_id = _create_note(sharer, _unique("Budget"), "numbers for the quarter")
    page = page_for(sharer)
    _open_note(page, note_id, "numbers for the quarter")
    panel = _open_access_panel(page)
    _add_access(page, panel, former.name)
    assert _stored_grants(sharer, note_id) == {("user", former.id, "read")}

    row = panel.get_by_role("combobox", name="Access level").locator("xpath=..")
    row.get_by_role("button").click()

    expect(panel.get_by_text("No access grants. Private to you.")).to_be_visible()
    _eventually(lambda: _stored_grants(sharer, note_id), set())
    shut_out = page_for(former)
    shut_out.goto(f"/notes/{note_id}")
    expect(shut_out).to_have_url(f"{former.base_url}/")
    expect(chat_input(shut_out)).to_be_visible()


def test_making_a_note_public_lets_any_account_read_it(page_for, sharer, make_user):
    anyone = make_user()
    note_id = _create_note(sharer, _unique("Notice"), "office closed on Monday")
    page = page_for(sharer)
    _open_note(page, note_id, "office closed on Monday")
    panel = _open_access_panel(page)

    try:
        panel.get_by_role("combobox").first.select_option("public")

        expect(page.get_by_text("Saved", exact=True)).to_be_visible()
        expect(panel.get_by_text("Accessible to all users")).to_be_visible()
        assert ("user", "*", "read") in _stored_grants(sharer, note_id)
        _expect_read_only(page_for(anyone), note_id, "office closed on Monday")
    finally:
        # a public note shows in every later account's notes list
        with sharer.client() as client:
            client.delete(f"/api/v1/notes/{note_id}/delete").raise_for_status()


def test_without_the_sharing_permission_the_access_panel_has_no_list(page_for, make_user):
    owner = make_user()
    note_id = _create_note(owner, _unique("Private"), "just for me")
    page = page_for(owner)
    _open_note(page, note_id, "just for me")

    panel = _open_access_panel(page)

    private = panel.get_by_text("Only select users and groups with permission can access")
    expect(private).to_be_visible()
    expect(panel.get_by_role("button", name="Add Access")).to_have_count(0)
    expect(panel.get_by_role("combobox").first.get_by_role("option")).to_have_text(["Private"])
