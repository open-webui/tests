"""Journey: the Notes page, where a user imports notes, switches layouts and filters the list.

Import txt/md in the Create menu makes one note per file, titled after the file name and holding
its text, and a Markdown file dropped onto the list does the same; any other kind of file is
refused. The list switches between List and Grid, and the choice survives a reload; a note opens
from either layout. The All, Created by you and Shared with you filters, with Write or Read Only
beside them, keep the notes the account owns or was given at that level, and the choice survives
a reload too. The Title column sorts the list both ways, holding Shift turns every note's menu
into a delete button that deletes at once, and Create a new note in the global search opens a
note holding what was typed there, once the confirm dialog it shows is accepted (since 0ffd86967).

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the import storing notes without their text (both import tests), the file type check
dropped, the grid card showing no text, the grid title and the list row opening the Notes page,
the list asked for with no view option or access level, the Title column not sorting, the Shift
delete button asking for a confirmation and the search action dropping the typed text (the last
checked again on dev 22102e4a2).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTE_URL = re.compile(r"/notes/[0-9a-f-]+$")

DROP_FILE = """([selector, name, type, text]) => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([text], name, { type }));
    const target = document.querySelector(selector);
    target.dispatchEvent(new DragEvent('dragover', { dataTransfer: transfer, bubbles: true }));
    target.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true }));
}"""


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


def _create_note(owner: Actor, title: str, markdown: str, grants: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": title,
                "data": {"content": {"md": markdown}},
                "access_grants": grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_notes(owner: Actor) -> dict[str, str]:
    """`{title: markdown}` of the notes the account owns."""
    with owner.client() as client:
        notes = client.get("/api/v1/notes/").json()
        stored = {}
        for note in notes:
            full = client.get(f"/api/v1/notes/{note['id']}").json()
            stored[full["title"]] = full["data"]["content"]["md"]
    return stored


def _list_row(page: Page, title: str) -> Locator:
    return page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)


def _grid_title(page: Page, title: str) -> Locator:
    return page.get_by_role("main").get_by_role("link", name=title, exact=True)


def _choose(page: Page, current: str, option: str) -> None:
    """Pick `option` in the Notes page dropdown that currently reads `current`."""
    page.get_by_role("main").get_by_role("button", name=current, exact=True).click()
    page.get_by_role("button", name=option, exact=True).click()


def _open_notes_page(page: Page, title: str) -> None:
    page.goto("/notes")
    expect(page.get_by_role("main").get_by_text(title).first).to_be_visible()


# ---------------------------------------------------------------- import


def test_importing_txt_and_md_files_makes_a_note_of_each(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes")
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()

    page.get_by_role("main").get_by_role("button", name="Open create menu").first.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Import txt/md").click()
    chooser.value.set_files(
        [
            {"name": "Packing.txt", "mimeType": "text/plain", "buffer": b"tent and stove"},
            {"name": "Route.md", "mimeType": "text/markdown", "buffer": b"# Day one\n\n- ridge"},
        ]
    )

    expect(page.get_by_text("Imported notes successfully")).to_be_visible()
    expect(_list_row(page, "Packing")).to_have_count(1)
    expect(_list_row(page, "Route")).to_have_count(1)
    assert _stored_notes(author) == {"Packing": "tent and stove", "Route": "# Day one\n\n- ridge"}

    _list_row(page, "Route").click()
    editor = page.get_by_role("main").get_by_label("Write something...")
    expect(editor.get_by_role("heading", name="Day one")).to_be_visible()
    expect(editor.get_by_role("listitem")).to_contain_text("ridge")


def test_a_markdown_file_dropped_on_the_list_becomes_a_note(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes")
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()

    page.evaluate(DROP_FILE, ["#notes-container", "Recipes.md", "text/markdown", "leek soup"])

    expect(_list_row(page, "Recipes")).to_have_count(1)
    assert _stored_notes(author) == {"Recipes": "leek soup"}


def test_a_file_that_is_not_text_is_refused(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes")
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()

    page.evaluate(DROP_FILE, ["#notes-container", "Scan.pdf", "application/pdf", "%PDF-1.4"])

    expect(page.get_by_text("Only txt and md files are allowed")).to_be_visible()
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()
    assert _stored_notes(author) == {}


# ---------------------------------------------------------------- layouts


def test_the_grid_layout_shows_the_notes_text_and_stays_after_a_reload(page_for, make_user):
    author = make_user()
    title = _unique("Birdwatch")
    _create_note(author, title, "two herons by the weir")
    page = page_for(author)
    _open_notes_page(page, title)
    expect(page.get_by_role("main").get_by_text("two herons by the weir")).to_have_count(0)

    _choose(page, "List", "Grid")

    expect(_grid_title(page, title)).to_be_visible()
    expect(page.get_by_role("main").get_by_text("two herons by the weir")).to_be_visible()
    expect(page.get_by_role("main").get_by_text(author.name)).to_be_visible()
    page.reload()
    expect(page.get_by_role("main").get_by_text("two herons by the weir")).to_be_visible()

    _choose(page, "Grid", "List")

    expect(_list_row(page, title)).to_have_count(1)
    expect(page.get_by_role("main").get_by_text("two herons by the weir")).to_have_count(0)


@pytest.mark.parametrize("layout", ["List", "Grid"])
def test_a_note_opens_from_the_list_in_either_layout(page_for, make_user, layout):
    author = make_user()
    title = _unique("Allotment")
    note_id = _create_note(author, title, "sow the broad beans")
    page = page_for(author)
    _open_notes_page(page, title)
    if layout == "Grid":
        _choose(page, "List", "Grid")
        _grid_title(page, title).click()
    else:
        _list_row(page, title).click()

    expect(page).to_have_url(f"{author.base_url}/notes/{note_id}")
    editor_page = page.get_by_role("main")
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(title)
    expect(editor_page.get_by_label("Write something...")).to_have_text("sow the broad beans")


# ---------------------------------------------------------------- filters


@pytest.fixture
def shared_notes(make_user) -> tuple[Actor, dict[str, str]]:
    """An account with a note of its own, one shared with it to edit and one to read."""
    viewer, writer_owner, reader_owner = make_user(), make_user(), make_user()
    titles = {
        "own": _unique("My own"),
        "write": _unique("Shared to edit"),
        "read": _unique("Shared to read"),
    }
    _create_note(viewer, titles["own"], "mine")
    edit_grants = [grant("user", viewer.id, "read"), grant("user", viewer.id, "write")]
    _create_note(writer_owner, titles["write"], "ours", edit_grants)
    _create_note(reader_owner, titles["read"], "theirs", [grant("user", viewer.id, "read")])
    return viewer, titles


def _expect_listed(page: Page, titles: dict[str, str], *listed: str) -> None:
    for kind, title in titles.items():
        expect(_list_row(page, title)).to_have_count(1 if kind in listed else 0)


def test_all_lists_own_and_editable_notes_and_read_only_the_rest(page_for, shared_notes):
    viewer, titles = shared_notes
    page = page_for(viewer)
    _open_notes_page(page, titles["own"])

    _expect_listed(page, titles, "own", "write")
    _choose(page, "Write", "Read Only")
    _expect_listed(page, titles, "read")


def test_created_by_you_keeps_only_the_accounts_own_notes(page_for, shared_notes):
    viewer, titles = shared_notes
    page = page_for(viewer)
    _open_notes_page(page, titles["own"])

    _choose(page, "All", "Created by you")

    _expect_listed(page, titles, "own")
    expect(page.get_by_role("main").get_by_role("button", name="Write", exact=True)).to_have_count(
        0
    )


def test_shared_with_you_splits_notes_by_access_and_survives_a_reload(page_for, shared_notes):
    viewer, titles = shared_notes
    page = page_for(viewer)
    _open_notes_page(page, titles["own"])

    _choose(page, "All", "Shared with you")
    _expect_listed(page, titles, "write")
    _choose(page, "Write", "Read Only")
    _expect_listed(page, titles, "read")

    page.reload()
    expect(page.get_by_role("main").get_by_role("button", name="Shared with you")).to_be_visible()
    _expect_listed(page, titles, "write")


# ---------------------------------------------------------------- sorting and quick delete


def test_the_title_column_sorts_the_list_both_ways(page_for, make_user):
    author = make_user()
    tag = uuid.uuid4().hex[:6]
    for fruit in ("Mango", "Zucchini", "Apple"):
        _create_note(author, f"{fruit} {tag}", fruit.lower())
    page = page_for(author)
    _open_notes_page(page, f"Apple {tag}")
    rows = page.get_by_role("main").get_by_role("button", name="Open note")
    newest_first = [f"Apple {tag}", f"Zucchini {tag}", f"Mango {tag}"]
    expect(rows).to_have_text([re.compile(title) for title in newest_first])

    page.get_by_role("main").get_by_role("button", name="Title", exact=True).click()
    ascending = [f"Apple {tag}", f"Mango {tag}", f"Zucchini {tag}"]
    expect(rows).to_have_text([re.compile(title) for title in ascending])

    page.get_by_role("main").get_by_role("button", name="Title", exact=True).click()
    expect(rows).to_have_text([re.compile(title) for title in reversed(ascending)])


@pytest.mark.parametrize("layout", ["List", "Grid"])
def test_holding_shift_deletes_a_note_at_once(page_for, make_user, layout):
    author = make_user()
    doomed, kept = _unique("Old draft"), _unique("Keeper")
    _create_note(author, kept, "keep this")
    # the newest note is listed first
    _create_note(author, doomed, "scrap this")
    page = page_for(author)
    _open_notes_page(page, doomed)
    if layout == "Grid":
        _choose(page, "List", "Grid")
    main = page.get_by_role("main")
    expect(main.get_by_label("Note Menu")).to_have_count(2)

    page.keyboard.down("Shift")
    expect(main.get_by_label("Note Menu")).to_have_count(0)
    main.get_by_role("button", name="Delete").first.click()
    page.keyboard.up("Shift")

    expect(main.get_by_text(doomed)).to_have_count(0)
    expect(main.get_by_label("Note Menu")).to_have_count(1)
    assert list(_stored_notes(author)) == [kept]


# ---------------------------------------------------------------- global search


def test_create_a_new_note_in_the_search_opens_a_note_with_the_typed_text(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes")
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()

    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name="Search", exact=True).click()
    page.get_by_placeholder("Search").last.fill("call the plumber")
    page.get_by_role("button", name="Create a new note").click()
    confirm = page.get_by_role("dialog", name="Create a new note")
    expect(confirm).to_contain_text("call the plumber")
    confirm.get_by_role("button", name="Confirm").click()

    expect(page).to_have_url(NOTE_URL)
    editor = page.get_by_role("main").get_by_label("Write something...")
    expect(editor).to_have_text("call the plumber")
    stored = _stored_notes(author)
    assert list(stored.values()) == ["call the plumber"], stored
