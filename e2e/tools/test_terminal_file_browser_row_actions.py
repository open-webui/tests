"""Journey: what a person does with the entries of a terminal's file browser, and what a
read-only folder or entry holds back.

A real Open Terminal runs with a folder of the test's own in its home. Rename in a file's More
menu marks the name up to its ending, Esc leaves the name alone, and a name another file already
has shows the terminal's refusal. Delete asks first and Cancel keeps the file. New Folder and New
File in a folder's More menu make them inside that folder, and Esc in the name field makes
nothing. A file dragged onto a folder in the path above the list moves there, selected files
move together, a folder dropped onto a folder inside itself stays where it is, and a file held
over a closed folder opens it so that the drop can go on a folder inside. Copy Path puts
the entry's full path on the clipboard. A click with Shift selects every row up to the one
clicked before and Esc takes the selection off again; Compare in the selection bar is only on
offer for exactly two files and opens them side by side.

A folder the terminal cannot write to is marked Read-only beside its path: New File, New Folder,
Upload and Upload Folder are switched off in the Actions menu and in the right-click menu while
Download stays, and nothing inside it can be renamed or deleted. A single read-only file is
marked Read-only beside its name, its Rename and Delete are switched off, it cannot be dragged
into a folder and Delete in the selection bar is off while it is selected (docs: File Browser,
"Read-only files and folders").

Discriminates: passes on the dev c15f28649 build, three runs. Each test fails on a frontend build
of it (A, B or C) at the step named. On A: Rename marking the whole name (rename test), Cancel
deleting (delete test), a folder's New Folder asking in the open folder (folder menu test), a
drop on the path moving nothing (path test), a drag carrying the dragged row alone (together
test), a folder allowed into itself (inside itself test), a held drag never opening the folder
(held file test), Shift selecting the clicked row alone (selection test) and an entry never read
only (read-only file test). On B: Esc in the rename field renaming (rename test), Esc in the
folder name field making the folder (folder menu test), Copy Path copying the name (copy test),
Compare on offer for more than two entries (selection test) and a selection with a read-only
file open to Delete (read-only file test). On C: a folder never read only (read-only folder
test, at the mark beside the path).
"""

from __future__ import annotations

import stat
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.chat_ui import close_menu
from utils.clipboard import copied, loopback_clipboard
from utils.file_browser import (
    connected_browser,
    crumb,
    enter,
    expect_listed,
    folder_menu,
    open_actions,
    row,
    row_menu,
    select,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

READ_ONLY = stat.S_IRUSR | stat.S_IXUSR
WRITABLE = stat.S_IRWXU


@pytest.fixture
def quay(open_terminal) -> Path:
    """A folder of the test's own in the terminal's home."""
    folder = open_terminal.home / f"quay-{uuid.uuid4().hex[:6]}"
    (folder / "sheds").mkdir(parents=True)
    (folder / "sheds" / "ledger.txt").write_text("two ropes\n")
    (folder / "nets.txt").write_text("mended\n")
    (folder / "oars.txt").write_text("four\n")
    return folder


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def browser(quay, page_for, person, admin, preserve, open_terminal) -> Locator:
    """The File browser of a chat with the Open Terminal picked, showing the test's folder."""
    preserve(TERMINAL_SERVERS_CONFIG)
    region = connected_browser(page_for, admin, person, open_terminal)
    enter(region, quay.name, "nets.txt")
    return region


def _sent(page: Page, path: str) -> list[str]:
    """The requests the page sends to the terminal's `path` from now on, as method and body."""
    seen: list[str] = []
    page.on(
        "request",
        lambda request: (
            seen.append(f"{request.method} {request.post_data or request.url}")
            if path in request.url
            else None
        ),
    )
    return seen


def test_rename_marks_the_name_up_to_its_ending_and_escape_leaves_it(browser, quay):
    moves = _sent(browser.page, "/files/move")
    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Rename", exact=True).click()
    field = browser.get_by_role("textbox", name="Rename", exact=True)
    expect(field).to_be_focused()
    expect(field).to_have_value("nets.txt")
    marked = field.evaluate("field => field.value.slice(field.selectionStart, field.selectionEnd)")
    assert marked == "nets", f"Rename marks {marked!r} of the name"

    field.fill("sails.txt")
    field.press("Escape")
    expect(field).to_have_count(0)
    expect(row(browser, "nets.txt")).to_be_visible()

    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Rename", exact=True).click()
    field.fill("oars.txt")
    field.press("Enter")
    # the terminal's own words
    expect(browser.page.get_by_text("Destination already exists")).to_be_visible()
    expect(row(browser, "nets.txt")).to_be_visible()
    assert len(moves) == 1, f"Esc should send no rename, the taken name one: {moves}"
    assert (quay / "nets.txt").read_text() == "mended\n", "the file to rename changed"
    assert (quay / "oars.txt").read_text() == "four\n", "the file with the taken name changed"
    assert not (quay / "sails.txt").exists(), "Esc renamed the file"


def test_delete_asks_first_and_cancel_keeps_the_file(browser, quay):
    deletes = _sent(browser.page, "/files/delete")
    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Delete", exact=True).click()
    question = browser.page.get_by_role("dialog")
    expect(question).to_contain_text('Are you sure you want to delete "nets.txt"?')
    question.get_by_role("button", name="Cancel", exact=True).click()
    expect(question).to_have_count(0)
    expect(row(browser, "nets.txt")).to_be_visible()
    assert (quay / "nets.txt").exists(), "Cancel deleted the file"

    row_menu(browser, "oars.txt").get_by_role("menuitem", name="Delete", exact=True).click()
    expect(question).to_contain_text('Are you sure you want to delete "oars.txt"?')
    question.get_by_role("button", name="Confirm", exact=True).click()
    expect(browser.page.get_by_text("oars.txt deleted")).to_be_visible()
    expect(row(browser, "oars.txt")).to_have_count(0)
    assert not (quay / "oars.txt").exists(), "Confirm left the file on disk"
    assert len(deletes) == 1 and "oars.txt" in deletes[0], f"only oars.txt is deleted: {deletes}"


def test_new_folder_and_new_file_in_a_folders_menu_are_made_inside_it(browser, quay):
    row_menu(browser, "sheds").get_by_role("menuitem", name="New Folder", exact=True).click()
    # the browser opens the folder and asks for the name there
    expect(crumb(browser, "sheds")).to_be_visible()
    name = browser.get_by_placeholder("Folder name")
    name.fill("loft")
    name.press("Escape")
    expect(name).to_have_count(0)
    expect_listed(browser, "ledger.txt")

    crumb(browser, quay.name).click()
    row_menu(browser, "sheds").get_by_role("menuitem", name="New Folder", exact=True).click()
    name.fill("cellar")
    name.press("Enter")
    expect(browser.page.get_by_text("Folder created")).to_be_visible()
    expect_listed(browser, "cellar/", "ledger.txt")

    crumb(browser, quay.name).click()
    row_menu(browser, "sheds").get_by_role("menuitem", name="New File", exact=True).click()
    browser.get_by_placeholder("File name").fill("stock.txt")
    browser.get_by_placeholder("File name").press("Enter")
    expect(browser.page.get_by_text("File created")).to_be_visible()
    expect_listed(browser, "cellar/", "ledger.txt", "stock.txt")

    assert (quay / "sheds" / "cellar").is_dir(), "the new folder is not inside the folder"
    assert (quay / "sheds" / "stock.txt").is_file(), "the new file is not inside the folder"
    assert not (quay / "sheds" / "loft").exists(), "Esc in the name field made the folder"
    assert sorted(entry.name for entry in quay.iterdir()) == ["nets.txt", "oars.txt", "sheds"], (
        "something was made beside the folder"
    )


def test_a_file_dragged_onto_a_folder_in_the_path_moves_there(browser, quay):
    browser.get_by_text("sheds", exact=True).click()
    expect_listed(browser, "ledger.txt")

    browser.get_by_text("ledger.txt", exact=True).drag_to(crumb(browser, quay.name))

    expect(browser.page.get_by_text("Moved ledger.txt")).to_be_visible()
    expect(browser.get_by_text("This folder is empty")).to_be_visible()
    assert wait_until((quay / "ledger.txt").exists, 10), "the file never reached the folder above"
    assert not (quay / "sheds" / "ledger.txt").exists(), "the file is still in its old folder"


def test_selected_files_dragged_onto_a_folder_move_together(browser, quay):
    (quay / "pots.txt").write_text("six\n")
    enter(browser, quay.name, "pots.txt")
    select(browser, "nets.txt")
    select(browser, "pots.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()

    browser.get_by_text("pots.txt", exact=True).drag_to(browser.get_by_text("sheds", exact=True))

    expect_listed(browser, "sheds/", "oars.txt")
    for name in ("nets.txt", "pots.txt"):
        assert wait_until((quay / "sheds" / name).exists, 10), f"{name} did not move"
        assert not (quay / name).exists(), f"{name} is still in its old folder"
    assert (quay / "oars.txt").exists(), "a file outside the selection moved"
    expect(browser.get_by_text("selected")).to_have_count(0)


def test_a_folder_dropped_onto_a_folder_inside_itself_stays_where_it_is(browser, quay):
    (quay / "sheds" / "loft").mkdir()
    moves = _sent(browser.page, "/files/move")
    enter(browser, quay.name, "nets.txt")
    row(browser, "sheds").get_by_role("button", name="Expand", exact=True).click()
    expect_listed(browser, "sheds/", "loft/", "ledger.txt", "nets.txt", "oars.txt")
    loft = browser.get_by_text("loft", exact=True)

    browser.get_by_text("sheds", exact=True).drag_to(loft)
    # the same drop with a file does move it, and is sent after the folder's
    browser.get_by_text("nets.txt", exact=True).drag_to(loft)

    expect(browser.page.get_by_text("Moved nets.txt")).to_be_visible()
    assert wait_until((quay / "sheds" / "loft" / "nets.txt").exists, 10), "the file did not move"
    assert len(moves) == 1 and "nets.txt" in moves[0], (
        f"only the file should be moved, the folder into itself never asked for: {moves}"
    )
    assert (quay / "sheds" / "ledger.txt").exists(), "the folder dropped into itself changed"


def test_a_file_held_over_a_closed_folder_opens_it_for_a_drop_on_a_folder_inside(browser, quay):
    (quay / "sheds" / "loft").mkdir()
    page = browser.page
    enter(browser, quay.name, "nets.txt")
    expect_listed(browser, "sheds/", "nets.txt", "oars.txt")
    source = browser.get_by_text("nets.txt", exact=True).bounding_box()
    sheds = browser.get_by_text("sheds", exact=True).bounding_box()
    assert source and sheds, "the file or the folder has no place on the page"

    page.mouse.move(source["x"] + 10, source["y"] + source["height"] / 2)
    page.mouse.down()
    page.mouse.move(sheds["x"] + 10, sheds["y"] + sheds["height"] / 2, steps=8)
    page.mouse.move(sheds["x"] + 14, sheds["y"] + sheds["height"] / 2, steps=2)
    # held there, the folder opens by itself
    loft = browser.get_by_text("loft", exact=True)
    expect(loft).to_be_visible()
    target = loft.bounding_box()
    assert target, "the folder inside has no place on the page"
    page.mouse.move(target["x"] + 10, target["y"] + target["height"] / 2, steps=8)
    page.mouse.up()

    moved = quay / "sheds" / "loft" / "nets.txt"
    assert wait_until(moved.exists, 10), "the held file never reached the folder inside"
    assert not (quay / "nets.txt").exists(), "the file is still in its old folder"


def test_copy_path_copies_the_full_path_of_the_entry(browser, quay, person):
    reader = loopback_clipboard(browser.page, person)

    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Copy Path", exact=True).click()

    expect(browser.page.get_by_text("Path copied")).to_be_visible()
    assert copied(reader) == str(quay / "nets.txt"), "Copy Path on a file"


def _bar_button(browser: Locator, name: str) -> Locator:
    return browser.get_by_role("button", name=name, exact=True)


def test_shift_selects_up_to_the_row_clicked_and_compare_takes_exactly_two_files(browser, quay):
    (quay / "pots.txt").write_text("six\n")
    enter(browser, quay.name, "pots.txt")
    expect_listed(browser, "sheds/", "nets.txt", "oars.txt", "pots.txt")

    select(browser, "nets.txt")
    expect(browser.get_by_text("1 selected", exact=True)).to_be_visible()
    expect(_bar_button(browser, "Compare"), "Compare with one file selected").to_be_disabled()
    select(browser, "pots.txt", "Shift")
    expect(browser.get_by_text("3 selected", exact=True)).to_be_visible()
    for name in ("nets.txt", "oars.txt", "pots.txt"):
        expect(browser.get_by_role("checkbox", name=f"Select {name}")).to_be_checked()
    expect(browser.get_by_role("checkbox", name="Select sheds")).to_have_count(0)
    expect(_bar_button(browser, "Compare"), "Compare with three files selected").to_be_disabled()

    browser.page.keyboard.press("Escape")
    expect(browser.get_by_text("selected")).to_have_count(0)
    expect(browser.get_by_role("checkbox")).to_have_count(0)

    select(browser, "sheds")
    select(browser, "nets.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()
    expect(_bar_button(browser, "Compare"), "Compare with a folder and a file").to_be_disabled()
    browser.get_by_role("checkbox", name="Select sheds").click()
    select(browser, "oars.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()
    _bar_button(browser, "Compare").click()

    compare = browser.page.get_by_role("dialog", name="Compare files")
    panes = compare.get_by_role("textbox")
    expect(panes).to_have_count(2)
    expect(panes.first).to_have_accessible_name("nets.txt")
    expect(panes.first).to_have_text("mended")
    expect(panes.last).to_have_accessible_name("oars.txt")
    expect(panes.last).to_have_text("four")


@pytest.fixture
def locked(quay):
    """A folder in the test's folder that the terminal cannot write to, holding one file."""
    folder = quay / "customs"
    folder.mkdir()
    (folder / "seal.txt").write_text("unbroken\n")
    folder.chmod(READ_ONLY)
    yield folder
    folder.chmod(WRITABLE)


def _disabled(menu: Locator, role: str) -> dict[str, bool]:
    entries = menu.get_by_role(role)
    return {
        entries.nth(index).inner_text().strip(): entries.nth(index).is_disabled()
        for index in range(entries.count())
    }


WRITING_ACTIONS = {"New File": True, "New Folder": True, "Upload": True, "Upload Folder": True}


def test_a_read_only_folder_offers_nothing_that_would_write_to_it(browser, quay, locked):
    enter(browser, quay.name, "customs")
    expect(browser.get_by_text("Read-only", exact=True)).to_have_count(1)
    # the mark beside the folder the terminal cannot write to
    offered = _disabled(open_actions(browser), "menuitem")
    assert offered["New File"] is False and offered["Upload"] is False, (
        f"a folder that can be written to offers its actions: {offered}"
    )
    close_menu(browser.page)

    browser.get_by_text("customs", exact=True).click()
    expect_listed(browser, "seal.txt")
    # beside the path now, and beside the file in it
    expect(browser.get_by_text("Read-only", exact=True).first).to_be_visible()

    offered = _disabled(open_actions(browser), "menuitem")
    assert offered == {
        **WRITING_ACTIONS,
        "Download": False,
        "Show Hidden Files": False,
        "Grid view": False,
        "List view": False,
    }, f"the Actions menu of a read-only folder: {offered}"
    close_menu(browser.page)

    offered = _disabled(folder_menu(browser), "button")
    assert offered == {
        "Refresh": False,
        **WRITING_ACTIONS,
        "Download": False,
        "Show Hidden Files": False,
    }, f"the right-click menu of a read-only folder: {offered}"

    # the click that opens the file's menu closes the right-click menu
    offered = _disabled(row_menu(browser, "seal.txt"), "menuitem")
    assert offered["Rename"] and offered["Delete"], (
        f"a file in a read-only folder can be renamed or deleted: {offered}"
    )
    assert not offered["Open"] and not offered["Download"] and not offered["Copy Path"], (
        f"a file in a read-only folder can still be opened, downloaded and its path copied: "
        f"{offered}"
    )


def test_a_read_only_file_cannot_be_renamed_deleted_or_dragged_into_a_folder(browser, quay):
    sealed = quay / "manifest.txt"
    sealed.write_text("sealed\n")
    sealed.chmod(stat.S_IRUSR)
    moves = _sent(browser.page, "/files/move")
    enter(browser, quay.name, "manifest.txt")
    expect(row(browser, "manifest.txt").get_by_text("Read-only", exact=True)).to_be_visible()
    expect(browser.get_by_text("Read-only", exact=True)).to_have_count(1)

    offered = _disabled(row_menu(browser, "manifest.txt"), "menuitem")
    assert offered["Rename"] and offered["Delete"], (
        f"a read-only file can be renamed or deleted: {offered}"
    )
    close_menu(browser.page)
    offered = _disabled(row_menu(browser, "nets.txt"), "menuitem")
    assert not offered["Rename"] and not offered["Delete"], (
        f"a file that can be written cannot be renamed or deleted: {offered}"
    )
    close_menu(browser.page)

    sheds = browser.get_by_text("sheds", exact=True)
    browser.get_by_text("manifest.txt", exact=True).drag_to(sheds)
    browser.get_by_text("nets.txt", exact=True).drag_to(sheds)
    expect(browser.page.get_by_text("Moved nets.txt")).to_be_visible()
    assert len(moves) == 1 and "nets.txt" in moves[0], (
        f"only the writable file should be moved: {moves}"
    )
    assert sealed.exists(), "the read-only file left its folder"

    select(browser, "oars.txt")
    expect(browser.get_by_text("1 selected", exact=True)).to_be_visible()
    expect(_bar_button(browser, "Delete")).to_be_enabled()
    select(browser, "manifest.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()
    expect(_bar_button(browser, "Delete"), "Delete with a read-only file selected").to_be_disabled()
    expect(_bar_button(browser, "Download")).to_be_enabled()
