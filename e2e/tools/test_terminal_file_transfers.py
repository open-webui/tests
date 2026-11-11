"""Journey: a person brings files and folders into a terminal through the file browser and takes
them out again.

A real Open Terminal runs with a folder of the test's own in its home, open in the chat's file
browser. Upload Folder in the Actions menu takes a folder from the person's computer and makes it
on the terminal with every subfolder and file in place. Files and a folder dragged from the
computer onto the browser show "Drop to upload" while they hover and land in the open folder the
same way (docs: File Browser, "Uploading files"). A file whose name holds a backslash is refused
with the reason and nothing is written. Download in a file's More menu saves that file, Download
in the Actions menu and in the right-click menu saves the open folder as one ZIP named after it,
and Download in the selection bar saves the selected entries as one ZIP, or the file itself when
only one file is selected.

Discriminates: passes on the dev c15f28649 build, three runs. Each test fails on a frontend build
of it (A or B) at the step named. On A: a chosen folder's files uploaded without their folders
(upload folder test), a dropped folder not walked (drop test), an upload's name not checked
(backslash test), a saved file called "file" (file download and selection tests) and a ZIP
always called download.zip (folder test). On B: no "Drop to upload" while files hover (drop
test) and a selection's ZIP holding its first entry alone (selection test).
"""

from __future__ import annotations

import io
import uuid
import zipfile
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.file_browser import (
    choose_action,
    connected_browser,
    enter,
    expect_listed,
    folder_menu,
    open_actions,
    row,
    row_menu,
    select,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


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
def browser(quay, page_for, make_user, admin, preserve, open_terminal) -> Locator:
    """The File browser of a chat with the Open Terminal picked, showing the test's folder."""
    preserve(TERMINAL_SERVERS_CONFIG)
    region = connected_browser(page_for, admin, make_user(), open_terminal)
    enter(region, quay.name, "nets.txt")
    return region


@pytest.fixture
def cargo(tmp_path) -> Path:
    """A folder on the person's computer with a file, a subfolder with a file and an empty one."""
    folder = tmp_path / "cargo"
    (folder / "crates" / "fragile").mkdir(parents=True)
    (folder / "empty-hold").mkdir()
    (folder / "manifest.txt").write_text("three crates\n")
    (folder / "crates" / "tea.txt").write_text("assam\n")
    (folder / "crates" / "fragile" / "glass.txt").write_text("this side up\n")
    return folder


def _tree(folder: Path) -> dict[str, str | None]:
    """Every entry under `folder` by its path from there: a file's text, None for a folder."""
    return {
        str(entry.relative_to(folder)): entry.read_text() if entry.is_file() else None
        for entry in sorted(folder.rglob("*"))
    }


def test_upload_folder_makes_the_folder_with_its_subfolders_on_the_terminal(browser, quay, cargo):
    menu = open_actions(browser)
    with browser.page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload Folder", exact=True).click()
    chooser.value.set_files(cargo)

    expect(row(browser, "cargo")).to_be_visible()
    uploaded = quay / "cargo"
    # a folder picker hands over the files alone, so the folder without one is not made
    expected = {path: text for path, text in _tree(cargo).items() if path != "empty-hold"}
    assert wait_until(lambda: _tree(uploaded) == expected, 10), (
        f"the terminal holds {_tree(uploaded)} of the uploaded folder {expected}"
    )

    browser.get_by_text("cargo", exact=True).click()
    expect_listed(browser, "crates/", "manifest.txt")


def _drag_from_the_computer(page: Page, onto: Locator, *paths: Path) -> dict:
    """Bring files or folders of this computer over `onto` and hold them there; returns what
    `_drop` needs."""
    box = onto.bounding_box()
    assert box, "the place to drop on is not on the page"
    session = page.context.new_cdp_session(page)
    held = {
        "x": box["x"] + box["width"] / 2,
        "y": box["y"] + box["height"] / 2,
        "data": {"items": [], "files": [str(path) for path in paths], "dragOperationsMask": 1},
    }
    # the page offers no way to start a drag of the computer's own files
    session.send("Input.dispatchDragEvent", {"type": "dragEnter", **held})
    session.send("Input.dispatchDragEvent", {"type": "dragOver", **held})
    return {"session": session, **held}


def _drop(held: dict) -> None:
    session = held.pop("session")
    session.send("Input.dispatchDragEvent", {"type": "drop", **held})
    session.detach()


def test_files_and_a_folder_dropped_from_the_computer_land_in_the_open_folder(
    browser, quay, cargo, tmp_path
):
    lone = tmp_path / "tide-table.txt"
    lone.write_text("high at six\n")

    held = _drag_from_the_computer(browser.page, browser, cargo, lone)
    expect(browser.get_by_role("status").filter(has_text="Drop to upload")).to_be_visible()
    _drop(held)

    expect(row(browser, "tide-table.txt")).to_be_visible()
    expect(row(browser, "cargo")).to_be_visible()
    expect(browser.get_by_text("Drop to upload")).to_have_count(0)
    assert wait_until(lambda: _tree(quay / "cargo") == _tree(cargo), 10), (
        f"the terminal holds {_tree(quay / 'cargo')} of the dropped folder {_tree(cargo)}"
    )
    assert (quay / "tide-table.txt").read_text() == "high at six\n"


def _choose_upload(browser: Locator, *files: dict) -> None:
    menu = open_actions(browser)
    with browser.page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload", exact=True).click()
    chooser.value.set_files(list(files))


def test_a_file_with_a_backslash_in_its_name_is_refused_and_nothing_is_written(browser, quay):
    before = _tree(quay)
    uploads: list[str] = []
    browser.page.on(
        "request",
        lambda request: uploads.append(request.url) if "/files/upload" in request.url else None,
    )

    _choose_upload(
        browser, {"name": "tide\\table.txt", "mimeType": "text/plain", "buffer": b"high at six\n"}
    )

    expect(
        browser.page.get_by_text("Upload failed: Invalid upload path: tide\\table.txt")
    ).to_be_visible()
    assert not uploads, f"the refused file was sent to the terminal: {uploads}"

    _choose_upload(
        browser, {"name": "tide-table.txt", "mimeType": "text/plain", "buffer": b"high at six\n"}
    )
    expect(row(browser, "tide-table.txt")).to_be_visible()
    assert len(uploads) == 1, f"a file with a plain name is uploaded once: {uploads}"
    assert _tree(quay) == {**before, "tide-table.txt": "high at six\n"}, (
        "only the file with the plain name should be on the terminal"
    )


def _saved(download) -> bytes:
    return Path(download.path()).read_bytes()


def _zipped(download) -> dict[str, str]:
    """The files of a downloaded ZIP by their path in it."""
    with zipfile.ZipFile(io.BytesIO(_saved(download))) as archive:
        return {name: archive.read(name).decode() for name in archive.namelist() if name[-1] != "/"}


def test_download_in_a_files_menu_saves_the_file(browser):
    menu = row_menu(browser, "nets.txt")
    with browser.page.expect_download() as download:
        menu.get_by_role("menuitem", name="Download", exact=True).click()

    assert download.value.suggested_filename == "nets.txt"
    assert _saved(download.value) == b"mended\n", "the saved file is not the terminal's file"


def test_download_of_the_open_folder_saves_it_as_one_zip_named_after_it(browser, quay):
    expected = {
        f"{quay.name}/nets.txt": "mended\n",
        f"{quay.name}/oars.txt": "four\n",
        f"{quay.name}/sheds/ledger.txt": "two ropes\n",
    }

    with browser.page.expect_download() as from_actions:
        choose_action(browser, "Download")
    assert from_actions.value.suggested_filename == f"{quay.name}.zip"
    assert _zipped(from_actions.value) == expected, "the ZIP from the Actions menu"

    menu = folder_menu(browser)
    with browser.page.expect_download() as from_right_click:
        menu.get_by_role("button", name="Download", exact=True).click()
    assert from_right_click.value.suggested_filename == f"{quay.name}.zip"
    assert _zipped(from_right_click.value) == expected, "the ZIP from the right-click menu"


def test_download_of_a_selection_saves_one_zip_or_the_one_file_selected(browser):
    select(browser, "oars.txt")
    expect(browser.get_by_text("1 selected", exact=True)).to_be_visible()
    bar_download = browser.get_by_role("button", name="Download", exact=True)
    with browser.page.expect_download() as one_file:
        bar_download.click()
    assert one_file.value.suggested_filename == "oars.txt"
    assert _saved(one_file.value) == b"four\n", "one selected file is saved as itself"

    select(browser, "sheds")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()
    with browser.page.expect_download() as selection:
        bar_download.click()
    assert selection.value.suggested_filename == "download.zip"
    assert _zipped(selection.value) == {
        "oars.txt": "four\n",
        "sheds/ledger.txt": "two ropes\n",
    }, "the ZIP holds the selected file and folder and nothing else"
