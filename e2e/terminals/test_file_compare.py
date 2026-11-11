"""Journey: a person compares two files of a terminal from its file browser.

A real Open Terminal runs with a folder of the test's own. Select for Compare in a file's More
menu names the file above the list until it is cleared, and offers Compare with Selected on the
other files only. Two selected files are compared from the selection bar, which offers it for
exactly two. Text files are read in full by the page: one past 2 MiB, one that is not UTF-8
and a picture are each refused with the reason beside the file's path, and a file deleted
since it was picked says it was not found, with a Retry that compares once the file is back.
Word documents are compared by the terminal: the counts of added and removed lines, the lines
themselves, a "Text only" note, Swap turning the two around, and "No text differences" for two
documents with the same text. The Split view and Hide whitespace are still in place the next
time documents are compared. A terminal from before the comparison (scripted: it answers the
request with 404) says to update Open Terminal, and Retry shows the comparison once it can.

Discriminates: all pass on dev c15f28649 with Open Terminal 0.17.7. On one frontend build of it
with these mutations each named test fails: Clear leaving the picked file in place and Compare
with Selected offered on the picked file itself (picking); the selection bar's Compare offered
for any number of files (selection); the 2 MiB limit, the UTF-8 check and the picture check
removed (refusals); every failed read reported as "Could not read file." (deleted file);
Swap doing nothing (documents); the view choice not stored (kept choices); a terminal's 404
for the comparison shown as it came (old terminal).
"""

from __future__ import annotations

import json
import shutil
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import sample_documents as samples
from harness.listener import json_answer
from harness.terminal_files import HOME, serving_files
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.file_views import files_of, rest_elsewhere, row_menu, toggle_selected

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MEBIBYTE = 1024 * 1024
LETTER = ["Dear harbour master,", "the cider boat lands on Friday.", "Yours, Mara"]
REVISED = [
    "Dear harbour master,",
    "the cider boat lands on Monday.",
    "Bring the ledger.",
    "Yours, Mara",
]
RESPACED = [LETTER[0], "the cider  boat lands  on Friday.", LETTER[2]]


@pytest.fixture
def folder(open_terminal):
    """A folder of this test's own in the terminal's home, which the whole session shares."""
    path = open_terminal.home / f"compare-{uuid.uuid4().hex[:6]}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def browse(folder, make_user, page_for, admin, preserve, open_terminal):
    """`browse()` opens the folder in a person's file browser, once its files are written."""

    def open_folder() -> Locator:
        preserve(TERMINAL_SERVERS_CONFIG)
        return files_of(page_for, admin, make_user(), open_terminal, folder.name)

    return open_folder


def _pick(browser: Locator, name: str) -> None:
    row_menu(browser, name).get_by_role("menuitem", name="Select for Compare").click()
    expect(browser.get_by_text(f"Comparing with {name}", exact=True)).to_be_visible()
    rest_elsewhere(browser.page)


def _compare(browser: Locator, picked: str, other: str) -> Locator:
    """Pick one file, compare the other with it; returns the comparison's dialog."""
    _pick(browser, picked)
    row_menu(browser, other).get_by_role("menuitem", name="Compare with Selected").click()
    comparison = browser.page.get_by_role("dialog", name="Compare files")
    expect(comparison).to_be_visible()
    return comparison


def _close(page: Page, comparison: Locator) -> None:
    comparison.get_by_role("button", name="Close", exact=True).click()
    expect(comparison).to_have_count(0)
    rest_elsewhere(page)


def test_a_file_picked_for_comparing_is_named_above_the_list_until_it_is_cleared(folder, browse):
    (folder / "draft.txt").write_text("Dear harbour master,\n")
    (folder / "final.txt").write_text("Dear harbour mistress,\n")
    browser = browse()
    page = browser.page

    # nothing is picked yet, so there is nothing to compare a file with
    menu = row_menu(browser, "final.txt")
    expect(menu.get_by_role("menuitem", name="Select for Compare")).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Compare with Selected")).to_have_count(0)
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)

    _pick(browser, "draft.txt")
    expect(
        row_menu(browser, "draft.txt").get_by_role("menuitem", name="Compare with Selected")
    ).to_be_disabled()
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    expect(
        row_menu(browser, "final.txt").get_by_role("menuitem", name="Compare with Selected")
    ).to_be_enabled()
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)

    browser.get_by_role("button", name="Clear comparison selection").click()

    expect(browser.get_by_text("Comparing with draft.txt", exact=True)).to_have_count(0)
    menu = row_menu(browser, "final.txt")
    expect(menu.get_by_role("menuitem", name="Select for Compare")).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Compare with Selected")).to_have_count(0)


def test_two_selected_files_are_compared_from_the_selection_bar(folder, browse):
    (folder / "draft.txt").write_text("Dear harbour master,\n")
    (folder / "final.txt").write_text("Dear harbour mistress,\n")
    (folder / "notes.txt").write_text("ask about the quay\n")
    browser = browse()
    compare = browser.get_by_role("button", name="Compare", exact=True)

    toggle_selected(browser, "draft.txt")
    expect(browser.get_by_text("1 selected", exact=True)).to_be_visible()
    expect(compare).to_be_disabled()
    toggle_selected(browser, "final.txt")
    toggle_selected(browser, "notes.txt")
    expect(browser.get_by_text("3 selected", exact=True)).to_be_visible()
    expect(compare).to_be_disabled()
    toggle_selected(browser, "notes.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()

    compare.click()

    comparison = browser.page.get_by_role("dialog", name="Compare files")
    panes = comparison.get_by_role("textbox")
    expect(panes).to_have_count(2)
    expect(panes.first).to_have_accessible_name("draft.txt")
    expect(panes.first).to_have_text("Dear harbour master,")
    expect(panes.last).to_have_accessible_name("final.txt")
    expect(panes.last).to_have_text("Dear harbour mistress,")


def test_a_file_too_large_not_utf8_or_a_picture_is_refused_with_the_reason(folder, browse):
    (folder / "draft.txt").write_text("Dear harbour master,\n")
    (folder / "long.log").write_text("pressed 40 crates\n" * (2 * MEBIBYTE // 18 + 1))
    (folder / "old-menu.txt").write_bytes("café crème\n".encode("latin-1"))
    (folder / "label.png").write_bytes(samples.png())
    browser = browse()
    page = browser.page
    refusals = {
        "long.log": "File exceeds the 2 MiB comparison limit.",
        "old-menu.txt": "Only UTF-8 text files are supported for source comparison.",
        "label.png": "This file is not supported for source comparison.",
    }

    for name, reason in refusals.items():
        comparison = _compare(browser, "draft.txt", name)
        expect(comparison.get_by_role("alert")).to_have_text(f"{folder}/{name}: {reason}")
        expect(comparison.get_by_role("textbox")).to_have_count(0)
        _close(page, comparison)


def test_a_file_deleted_since_it_was_picked_says_not_found_and_retry_compares_once_it_is_back(
    folder, browse
):
    (folder / "draft.txt").write_text("Dear harbour master,\n")
    (folder / "final.txt").write_text("Dear harbour mistress,\n")
    browser = browse()
    _pick(browser, "draft.txt")
    (folder / "draft.txt").unlink()

    row_menu(browser, "final.txt").get_by_role("menuitem", name="Compare with Selected").click()

    comparison = browser.page.get_by_role("dialog", name="Compare files")
    expect(comparison.get_by_role("alert")).to_have_text(f"{folder}/draft.txt: File not found.")
    (folder / "draft.txt").write_text("Dear harbour master,\n")
    comparison.get_by_role("button", name="Retry", exact=True).click()
    panes = comparison.get_by_role("textbox")
    expect(panes).to_have_count(2)
    expect(panes.first).to_have_text("Dear harbour master,")
    expect(panes.last).to_have_text("Dear harbour mistress,")
    expect(comparison.get_by_role("alert")).to_have_count(0)


def _top(line: Locator) -> float:
    box = line.bounding_box()
    assert box, "the compared line has no place on the page"
    return box["y"]


def test_two_word_documents_are_compared_by_their_text(folder, browse):
    (folder / "letter.docx").write_bytes(samples.docx(LETTER))
    (folder / "letter-revised.docx").write_bytes(samples.docx(REVISED))
    (folder / "letter-copy.docx").write_bytes(samples.docx(LETTER))
    browser = browse()
    page = browser.page

    comparison = _compare(browser, "letter.docx", "letter-revised.docx")

    expect(comparison.get_by_label("2 added lines")).to_have_text("+2")
    expect(comparison.get_by_label("1 removed lines")).to_have_text("−1")
    expect(comparison.get_by_role("button", name="Text only")).to_be_visible()
    differences = comparison.get_by_role("region", name="File differences")
    friday = differences.get_by_text("the cider boat lands on Friday.")
    monday = differences.get_by_text("the cider boat lands on Monday.")
    expect(friday).to_be_visible()
    expect(differences.get_by_text("Bring the ledger.")).to_be_visible()
    # one column: the removed line stands above the line that replaced it
    assert _top(friday) < _top(monday), "the removed and the added line are not one above the other"
    # no editor: the documents' text is only shown
    expect(comparison.get_by_role("textbox")).to_have_count(0)

    comparison.get_by_role("button", name="Swap", exact=True).click()
    expect(comparison.get_by_label("1 added lines")).to_have_text("+1")
    expect(comparison.get_by_label("2 removed lines")).to_have_text("−2")
    _close(page, comparison)

    comparison = _compare(browser, "letter.docx", "letter-copy.docx")
    expect(comparison.get_by_role("status")).to_have_text("No text differences")
    expect(comparison.get_by_label("0 added lines")).to_have_text("+0")


def test_the_split_view_and_hide_whitespace_are_kept_for_the_next_comparison(folder, browse):
    (folder / "letter.docx").write_bytes(samples.docx(LETTER))
    (folder / "letter-revised.docx").write_bytes(samples.docx(REVISED))
    (folder / "letter-respaced.docx").write_bytes(samples.docx(RESPACED))
    browser = browse()
    page = browser.page
    comparison = _compare(browser, "letter.docx", "letter-respaced.docx")
    expect(comparison.get_by_label("1 added lines")).to_be_visible()

    comparison.get_by_role("button", name="Diff settings").click()
    # the settings open outside the dialog, where their buttons have no role to find them by
    page.get_by_text("Hide whitespace", exact=True).click()
    expect(comparison.get_by_role("status")).to_have_text("No text differences")
    page.get_by_text("Split", exact=True).click()
    # picking a view closes the settings
    expect(page.get_by_text("Hide whitespace", exact=True)).to_have_count(0)
    _close(page, comparison)

    comparison = _compare(browser, "letter.docx", "letter-revised.docx")
    differences = comparison.get_by_role("region", name="File differences")
    friday = differences.get_by_text("the cider boat lands on Friday.")
    monday = differences.get_by_text("the cider boat lands on Monday.")
    expect(monday).to_be_visible()
    # two columns: the removed line and the line that replaced it stand side by side
    assert _top(friday) == _top(monday), "the comparison did not open in the Split view again"
    _close(page, comparison)

    # and the whitespace choice: the respaced letter still counts as the same text
    comparison = _compare(browser, "letter.docx", "letter-respaced.docx")
    expect(comparison.get_by_role("status")).to_have_text("No text differences")


UPDATE_IT = (
    "File comparison is not available on this terminal. Update Open Terminal to use Compare."
)


def _line(kind: str, text: str, old: int | None, new: int | None) -> dict:
    return {
        "type": kind,
        "oldNumber": old,
        "newNumber": new,
        "content": text,
        "segments": [{"text": text, "changed": kind != "context"}],
    }


def test_a_terminal_too_old_to_compare_documents_says_to_update_it(
    make_user, page_for, admin, preserve
):
    with serving_files() as terminal:
        terminal.files["letter.docx"] = samples.docx(LETTER)
        terminal.files["letter-revised.docx"] = samples.docx(REVISED)
        # an Open Terminal from before the comparison knows no such path
        terminal.route("POST", "/files/compare", json_answer({"detail": "Not Found"}, 404))
        preserve(TERMINAL_SERVERS_CONFIG)
        browser = files_of(page_for, admin, make_user(), terminal)

        comparison = _compare(browser, "letter.docx", "letter-revised.docx")

        expect(comparison.get_by_role("alert")).to_contain_text(UPDATE_IT)
        # once the terminal is updated, Retry shows what it compared
        compared = {
            "original": {"name": "letter.docx", "path": f"{HOME}/letter.docx", "notices": []},
            "revised": {
                "name": "letter-revised.docx",
                "path": f"{HOME}/letter-revised.docx",
                "notices": [],
            },
            "additions": 1,
            "deletions": 1,
            "hunks": [
                {
                    "header": "@@ -2 +2 @@",
                    "lines": [
                        _line("removed", LETTER[1], 2, None),
                        _line("added", REVISED[1], None, 2),
                    ],
                }
            ],
        }
        terminal.route("POST", "/files/compare", json_answer(compared))
        comparison.get_by_role("button", name="Retry", exact=True).click()
        differences = comparison.get_by_role("region", name="File differences")
        expect(differences.get_by_text(REVISED[1])).to_be_visible()
        expect(comparison.get_by_role("alert")).to_have_count(0)
        asked = json.loads(terminal.requests_to("/files/compare")[-1].body)
        assert asked == {
            "original": f"{HOME}/letter.docx",
            "revised": f"{HOME}/letter-revised.docx",
            "ignore_whitespace": False,
        }, f"the terminal was asked to compare {asked}"
