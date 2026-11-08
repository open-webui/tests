"""Journey: a person uploads, finds, bulk deletes and edits files in the terminal's file browser.

A real Open Terminal runs with a few files in its home. In a chat the person picks the terminal
from the input's Terminal menu and opens Files. From the File browser's Actions menu they upload
a file through the file chooser and it shows in the list and lands in the terminal's home. They
open the search box and find one file by its contents and another by its name. They Ctrl-click
two rows, press Select All, delete the selection and confirm: all the selected files are gone from
the list and from the disk while an unselected one stays. A text file opens straight in the
editor; typed text and Save put the new text on disk, and in a code file Ctrl+S does the same and
leaves no unsaved change behind. They change a code file, which opens in a tab of its own, and
close that tab without saving: the Unsaved changes prompt asks whether to discard the change;
Confirm closes the tab with the file on disk as it was, and Cancel stays on the edited file, still
marked unsaved, which Ctrl+S then puts on disk.

Discriminates: passes on dev 30f3f6a8f; on a build where the file browser's upload handler does
nothing, its search shows no results, its bulk delete returns early and its save handler returns
early, every test here fails. The unsaved-changes tests pass on dev 3b8205505 and fail on a build
where the file browser leaves an edited file without asking. The text, Ctrl+S and Confirm tests
fail on one frontend build of dev f659c0d12 whose Save button does nothing, whose code editor
ignores Ctrl+S and whose prompt's Confirm keeps the tab.

Since 10f1226f0 selecting a file no longer pushes the rows down to make room for the bulk action
bar, and a click on an empty part of the file list, such as the grid's edge, clears the selection.
The selection test passes on dev e8c0ad85e and fails in a frontend copy with 10f1226f0 reverted.

Since 3797ce419 (0.13.1) Sign Out in the user menu warns while a file has unsaved edits: the
Unsaved changes dialog asks to discard them and sign out; Cancel keeps the person signed in on
the edited file and Sign Out there signs them out. With no unsaved edit Sign Out asks nothing.
Both pass on dev e8c0ad85e; the warning test fails on a frontend build of it whose Sign Out skips
the check for unsaved edits, where the no-edit test stays green.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal
from utils.user_menu import user_menu_trigger

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def browser(page_for, make_user, admin, preserve, open_terminal):
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    return page.get_by_role("region", name="File browser")


def test_a_chosen_file_is_uploaded_into_the_terminal(browser, open_terminal):
    browser.get_by_label("Actions", exact=True).click()
    with browser.page.expect_file_chooser() as chooser:
        browser.page.get_by_role("menuitem", name="Upload", exact=True).click()
    chooser.value.set_files(
        files=[{"name": "ledger-upload.txt", "mimeType": "text/plain", "buffer": b"two kegs\n"}]
    )

    expect(browser.get_by_text("ledger-upload.txt", exact=True)).to_be_visible()
    uploaded = open_terminal.home / "ledger-upload.txt"
    assert uploaded.read_text() == "two kegs\n"


def test_search_finds_a_file_by_its_contents_and_another_by_its_name(browser, open_terminal):
    (open_terminal.home / "cellar-log.txt").write_text("racked the quince perry\n")
    (open_terminal.home / "marmalade-recipe.txt").write_text("oranges and sugar\n")
    (open_terminal.home / "unrelated-note.txt").write_text("nothing here\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    search_toggle = browser.get_by_role("button", name="Search", exact=True)
    expect(search_toggle).to_have_attribute("aria-expanded", "false")
    search_toggle.click()
    search = browser.get_by_placeholder("Search files and contents")

    search.fill("quince")

    expect(browser.get_by_text("Content matches")).to_be_visible()
    expect(browser.get_by_text("cellar-log.txt")).to_be_visible()
    expect(browser.get_by_text("marmalade-recipe.txt")).to_have_count(0)

    search.fill("marmalade")

    expect(browser.get_by_text("Filename matches")).to_be_visible()
    expect(browser.get_by_text("marmalade-recipe.txt")).to_be_visible()
    expect(browser.get_by_text("cellar-log.txt")).to_have_count(0)


def test_select_all_then_delete_removes_the_selected_files(browser, open_terminal):
    home = open_terminal.home
    for name in ("bulk-a.txt", "bulk-b.txt", "bulk-c.txt"):
        (home / name).write_text(name)
    browser.get_by_label("Refresh", exact=True).first.click()
    expect(browser.get_by_text("bulk-c.txt", exact=True)).to_be_visible()

    browser.get_by_text("bulk-a.txt", exact=True).click(modifiers=["Control"])
    browser.get_by_role("button", name="Select All").click()
    expect(browser.get_by_role("checkbox", name="Select bulk-c.txt")).to_be_checked()
    browser.get_by_role("checkbox", name="Select bulk-b.txt").click()
    browser.get_by_role("button", name="Delete", exact=True).click()
    browser.page.get_by_role("dialog").get_by_role("button", name="Confirm").click()

    expect(browser.get_by_text("bulk-a.txt", exact=True)).to_have_count(0)
    expect(browser.get_by_text("bulk-c.txt", exact=True)).to_have_count(0)
    expect(browser.get_by_text("bulk-b.txt", exact=True)).to_be_visible()
    assert not (home / "bulk-a.txt").exists()
    assert not (home / "bulk-c.txt").exists()
    assert (home / "bulk-b.txt").exists()


def _top(row) -> float:
    return row.evaluate("row => row.getBoundingClientRect().top")


def test_selecting_keeps_the_rows_in_place_and_a_click_beside_them_clears_it(
    browser, open_terminal
):
    for name in ("pick-a.txt", "pick-b.txt"):
        (open_terminal.home / name).write_text(name)
    browser.get_by_label("Refresh", exact=True).first.click()
    first = browser.get_by_text("pick-a.txt", exact=True)
    expect(browser.get_by_text("pick-b.txt", exact=True)).to_be_visible()
    before = _top(first)

    first.click(modifiers=["Control"])

    expect(browser.get_by_role("button", name="Select All")).to_be_visible()
    assert _top(first) == before, f"the rows moved from {before} to {_top(first)} on selecting"

    browser.get_by_label("Actions", exact=True).click()
    browser.page.get_by_role("menu").get_by_role("menuitem", name="Grid view").click()
    # the grid has no name or role of its own
    tiles = browser.locator("ul").filter(has=browser.page.get_by_text("pick-b.txt", exact=True))
    browser.get_by_text("pick-b.txt", exact=True).click(modifiers=["Control"])
    expect(browser.get_by_role("checkbox", name="Select pick-b.txt")).to_be_checked()
    # the grid's padding, beside every tile
    tiles.last.click(position={"x": 2, "y": 2})
    expect(browser.get_by_role("button", name="Select All")).to_have_count(0)


def test_an_edited_text_file_is_saved_to_the_terminal(browser, open_terminal):
    note = open_terminal.home / f"edit-me-{uuid.uuid4().hex[:8]}.txt"
    note.write_text("old words\n")
    browser.get_by_label("Refresh", exact=True).first.click()

    # since b5eaa59d0 a text file opens straight in the editor, with no Edit step
    browser.get_by_text(note.name, exact=True).click()
    browser.get_by_text("old words", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.type(" and new")
    expect(browser.get_by_text("old words and new", exact=True)).to_be_visible()
    browser.get_by_role("button", name="Save", exact=True).click()

    assert wait_until(lambda: note.read_text() == "old words and new\n", 10), (
        f"Save left the file as {note.read_text()!r}"
    )


def _leave_an_edited_file(browser, open_terminal):
    """Change a code file, close its tab without saving and return the file and the prompt."""
    script = open_terminal.home / f"leave-me-{uuid.uuid4().hex[:8]}.py"
    script.write_text("barrels = 4\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(script.name, exact=True).click()
    browser.get_by_text("barrels = 4", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.press("Backspace")
    browser.page.keyboard.type("5")
    expect(browser.get_by_text("barrels = 5", exact=True)).to_be_visible()

    browser.page.get_by_role("tablist", name="Tabs").get_by_role(
        "button", name=f"Close {script.name}", exact=True
    ).click()
    prompt = browser.page.get_by_role("dialog", name="Unsaved changes")
    expect(prompt).to_be_visible()
    return script, prompt


def test_confirming_the_unsaved_changes_prompt_discards_the_edit(browser, open_terminal):
    script, prompt = _leave_an_edited_file(browser, open_terminal)
    # since d4e8493cc the prompt asks whether to discard the change
    expect(prompt).to_contain_text("Discard unsaved changes and continue?")

    prompt.get_by_role("button", name="Confirm", exact=True).click()

    expect(prompt).to_have_count(0)
    tabs = browser.page.get_by_role("tablist", name="Tabs")
    expect(tabs.get_by_role("button", name=f"Close {script.name}", exact=True)).to_have_count(0)
    expect(browser.get_by_text(script.name, exact=True)).to_be_visible()
    expect(browser.get_by_text("barrels = ")).to_have_count(0)
    assert script.read_text() == "barrels = 4\n", f"the file now reads {script.read_text()!r}"


def test_cancelling_the_unsaved_changes_prompt_stays_on_the_edited_file(browser, open_terminal):
    script, prompt = _leave_an_edited_file(browser, open_terminal)

    prompt.get_by_role("button", name="Cancel", exact=True).click()

    expect(prompt).to_have_count(0)
    tabs = browser.page.get_by_role("tablist", name="Tabs")
    expect(tabs.get_by_role("button", name=f"Close {script.name}", exact=True)).to_be_visible()
    expect(browser.get_by_text("barrels = 5", exact=True)).to_be_visible()
    expect(tabs.get_by_label("Unsaved changes")).to_have_count(1)
    assert script.read_text() == "barrels = 4\n", "Cancel should leave the file on disk alone"

    browser.get_by_text("barrels = 5", exact=True).click()
    browser.page.keyboard.press("Control+s")

    assert wait_until(lambda: script.read_text() == "barrels = 5\n", 10), (
        f"Ctrl+S after Cancel left the file as {script.read_text()!r}"
    )
    expect(tabs.get_by_label("Unsaved changes")).to_have_count(0)


def test_ctrl_s_saves_an_edited_code_file_and_leaves_nothing_unsaved(browser, open_terminal):
    script = open_terminal.home / f"tally-{uuid.uuid4().hex[:8]}.py"
    script.write_text("casks = 4\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(script.name, exact=True).click()
    browser.get_by_text("casks = 4", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.press("Backspace")
    browser.page.keyboard.type("9")
    expect(browser.get_by_text("casks = 9", exact=True)).to_be_visible()

    browser.page.keyboard.press("Control+s")

    assert wait_until(lambda: script.read_text() == "casks = 9\n", 10), (
        f"Ctrl+S left the file as {script.read_text()!r}"
    )
    browser.page.get_by_role("tablist", name="Tabs").get_by_role(
        "button", name=f"Close {script.name}", exact=True
    ).click()
    expect(browser.get_by_text(script.name, exact=True)).to_be_visible()
    expect(browser.page.get_by_role("dialog", name="Unsaved changes")).to_have_count(0)


def _edit_a_code_file(browser, open_terminal) -> None:
    script = open_terminal.home / f"unsaved-{uuid.uuid4().hex[:8]}.py"
    script.write_text("ropes = 2\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(script.name, exact=True).click()
    browser.get_by_text("ropes = 2", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.type("0")
    expect(browser.get_by_text("ropes = 20", exact=True)).to_be_visible()


def _sign_out(page) -> None:
    user_menu_trigger(page).click()
    page.get_by_role("menu").get_by_role("menuitem", name="Sign Out").click()


def test_signing_out_with_an_unsaved_file_edit_asks_first(browser, open_terminal):
    page = browser.page
    _edit_a_code_file(browser, open_terminal)

    _sign_out(page)
    warning = page.get_by_role("dialog", name="Unsaved changes")
    expect(warning).to_contain_text("Discard unsaved file edits and sign out?")
    warning.get_by_role("button", name="Cancel", exact=True).click()

    expect(warning).to_have_count(0)
    expect(browser.get_by_text("ropes = 20", exact=True)).to_be_visible()
    expect(page).not_to_have_url(re.compile(r"/auth"))

    # the user menu may still be open behind the closed warning
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    _sign_out(page)
    page.get_by_role("dialog", name="Unsaved changes").get_by_role(
        "button", name="Sign Out", exact=True
    ).click()
    expect(page).to_have_url(re.compile(r"/auth"))


def test_signing_out_without_unsaved_edits_asks_nothing(browser):
    page = browser.page

    _sign_out(page)

    expect(page).to_have_url(re.compile(r"/auth"))
