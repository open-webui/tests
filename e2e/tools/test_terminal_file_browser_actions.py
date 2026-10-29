"""Journey: a person creates, renames, deletes and reveals files in the terminal's file browser.

A real Open Terminal runs with a few files in its home. In a chat the person picks the terminal
from the input's Terminal menu and works in the File browser: New Folder and New File in its
Actions menu make them in the terminal's home, Rename and Delete in a row's More menu rename the
file or remove it after a confirm, and Show Hidden Files lists a dotfile the browser hides by
default. Every change is read back from the terminal's disk. Select for Compare on one file and
Compare with Selected on another open both files side by side, read only, each under its name, and
Swap sides puts the second file on the left.

Discriminates: passes on dev 30f3f6a8f; on a build whose New Folder, New File, Rename and Delete
handlers send nothing and whose hidden-files switch does not change the listing, every test here
fails. The compare test passes on the dev 3b8205505 build and fails in a build of it whose Swap
sides leaves the files where they were.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def home(open_terminal):
    """The terminal's home, with the files the tests act on."""
    folder = open_terminal.home
    (folder / "draft-letter.txt").write_text("Dear harbour master,\n")
    (folder / "old-receipt.txt").write_text("two ropes, one lamp\n")
    (folder / ".boat-secrets").write_text("spare key under the buoy\n")
    return folder


@pytest.fixture
def browser(home, page_for, make_user, admin, preserve, open_terminal) -> Locator:
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    region = page.get_by_role("region", name="File browser")
    expect(region.get_by_text("draft-letter.txt", exact=True)).to_be_visible()
    return region


def _page(browser: Locator) -> Page:
    return browser.page


def _choose_action(browser: Locator, action: str) -> None:
    browser.get_by_label("Actions", exact=True).click()
    _page(browser).get_by_role("menu").get_by_role("menuitem", name=action).click()


def _row_menu(browser: Locator, name: str) -> Locator:
    row = browser.get_by_text(name, exact=True).locator(
        "xpath=ancestor::*[.//button[@aria-label='More']][1]"
    )
    row.hover()
    row.get_by_label("More", exact=True).click()
    return _page(browser).get_by_role("menu")


def test_new_folder_makes_the_folder_in_the_terminal(browser, home):
    _choose_action(browser, "New Folder")
    browser.get_by_placeholder("Folder name").fill("charts")
    browser.get_by_placeholder("Folder name").press("Enter")

    expect(browser.get_by_text("charts", exact=True)).to_be_visible()
    assert (home / "charts").is_dir()


def test_new_file_makes_the_file_in_the_terminal(browser, home):
    _choose_action(browser, "New File")
    browser.get_by_placeholder("File name").fill("tide-log.md")
    browser.get_by_placeholder("File name").press("Enter")

    expect(browser.get_by_text("tide-log.md", exact=True)).to_be_visible()
    assert (home / "tide-log.md").is_file()


def test_rename_renames_the_file_in_the_terminal(browser, home):
    _row_menu(browser, "draft-letter.txt").get_by_role("menuitem", name="Rename").click()
    field = browser.locator("input:focus")
    expect(field).to_have_value("draft-letter.txt")
    field.fill("sent-letter.txt")
    field.press("Enter")

    expect(browser.get_by_text("sent-letter.txt", exact=True)).to_be_visible()
    expect(browser.get_by_text("draft-letter.txt", exact=True)).to_have_count(0)
    assert (home / "sent-letter.txt").read_text() == "Dear harbour master,\n"
    assert not (home / "draft-letter.txt").exists()


def test_delete_removes_the_file_from_the_terminal_after_a_confirm(browser, home):
    _row_menu(browser, "old-receipt.txt").get_by_role("menuitem", name="Delete").click()
    assert (home / "old-receipt.txt").exists()
    _page(browser).get_by_role("button", name="Confirm").click()

    expect(browser.get_by_text("old-receipt.txt", exact=True)).to_have_count(0)
    assert not (home / "old-receipt.txt").exists()


def test_show_hidden_files_lists_a_dotfile(browser):
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_have_count(0)

    _choose_action(browser, "Show Hidden Files")
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_be_visible()

    _choose_action(browser, "Hide Hidden Files")
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_have_count(0)


def test_compare_with_selected_opens_both_files_and_swap_sides_swaps_them(browser):
    _row_menu(browser, "draft-letter.txt").get_by_role(
        "menuitem", name="Select for Compare"
    ).click()
    expect(browser.get_by_text("Comparing with draft-letter.txt")).to_be_visible()
    _row_menu(browser, "old-receipt.txt").get_by_role(
        "menuitem", name="Compare with Selected"
    ).click()

    compare = _page(browser).get_by_role("dialog", name="Compare files")
    panes = compare.get_by_role("textbox")
    expect(panes).to_have_count(2)
    expect(panes.first).to_have_accessible_name("draft-letter.txt")
    expect(panes.first).to_have_text("Dear harbour master,")
    expect(panes.last).to_have_accessible_name("old-receipt.txt")
    expect(panes.last).to_have_text("two ropes, one lamp")
    # not_to_be_editable reads only input readonly on the pinned Playwright 1.49
    expect(panes.first).to_have_attribute("contenteditable", "false")
    expect(compare.get_by_role("status")).to_have_text("Change 1 of 1")

    compare.get_by_role("button", name="Swap sides").click()
    expect(panes.first).to_have_accessible_name("old-receipt.txt")
    expect(panes.first).to_have_text("two ropes, one lamp")
    expect(panes.last).to_have_accessible_name("draft-letter.txt")
    expect(panes.last).to_have_text("Dear harbour master,")
