"""Journey: a person uploads, finds, bulk deletes and edits files in the terminal's file browser.

A real Open Terminal runs with a few files in its home. In a chat the person picks the terminal
from the input's Terminal menu. From the File browser's Actions menu they upload a file through
the file chooser and it shows in the list and lands in the terminal's home. They type into the
search box and find one file by its contents and another by its name. They Ctrl-click two rows,
press Select All, delete the selection and confirm: all the selected files are gone from the
list and from the disk while an unselected one stays. They open a text file, press Edit, type new
text and press Save: the new text is on disk.

Discriminates: passes on dev 30f3f6a8f; on a build where the file browser's upload handler
does nothing, its search shows no results, its bulk delete returns early and its save handler
returns early, every test here fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

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
    page.goto("/")
    pick_terminal(page, connection["name"])
    return page.get_by_role("region", name="File browser")


def test_a_chosen_file_is_uploaded_into_the_terminal(browser, open_terminal):
    browser.get_by_label("Actions", exact=True).click()
    with browser.page.expect_file_chooser() as chooser:
        browser.page.get_by_role("button", name="Upload", exact=True).click()
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


def test_an_edited_text_file_is_saved_to_the_terminal(browser, open_terminal):
    note = open_terminal.home / "edit-me.txt"
    note.write_text("old words\n")
    browser.get_by_label("Refresh", exact=True).first.click()

    browser.get_by_text("edit-me.txt", exact=True).click()
    browser.get_by_role("button", name="Edit").click()
    editor = browser.get_by_role("textbox")
    editor.fill("new words\n")
    browser.get_by_role("button", name="Save").click()

    expect(browser.get_by_role("button", name="Edit")).to_be_visible()
    assert note.read_text() == "new words\n"
