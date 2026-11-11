"""Journey: on a terminal with a machine per chat, the file browser works in the open chat's
machine.

An admin connects a Terminals orchestrator and shares a terminal scoped to the chat, so every
saved chat has a machine of its own. A person has two chats, each with a file of its own in its
machine. In the first chat they pick the terminal and open Files: the list shows that chat's file
and not the other's. They make a file, upload one, rename it, open a folder they made, download
a file and the folder as a ZIP and delete a file; every change is on the first chat's machine
and on no other. The second chat's Files shows its own file alone, and a file made there lands in
the second chat's machine. Each call of the file browser carries the chat it is made in, which is
what picks the machine.

The real Terminals service runs on Docker (`harness.terminals`); without its images or Docker the
test skips.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7,
three runs. On a frontend build whose file calls do not name the chat the first chat's Files
lists nothing and the test fails there. With the chat no longer named to the orchestrator
(`X-Terminal-Chat-Id` left out), in a backend copy, it fails at the first command of its setup
(409 "This terminal requires a chat").
"""

from __future__ import annotations

import io
import uuid
import zipfile
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat
from harness.terminal_server import read_grant
from harness.terminals import (
    create_template,
    execute,
    output_of,
    share_template,
    wait_until_addresses_settle,
)
from utils.cached_chat import choose_terminal, open_files_tab
from utils.file_browser import choose_action, expect_listed, open_actions, row, row_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _files_of_chat(page_for, person, chat_id: str, terminal: str) -> Locator:
    """The File browser of the person's saved chat with the terminal picked."""
    page: Page = page_for(person, path=f"/c/{chat_id}")
    choose_terminal(page, terminal)
    return open_files_tab(page)


def _new(browser: Locator, kind: str, name: str) -> None:
    choose_action(browser, f"New {kind}")
    field = browser.get_by_placeholder(f"{kind} name")
    field.fill(name)
    field.press("Enter")
    expect(row(browser, name)).to_be_visible()


def test_the_file_browser_of_a_chat_works_in_that_chats_machine(
    page_for, make_user, admin, connect_terminals, terminals
):
    person = make_user()
    tag = uuid.uuid4().hex[:6]
    name = f"Per chat box {tag}"
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name, scope="chat")["terminal_id"]
        share_template(client, terminal_id, read_grant(person.id))
    with person.client() as client:
        first_chat, _ = seed_chat(client, [{"role": "user", "content": f"first {tag}"}])
        second_chat, _ = seed_chat(client, [{"role": "user", "content": f"second {tag}"}])
        # each machine starts here, before a page is open that its start could cut off
        output_of(execute(client, terminal_id, "echo harbour > first-log.txt", first_chat))
        output_of(execute(client, terminal_id, "echo cargo > second-log.txt", second_chat))
    wait_until_addresses_settle()
    first, second = f"chat:{first_chat}", f"chat:{second_chat}"

    browser = _files_of_chat(page_for, person, first_chat, name)
    page = browser.page
    expect_listed(browser, "first-log.txt")

    _new(browser, "File", "tides.md")
    menu = open_actions(browser)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload", exact=True).click()
    chooser.value.set_files({"name": "ledger.txt", "mimeType": "text/plain", "buffer": b"kegs\n"})
    expect(row(browser, "ledger.txt")).to_be_visible()

    row_menu(browser, "ledger.txt").get_by_role("menuitem", name="Rename", exact=True).click()
    field = browser.get_by_role("textbox", name="Rename", exact=True)
    field.fill("stock.txt")
    field.press("Enter")
    expect(page.get_by_text("Renamed to stock.txt")).to_be_visible()
    _new(browser, "Folder", "sheds")
    expect_listed(browser, "sheds/", "first-log.txt", "stock.txt", "tides.md")

    menu = row_menu(browser, "stock.txt")
    with page.expect_download() as download:
        menu.get_by_role("menuitem", name="Download", exact=True).click()
    assert Path(download.value.path()).read_bytes() == b"kegs\n", "the downloaded file"

    browser.get_by_text("sheds", exact=True).click()
    expect(browser.get_by_text("This folder is empty")).to_be_visible()
    _new(browser, "File", "nets.txt")
    with page.expect_download() as download:
        choose_action(browser, "Download")
    with zipfile.ZipFile(io.BytesIO(Path(download.value.path()).read_bytes())) as archive:
        assert archive.namelist() == ["sheds/nets.txt"], "the folder's ZIP"
    browser.get_by_role("button", name="Back", exact=True).click()

    row_menu(browser, "tides.md").get_by_role("menuitem", name="Delete", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.get_by_text("tides.md deleted")).to_be_visible()
    expect_listed(browser, "sheds/", "first-log.txt", "stock.txt")

    assert terminals.holding("stock.txt") == {first: "kegs\n"}, "the uploaded and renamed file"
    assert terminals.holding("sheds/nets.txt") == {first: ""}, "the file made in the new folder"
    assert terminals.holding("ledger.txt") == {}, "the file's name before the rename"
    assert terminals.holding("tides.md") == {}, "the deleted file"

    other = _files_of_chat(page_for, person, second_chat, name)
    expect_listed(other, "second-log.txt")
    _new(other, "File", "manifest.txt")
    assert terminals.holding("manifest.txt") == {second: ""}, "the second chat's new file"
    assert terminals.holding("first-log.txt") == {first: "harbour\n"}
    assert terminals.holding("second-log.txt") == {second: "cargo\n"}
