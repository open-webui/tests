"""Journey: a file the terminal marks read only stays read only in the chat's file browser.

A real Open Terminal holds a file its owner cannot write and one it can. In a chat with the
terminal picked, the File browser marks the first Read-only and opens it in a code editor that
takes no typing, while the writable file's editor does.

Discriminates: passes on the dev 9bbb95048 build; in a frontend copy whose code editor ignores
the read-only flag (as before 9bbb95048), the test goes red.
"""

from __future__ import annotations

import stat

import pytest
from playwright.sync_api import Locator, expect

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def home(open_terminal):
    """The terminal's home with a read-only file and a writable one."""
    folder = open_terminal.home
    locked = folder / "harbour-rules.py"
    locked.write_text("SPEED_LIMIT = 5\n")
    locked.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    (folder / "log-book.py").write_text("ENTRIES = []\n")
    yield folder
    locked.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.fixture
def browser(home, page_for, make_user, admin, preserve, open_terminal) -> Locator:
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    page.goto("/")
    pick_terminal(page, connection["name"])
    region = page.get_by_role("region", name="File browser")
    expect(region.get_by_text("harbour-rules.py", exact=True)).to_be_visible()
    return region


def _code_editor(browser: Locator) -> Locator:
    # the code editor has no accessible name; its editable area is the only one in the panel
    return browser.page.locator(".cm-content")


def test_a_read_only_file_opens_in_an_editor_that_takes_no_typing(browser, home):
    expect(browser.get_by_text("Read-only", exact=True)).to_have_count(1)
    browser.get_by_text("harbour-rules.py", exact=True).click()
    editor = _code_editor(browser)
    expect(editor).to_contain_text("SPEED_LIMIT = 5")
    editor.click()
    browser.page.keyboard.type("# changed")
    expect(editor).not_to_contain_text("# changed")

    browser.page.get_by_role("button", name="Back").first.click()
    browser.get_by_text("log-book.py", exact=True).click()
    expect(editor).to_contain_text("ENTRIES = []")
    editor.click()
    browser.page.keyboard.type("# changed")
    expect(editor).to_contain_text("# changed")
    assert (home / "harbour-rules.py").read_text() == "SPEED_LIMIT = 5\n"
