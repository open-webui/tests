"""Journey: Quick Open finds a terminal file by name; a new file is written and saved in a tab.

On a real Open Terminal picked for the chat, Ctrl+P opens Quick Open (22bce6021), a search box
over the terminal's files: typing part of a name, with or without the folders above it, lists the
matching files with their paths, the arrow keys move the pick, Enter opens the picked file in a
tab of its own (or selects its tab when it is already open), Escape or Ctrl+P again closes it
without opening anything, and a name nothing matches says so. A file opened this way is one of
the chat's saved tabs and is back after a reload. An empty Quick Open offers nothing yet, where
an editor's Quick Open starts with the recently opened files; that gap is a strict xfail.

New File in the New tab menu opens an Untitled tab. It is marked unsaved once typed into; Save
asks for a name in a Save As dialog, refuses a name with a slash, asks before replacing an
existing file, and writes the text to the terminal, after which the tab carries the file's name.
Closing a typed Untitled tab asks before throwing the text away, leaving the page asks too, and an
Untitled tab is never brought back by a reload.

Discriminates: passes on the dev f659c0d12 build. On that build patched so Enter in Quick Open
opens nothing, every Quick Open test and the reload test fail; with the arrow keys not moving the
pick, the arrows test fails; with Save As taking a name with a slash, the save test fails; with
Save As skipping the existing-file check, the replace test fails as the file is overwritten
unasked; with the saved view keeping Untitled tabs, the reload test fails; with leaving the page
never asking about unsaved tabs, the leaving test fails.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from e2e.conftest import APP_LOADED
from harness import upstream as reply
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import ask, choose_terminal
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "When is high water today?"
ANSWER = "The tide table in the terminal has it."


@pytest.fixture
def folder(open_terminal) -> Path:
    """A folder of this test's own in the terminal's home, with two files of a shared stem."""
    home = open_terminal.home / f"harbour-{uuid.uuid4().hex[:6]}"
    home.mkdir()
    (home / "tide-table.md").write_text("# Tides\n\nHigh water at six.\n")
    (home / "tide-notes.txt").write_text("the spring tide comes on Sunday\n")
    return home


@pytest.fixture
def chat(page_for, make_user, admin, preserve, open_terminal, upstream) -> Page:
    """A chat of a fresh account with the Open Terminal picked and one answered question.

    A terminal is picked before the first message, and only a saved chat keeps its tabs."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    choose_terminal(page, connection["name"])
    question = f"{QUESTION} ({uuid.uuid4().hex[:8]})"
    ask(page, upstream, question, reply.text(ANSWER))
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    return page


def _quick_open(page: Page) -> Locator:
    page.keyboard.press("Control+P")
    dialog = page.get_by_role("dialog", name="Quick Open")
    expect(dialog).to_be_visible()
    return dialog


def _search(dialog: Locator, query: str) -> Locator:
    dialog.get_by_role("combobox", name="Search Files").fill(query)
    return dialog.get_by_role("listbox", name="Search Files").get_by_role("option")


def _open_by_name(page: Page, folder: Path, name: str) -> None:
    """Open `name` from `folder` with Quick Open, once it is the one file listed."""
    dialog = _quick_open(page)
    expect(_search(dialog, f"{folder.name}/{name}")).to_have_count(1)
    page.keyboard.press("Enter")
    expect(page.get_by_role("tab", name=name)).to_have_attribute("aria-selected", "true")


def _reloaded(page: Page) -> None:
    """Reload and wait for the app to load again, as `page_for` does for the first load."""
    page.reload()
    page.wait_for_function(APP_LOADED, timeout=REPLY_TIMEOUT_MS)


def _add_tab(page: Page, entry: str) -> None:
    page.get_by_role("button", name="New tab", exact=True).click()
    page.get_by_role("menu").get_by_role("menuitem", name=entry).click()


def _untitled(page: Page) -> Locator:
    _add_tab(page, "New File")
    panel = page.get_by_role("tabpanel", name="Untitled-1")
    expect(panel).to_be_visible()
    return panel


def _type_into(panel: Locator, text: str) -> None:
    editor = panel.get_by_role("textbox", name="File content")
    editor.click()
    editor.page.keyboard.type(text)
    expect(editor).to_contain_text(text)


def _save_as(page: Page, panel: Locator, folder: Path, name: str) -> Locator:
    """Press Save, open `folder` in the Save As dialog and type `name`; returns the dialog."""
    panel.get_by_role("button", name="Save", exact=True).click()
    dialog = page.get_by_role("dialog", name="Save As")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name=folder.name).click()
    expect(dialog.get_by_text(str(folder), exact=True)).to_be_visible()
    dialog.get_by_role("textbox", name="File name").fill(name)
    return dialog


def test_quick_open_finds_a_file_by_name_and_opens_it_in_a_tab(chat, folder):
    expect(chat_input(chat)).to_be_focused()
    dialog = _quick_open(chat)

    results = _search(dialog, f"{folder.name}/tide-ta")

    expect(results).to_have_count(1)
    expect(results.first).to_contain_text("tide-table.md")
    expect(results.first).to_contain_text(str(folder / "tide-table.md"))
    expect(results.first).to_have_attribute("aria-selected", "true")
    chat.keyboard.press("Enter")

    expect(dialog).to_have_count(0)
    expect(chat.get_by_role("tab", name="tide-table.md")).to_have_attribute("aria-selected", "true")
    expect(chat.get_by_role("tabpanel", name="tide-table.md")).to_contain_text("High water at six.")

    _reloaded(chat)
    expect(chat.get_by_role("tabpanel", name="tide-table.md")).to_contain_text("High water at six.")


def test_quick_open_moves_the_pick_with_the_arrows_and_says_when_nothing_matches(chat, folder):
    expect(chat_input(chat)).to_be_focused()
    dialog = _quick_open(chat)
    results = _search(dialog, f"{folder.name}/tide-")
    expect(results).to_have_count(2)
    chat.keyboard.press("ArrowDown")
    expect(results.nth(1)).to_have_attribute("aria-selected", "true")
    expect(results.nth(0)).to_have_attribute("aria-selected", "false")
    picked, unpicked = (
        ("tide-notes.txt", "tide-table.md")
        if "tide-notes.txt" in results.nth(1).inner_text()
        else ("tide-table.md", "tide-notes.txt")
    )
    chat.keyboard.press("Enter")

    expect(dialog).to_have_count(0)
    expect(chat.get_by_role("tab", name=picked)).to_have_attribute("aria-selected", "true")

    dialog = _quick_open(chat)
    _search(dialog, f"{folder.name}/no-such-chart")
    expect(dialog.get_by_role("status")).to_have_text("No files found")
    chat.keyboard.press("Escape")
    expect(dialog).to_have_count(0)

    dialog = _quick_open(chat)
    _search(dialog, f"{folder.name}/tide-")
    chat.keyboard.press("Control+P")
    expect(dialog).to_have_count(0)
    expect(chat.get_by_role("tab", name=picked)).to_have_count(1)
    expect(chat.get_by_role("tab", name=unpicked)).to_have_count(0)


def test_a_new_file_is_saved_to_the_terminal_under_the_name_given(chat, folder):
    panel = _untitled(chat)
    tab = chat.get_by_role("tab", name="Untitled-1")
    _type_into(panel, "Neap tide on Thursday.")
    expect(chat.get_by_label("Unsaved changes")).to_be_visible()

    dialog = _save_as(chat, panel, folder, "tides/next-week.md")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.get_by_role("alert")).to_have_text("Enter a file name without slashes.")

    dialog.get_by_role("textbox", name="File name").fill("next-week.md")
    dialog.get_by_role("button", name="Save", exact=True).click()

    expect(dialog).to_have_count(0)
    expect(chat.get_by_role("tab", name="next-week.md")).to_have_attribute("aria-selected", "true")
    expect(tab).to_have_count(0)
    saved = folder / "next-week.md"
    assert saved.exists() and saved.read_text() == "Neap tide on Thursday.", (
        f"Save wrote {saved.read_text() if saved.exists() else 'nothing'!r} to {saved}"
    )


def test_saving_over_an_existing_file_asks_before_replacing_it(chat, folder):
    panel = _untitled(chat)
    _type_into(panel, "Replaced tide table.")
    dialog = _save_as(chat, panel, folder, "tide-table.md")

    dialog.get_by_role("button", name="Save", exact=True).click()

    expect(dialog.get_by_role("alert")).to_have_text(
        "A file with this name already exists. Replace it?"
    )
    assert (folder / "tide-table.md").read_text() == "# Tides\n\nHigh water at six.\n", (
        "the existing file was overwritten before the person agreed"
    )
    dialog.get_by_role("button", name="Replace", exact=True).click()
    expect(dialog).to_have_count(0)
    assert (folder / "tide-table.md").read_text() == "Replaced tide table."


def test_closing_a_typed_new_file_asks_first_and_a_reload_never_brings_it_back(chat, folder):
    panel = _untitled(chat)
    _type_into(panel, "Draft that is never saved.")

    chat.get_by_role("button", name="Close Untitled-1").click()
    confirm = chat.get_by_role("dialog", name="Unsaved changes")
    expect(confirm).to_contain_text("Discard unsaved changes and close this file?")
    confirm.get_by_role("button", name="Cancel").click()
    expect(panel).to_contain_text("Draft that is never saved.")

    chat.get_by_role("button", name="Close Untitled-1").click()
    confirm.get_by_role("button", name="Confirm").click()
    expect(chat.get_by_role("tab", name="Untitled-1")).to_have_count(0)
    assert not [path for path in folder.iterdir() if "Draft" in path.read_text()]

    # an empty Untitled tab beside a file tab: after a reload only the file's tab is back
    _open_by_name(chat, folder, "tide-table.md")
    _untitled(chat)
    _reloaded(chat)
    expect(chat.get_by_role("tabpanel", name="tide-table.md")).to_contain_text("High water at six.")
    expect(chat.get_by_role("tab", name="Untitled-1")).to_have_count(0)


def test_quick_open_finds_a_file_deep_in_a_folder_and_reuses_its_open_tab(chat, folder):
    deep = folder / "charts" / "2026"
    deep.mkdir(parents=True)
    name = f"soundings-{uuid.uuid4().hex[:6]}.csv"
    (deep / name).write_text("depth,metres\nquay,4.2\n")
    expect(chat_input(chat)).to_be_focused()

    # the name alone, as in an editor's Quick Open, without the folders above it
    dialog = _quick_open(chat)
    results = _search(dialog, name[:-4])
    expect(results).to_have_count(1)
    expect(results.first).to_contain_text(str(deep / name))
    chat.keyboard.press("Enter")
    expect(chat.get_by_role("tab", name=name)).to_have_attribute("aria-selected", "true")

    _open_by_name(chat, folder, "tide-table.md")
    dialog = _quick_open(chat)
    expect(_search(dialog, name[:-4])).to_have_count(1)
    chat.keyboard.press("Enter")

    expect(chat.get_by_role("tab", name=name)).to_have_attribute("aria-selected", "true")
    expect(chat.get_by_role("tab", name=name)).to_have_count(1)


def test_leaving_the_page_with_an_unsaved_new_file_asks_first(chat, folder):
    panel = _untitled(chat)
    _type_into(panel, "Spring tide notes, not saved yet.")

    with chat.expect_event("dialog") as asked:
        chat.close(run_before_unload=True)

    assert asked.value.type == "beforeunload", f"leaving asked with a {asked.value.type} dialog"
    asked.value.dismiss()
    expect(panel).to_contain_text("Spring tide notes, not saved yet.")


@pytest.mark.xfail(
    strict=True,
    reason="gap: an empty Quick Open lists no files, not even the ones just opened, "
    "where an editor's Quick Open starts with the recently opened files",
)
def test_an_empty_quick_open_offers_the_files_opened_recently(chat, folder):
    expect(chat_input(chat)).to_be_focused()
    _open_by_name(chat, folder, "tide-notes.txt")

    dialog = _quick_open(chat)

    expect(
        dialog.get_by_role("listbox", name="Search Files").get_by_role("option")
    ).to_contain_text(["tide-notes.txt"])
