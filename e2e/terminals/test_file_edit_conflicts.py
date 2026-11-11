"""Journey: an edit a person has not saved yet is never lost and never silently overwrites.

A real Open Terminal runs with a folder of the test's own. The person opens a text file beside
the conversation and types into it. When the model rewrites that file meanwhile, the editor
keeps the person's text and a line says the file was updated on disk, with Reload and Save
local changes: saving asks before it replaces the model's version and writes nothing when the
person cancels; Reload asks before it drops the edit and then shows the model's text. A save
the terminal refuses says the changes are still in the editor, keeps them and the tab's
unsaved mark, and goes through once the terminal accepts it. An unsaved edit is still in the
editor after the person looked at another tab. A workbook that changed on disk since it was
opened is not overwritten by a save.

Two tests are red on dev c15f28649 for product bugs:

- test_a_file_opened_before_the_first_message_is_saved_after_it: Save answers "File is not
  writable." and writes nothing. Since 6f2fc7bc7 the save compares the chat the file was opened
  in with the chat now, and a new chat has none until its first message. Passes on 0.13.3.
  Finding `open-webui-file-opened-before-first-message-not-saved`.
- test_an_unsaved_edit_is_still_there_after_visiting_another_chat: after another chat was
  opened and left again the file shows what is on disk, with no unsaved mark and without a
  question on the way. The same on 0.13.3. Finding
  `open-webui-unsaved-file-edit-lost-on-chat-switch`.

Discriminates: on dev c15f28649 with Open Terminal 0.17.7 all pass but the two above. On one
frontend build of it with these mutations each named test fails: the overwrite question skipped
(rewrite under an edit); the notice's Reload doing nothing (reload); a refused save treated as
saved (refused save); a file's edit dropped when its tab is left (another tab); the workbook's
check against the disk skipped (workbook). On that build the first-message test passes, as its
save compares the terminal only.
"""

from __future__ import annotations

import io
import re
import shutil
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import slow_machine
from harness import upstream as reply
from harness.backends import wait_until
from harness.chat_history import seed_chat
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import calling
from utils.chat_ui import expect_reply, send, split_screen
from utils.file_browser import file_browser
from utils.file_views import files_of, open_file, tab
from utils.sidebar_chats import chat_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# the page parses a workbook and lays out its whole grid in the browser, seconds on a slow CPU
WORKBOOK_LOAD_TIMEOUT_MS = slow_machine.scaled(5_000)
WRITTEN = "pressed on Monday\n"
TYPED = "bottled on Friday"
FROM_THE_MODEL = "the model's own account\n"
CHANGED_ON_DISK = "File updated on disk. Your local changes are preserved."
# the title `seed_chat` stores a chat under
SEEDED_TITLE = "Seeded chat"


@pytest.fixture
def folder(open_terminal):
    """A folder of this test's own in the terminal's home, which the whole session shares."""
    path = open_terminal.home / f"edits-{uuid.uuid4().hex[:6]}"
    path.mkdir()
    (path / "log.txt").write_text(WRITTEN)
    yield path
    path.chmod(0o755)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def page(folder, person, page_for, admin, preserve, open_terminal) -> Page:
    """The person's chat with the folder's file browser beside the conversation."""
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, open_terminal, folder.name)
    split_screen(browser.page)
    return browser.page


def _say(page: Page, upstream, answer: str) -> str:
    """Send a message the model answers with `answer`; returns the prompt, the chat's title."""
    prompt = f"note for the cellar {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, answer)
    return prompt


def _editor(page: Page) -> Locator:
    return file_browser(page).get_by_role("textbox")


def _type_at_the_end(page: Page, text: str = TYPED) -> None:
    editor = _editor(page)
    expect(editor).to_contain_text(WRITTEN.strip())
    editor.click()
    page.keyboard.press("Control+End")
    # in one piece, so the editor completes nothing of its own
    page.keyboard.insert_text(text)
    expect(editor).to_contain_text(text)


def _unsaved_marks(page: Page) -> Locator:
    return page.get_by_role("tablist").get_by_label("Unsaved changes")


def _have_the_model_rewrite_the_file_under_an_edit(page: Page, upstream, path) -> None:
    """The person asks for a rewrite and types into the file before the model writes it."""
    writing = threading.Event()
    prompt = f"rewrite the log {uuid.uuid4().hex[:6]}"
    rewrite = calling("write_file", {"path": str(path), "content": FROM_THE_MODEL}, hold=writing)
    rewrite.match = reply.answering(prompt)
    upstream.queue(rewrite, reply.text("Rewritten.", match=reply.answering(prompt)))
    send(page, prompt)
    _type_at_the_end(page)
    writing.set()
    expect_reply(page, "Rewritten.")
    assert path.read_text() == FROM_THE_MODEL, (
        f"the model's rewrite is not on disk: {path.read_text()!r}"
    )


def test_a_file_the_model_rewrites_under_an_edit_keeps_the_edit_and_asks_before_replacing(
    page, folder, upstream
):
    log = folder / "log.txt"
    # the chat is stored first: see the regression test below
    _say(page, upstream, "Noted.")
    open_file(file_browser(page), "log.txt")

    _have_the_model_rewrite_the_file_under_an_edit(page, upstream, log)

    browser = file_browser(page)
    notice = browser.get_by_role("status").filter(has_text=CHANGED_ON_DISK)
    expect(notice).to_be_visible()
    expect(_editor(page)).to_contain_text(TYPED)
    expect(_editor(page)).not_to_contain_text(FROM_THE_MODEL.strip())

    notice.get_by_role("button", name="Save local changes").click()
    question = page.get_by_role("dialog", name="File changed on disk")
    expect(question).to_contain_text("This file changed on disk. Replace it with your edits?")
    question.get_by_role("button", name="Cancel").click()
    expect(page.get_by_text("Save cancelled.")).to_be_visible()
    assert log.read_text() == FROM_THE_MODEL, f"a cancelled save wrote {log.read_text()!r}"
    expect(notice).to_be_visible()
    expect(_editor(page)).to_contain_text(TYPED)

    notice.get_by_role("button", name="Save local changes").click()
    question.get_by_role("button", name="Replace").click()
    assert wait_until(lambda: log.read_text() == WRITTEN + TYPED, 10), (
        f"the person's edit did not replace the model's version: {log.read_text()!r}"
    )
    expect(notice).to_have_count(0)
    expect(_unsaved_marks(page)).to_have_count(0)


def test_reload_drops_the_edit_only_after_asking_and_shows_what_the_model_wrote(
    page, folder, upstream
):
    log = folder / "log.txt"
    _say(page, upstream, "Noted.")
    open_file(file_browser(page), "log.txt")
    _have_the_model_rewrite_the_file_under_an_edit(page, upstream, log)
    browser = file_browser(page)
    notice = browser.get_by_role("status").filter(has_text=CHANGED_ON_DISK)

    notice.get_by_role("button", name="Reload").click()
    question = page.get_by_role("dialog", name="Unsaved changes")
    expect(question).to_contain_text("Discard unsaved changes and continue?")
    question.get_by_role("button", name="Cancel").click()
    expect(question).to_have_count(0)
    expect(_editor(page)).to_contain_text(TYPED)
    expect(notice).to_be_visible()

    notice.get_by_role("button", name="Reload").click()
    question.get_by_role("button", name="Confirm").click()

    expect(_editor(page)).to_have_text(FROM_THE_MODEL.strip())
    expect(notice).to_have_count(0)
    expect(_unsaved_marks(page)).to_have_count(0)
    assert log.read_text() == FROM_THE_MODEL, f"reloading wrote to the file: {log.read_text()!r}"


def test_a_save_the_terminal_refuses_keeps_the_text_and_says_so(page, folder):
    log = folder / "log.txt"
    open_file(file_browser(page), "log.txt")
    _type_at_the_end(page)
    expect(_unsaved_marks(page)).to_have_count(1)
    save = file_browser(page).get_by_role("button", name="Save", exact=True)
    # the folder turns read-only after the file was opened, so the terminal refuses the write
    folder.chmod(0o555)

    save.click()

    expect(
        page.get_by_text("Failed to save file. Your changes are still in the editor.")
    ).to_be_visible()
    expect(_editor(page)).to_contain_text(TYPED)
    expect(_unsaved_marks(page)).to_have_count(1)
    assert log.read_text() == WRITTEN, f"the refused save changed the file: {log.read_text()!r}"

    folder.chmod(0o755)
    save.click()
    assert wait_until(lambda: log.read_text() == WRITTEN + TYPED, 10), (
        f"the second save did not write the kept text: {log.read_text()!r}"
    )
    expect(_unsaved_marks(page)).to_have_count(0)


def test_a_file_opened_before_the_first_message_is_saved_after_it(page, folder, upstream):
    log = folder / "log.txt"
    open_file(file_browser(page), "log.txt")
    _say(page, upstream, "Noted.")
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]{36}"))
    _type_at_the_end(page)

    file_browser(page).get_by_role("button", name="Save", exact=True).click()

    refusal = page.get_by_text("File is not writable.")
    wait_until(lambda: log.read_text() == WRITTEN + TYPED or refusal.count() > 0, 10)
    assert log.read_text() == WRITTEN + TYPED, (
        "a file opened before the chat's first message was not saved after it; the file holds "
        f"{log.read_text()!r} and the page says {refusal.all_inner_texts()}"
    )
    expect(_unsaved_marks(page)).to_have_count(0)


def test_an_unsaved_edit_is_still_there_after_looking_at_another_tab(page, folder):
    log = folder / "log.txt"
    open_file(file_browser(page), "log.txt")
    _type_at_the_end(page)

    tab(page, folder.name).click()
    expect(file_browser(page).get_by_role("list")).to_be_visible()
    expect(_unsaved_marks(page)).to_have_count(1)
    tab(page, "log.txt").click()

    expect(_editor(page)).to_have_text(WRITTEN.strip() + TYPED)
    assert log.read_text() == WRITTEN, f"the edit was written without a save: {log.read_text()!r}"
    file_browser(page).get_by_role("button", name="Save", exact=True).click()
    assert wait_until(lambda: log.read_text() == WRITTEN + TYPED, 10), (
        f"the edit kept across the tabs was not saved: {log.read_text()!r}"
    )


def test_an_unsaved_edit_is_still_there_after_visiting_another_chat(page, person, folder, upstream):
    log = folder / "log.txt"
    with person.client() as client:
        seed_chat(
            client,
            [
                {"role": "user", "content": "an older question"},
                {"role": "assistant", "content": "An older answer."},
            ],
        )
    edited = _say(page, upstream, "Noted.")
    edited_chat = page.url
    open_file(file_browser(page), "log.txt")
    _type_at_the_end(page)

    page.get_by_role("button", name="Open Sidebar").click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    chat_row(sidebar, SEEDED_TITLE).click()
    expect_reply(page, "An older answer.")
    expect(tab(page, "log.txt")).to_have_count(0)
    chat_row(sidebar, edited).click()
    expect(page).to_have_url(edited_chat)

    expect(tab(page, "log.txt")).to_have_attribute("aria-selected", "true")
    expect(_editor(page), "the edit typed before the other chat was opened").to_have_text(
        WRITTEN.strip() + TYPED
    )
    expect(_unsaved_marks(page)).to_have_count(1)
    assert log.read_text() == WRITTEN, f"the edit was written without a save: {log.read_text()!r}"


def _workbook(rows: list[list]) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    for row in rows:
        book.active.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_a_workbook_changed_on_disk_since_it_was_opened_is_not_overwritten(page, folder):
    stock = folder / "stock.xlsx"
    stock.write_bytes(_workbook([["cask", "litres"], ["cider", 120]]))
    browser = file_browser(page)
    browser.get_by_role("button", name="Refresh", exact=True).first.click()
    open_file(browser, "stock.xlsx")
    grid = file_browser(page).get_by_role("grid", name="Worksheet")
    expect(grid.get_by_text("cider", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    grid.get_by_text("cider", exact=True).dblclick()
    cell = file_browser(page).get_by_role("textbox", name="Edit cell A2")
    cell.fill("perry")
    cell.press("Enter")
    expect(grid.get_by_text("perry", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    # someone else's version arrives on the terminal while the edit waits to be saved
    theirs = _workbook([["cask", "litres"], ["cider", 95]])
    stock.write_bytes(theirs)

    file_browser(page).get_by_role("button", name="Save", exact=True).click()

    expect(
        page.get_by_text(
            "The workbook changed on disk. Your edits are preserved. Reload the file before saving."
        )
    ).to_be_visible()
    assert stock.read_bytes() == theirs, "the save overwrote the version that arrived on disk"
    expect(grid.get_by_text("perry", exact=True)).to_be_visible()
    expect(_unsaved_marks(page)).to_have_count(1)
