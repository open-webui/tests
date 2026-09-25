"""Journey: a note is created, written in, still there after a reload, and deleted.

A fresh account creates a note from the Notes page, gives it a title and types into the editor.
The editor sends what is typed over the note's live document and the server stores it half a
second later, so the test waits for the stored note before reloading. The reloaded editor shows
the title and the text, and a note deleted from the list is gone from it after a reload too.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the live document's save handler
not writing the note the wait for the stored text times out, and with
`DELETE /api/v1/notes/{id}/delete` answering true without deleting the note stays in the list.
"""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE = "Grocery list"
TEXT = "buy oat milk and rye bread"


def _wait_until_stored(author, note_id: str, text: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    stored = ""
    with author.client() as client:
        while time.monotonic() < deadline:
            note = client.get(f"/api/v1/notes/{note_id}").json()
            stored = str((note.get("data") or {}).get("content"))
            if text in stored:
                return
            time.sleep(0.2)
    raise AssertionError(f"the note never stored {text!r}; it holds {stored}")


def _note_card(page: Page, title: str):
    return page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)


def test_a_written_note_survives_a_reload_and_a_deleted_one_is_gone(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/notes/[0-9a-f-]+$"))
    note_id = page.url.rsplit("/", 1)[1]

    editor_page = page.get_by_role("main")
    editor_page.get_by_role("textbox", name="Title").fill(TITLE)
    editor_page.get_by_label("Write something...").click()
    page.keyboard.type(TEXT)
    _wait_until_stored(author, note_id, TEXT)

    page.reload()
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(TITLE)
    expect(editor_page.get_by_label("Write something...")).to_have_text(TEXT)

    page.goto("/notes")
    card = _note_card(page, TITLE)
    card.get_by_role("button", name="Note Menu").first.click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    page.get_by_role("dialog", name="Delete note?").get_by_role("button", name="Confirm").click()
    expect(card).to_have_count(0)

    page.reload()
    expect(page.get_by_role("main").get_by_text("No Notes")).to_be_visible()
    expect(card).to_have_count(0)
