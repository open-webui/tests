"""Journey: opening the new-note address creates a note and lands in its editor.

Visiting `/notes/new` asks first (since 0ffd86967 a link no longer acts on its own), then makes the
note and replaces the address with the note's own. The note is titled with today's date unless the
address carries a title, and starts with the text of the address's content when it has one. The
saved note is listed on the Notes page and text typed into its editor is still there after a reload.

`test_text_typed_into_a_new_note_survives_a_reload` is red now and then: with keystrokes a few
milliseconds apart an older edit's save can replace the newest one, so the stored note stays a
few characters short (open-webui/open-webui#31585, fix PR #31596). Typed into 30 new notes, dev
7b7dba6ee stored 7 short and dev with #31596 applied none.

Discriminates: passes on dev 7b7dba6ee; in a frontend copy of dev 176d31d1d, with the route
ignoring the title and content of its address the listing and prefilled tests fail, and with the
route not creating a note all four fail.
"""

from __future__ import annotations

import re
import time
from datetime import date

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTE_URL = re.compile(r"/notes/[0-9a-f-]+$")
TITLE = "Trip packing"
PREFILLED_TEXT = "passport and charger"
TYPED_TEXT = "and a rain jacket"


def _stored_notes(author: Actor) -> list[dict]:
    with author.client() as client:
        return client.get("/api/v1/notes/").json()


def _wait_until_stored(author: Actor, note_id: str, text: str, timeout: float = 15.0) -> None:
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


def _confirm_dialog(page: Page) -> Locator:
    return page.get_by_role("dialog").filter(has_text="Create a new note")


def _open_new_note(page: Page, query: str = "") -> str:
    page.goto(f"/notes/new{query}")
    _confirm_dialog(page).get_by_role("button", name="Confirm").click()
    expect(page).to_have_url(NOTE_URL)
    return page.url.rsplit("/", 1)[1]


def test_new_note_address_lands_in_an_editor_for_a_note_titled_with_today(page_for, make_user):
    author = make_user()
    page = page_for(author)
    page.goto("/notes/new")
    expect(_confirm_dialog(page)).to_be_visible()
    assert _stored_notes(author) == [], "the address made a note before it was confirmed"

    _confirm_dialog(page).get_by_role("button", name="Confirm").click()
    expect(page).to_have_url(NOTE_URL)
    note_id = page.url.rsplit("/", 1)[1]

    editor_page = page.get_by_role("main")
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(date.today().isoformat())
    stored = _stored_notes(author)
    assert [note["id"] for note in stored] == [note_id]


def test_new_note_is_listed_on_the_notes_page(page_for, make_user):
    author = make_user()
    page = page_for(author)
    _open_new_note(page, f"?title={TITLE}")

    page.goto("/notes")
    card = page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=TITLE)
    expect(card).to_have_count(1)


def test_new_note_address_prefills_the_title_and_the_content(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _open_new_note(page, f"?title={TITLE}&content={PREFILLED_TEXT.replace(' ', '%20')}")

    editor_page = page.get_by_role("main")
    expect(editor_page.get_by_role("textbox", name="Title")).to_have_value(TITLE)
    expect(editor_page.get_by_label("Write something...")).to_have_text(PREFILLED_TEXT)
    with author.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    assert note["title"] == TITLE
    assert note["data"]["content"]["md"] == PREFILLED_TEXT


def test_text_typed_into_a_new_note_survives_a_reload(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _open_new_note(page)

    editor_page = page.get_by_role("main")
    editor_page.get_by_label("Write something...").click()
    page.keyboard.type(TYPED_TEXT)
    _wait_until_stored(author, note_id, TYPED_TEXT)

    page.reload()
    expect(editor_page.get_by_label("Write something...")).to_have_text(TYPED_TEXT)
