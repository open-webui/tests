"""Regression: a large paste into a note was gone after a reload.

Issue open-webui/open-webui#26140, fix 914ff1f11 (PR open-webui/open-webui#31893). The note editor
sends every edit over the socket with the whole note in it several times over (the Yjs update
and the markdown, HTML and JSON of the editor). Pasting a couple of hundred KB of text made that
message larger than the server's 1 MB cap, so the live connection dropped and reconnected and
the paste was never saved: the editor showed it until the page was reloaded. The server now
takes messages up to 16 MiB. Twin of integration/notes/test_large_note_edits.py.

The text goes through the browser clipboard and Control+V, as a person pastes it.

Discriminates: passes on dev b859124f9, fails on that backend with 914ff1f11 reverted (the paste
is never stored and the reloaded note is empty).
"""

from __future__ import annotations

import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.clipboard import write_text

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]
LARGE_TEXT = " ".join(f"word{number}" for number in range(25_000))
LAST_WORD = "word24999"
SAVE_WAIT = 20.0


def _create_note(owner: Actor) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"Pasted {uuid.uuid4().hex[:6]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_markdown(owner: Actor, note_id: str) -> str:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    return str(((note.get("data") or {}).get("content") or {}).get("md") or "")


def _wait_until_stored(owner: Actor, note_id: str) -> str:
    deadline = time.monotonic() + SAVE_WAIT
    stored = _stored_markdown(owner, note_id)
    while LAST_WORD not in stored and time.monotonic() < deadline:
        time.sleep(0.3)
        stored = _stored_markdown(owner, note_id)
    return stored


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def test_a_large_paste_into_a_note_is_still_there_after_a_reload(page_for, make_user):
    owner = make_user()
    note_id = _create_note(owner)
    page = page_for(owner, path=f"/notes/{note_id}", permissions=CLIPBOARD)
    editor = _note_editor(page)
    expect(editor).to_have_attribute("contenteditable", "true")

    write_text(page, LARGE_TEXT)
    editor.click()
    page.keyboard.press("Control+V")
    expect(editor).to_contain_text(LAST_WORD)

    stored = _wait_until_stored(owner, note_id)
    assert LAST_WORD in stored, f"#26140: the note holds {len(stored)} characters after the paste"
    page.reload()
    expect(_note_editor(page)).to_contain_text(LAST_WORD)
