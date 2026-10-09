"""Journey: two people writing in the same note at the same time, each in their own browser.

The owner and an account with write access open one note. What one types shows up in the other's
editor without a reload, both edits are still there after both pages reload and are stored in the
note, and typing in both pages at once ends with the same text in both. An account that may only
read the note opens it in an editor that takes no typing; a read-only editor does not join the
live document (upstream 5078d987f), so a reader who kept the note open sees the writing once they
reload.

`test_a_reader_with_the_note_open_sees_the_writing_after_a_reload` failed on dev 1c010b438 in CI:
the note was saved a few characters short (open-webui/open-webui#31585, fix PR #31596 open).

Discriminates: passes on dev 176d31d1d. In a frontend copy, the editor not applying remote Yjs
updates turns the three writer tests red and the editor staying editable for a reader turns the
reader typing test red. In a backend copy, `ydoc:document:update` not broadcasting to the room
turns the same three red, and the live document's save handler not writing the note turns the
stored-edits, typing-at-once and reader reload tests red.
"""

from __future__ import annotations

import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _create_shared_note(owner: Actor, other: Actor, permissions: list[str]) -> str:
    grants = [grant("user", other.id, permission) for permission in permissions]
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Live {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": ""}},
                "access_grants": grants,
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_markdown(owner: Actor, note_id: str) -> str:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    return str(((note.get("data") or {}).get("content") or {}).get("md"))


def _wait_until_stored(owner: Actor, note_id: str, texts: list[str], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    stored = _stored_markdown(owner, note_id)
    while not all(text in stored for text in texts) and time.monotonic() < deadline:
        time.sleep(0.2)
        stored = _stored_markdown(owner, note_id)
    assert all(text in stored for text in texts), f"the note holds {stored!r}, not {texts}"


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def _open_note(page: Page, note_id: str, editable: bool = True) -> Locator:
    page.goto(f"/notes/{note_id}")
    editor = _note_editor(page)
    expect(editor).to_have_attribute("contenteditable", "true" if editable else "false")
    return editor


def _document_text(editor: Locator) -> str:
    """The editor's text without the other writer's cursor label."""
    return editor.evaluate(
        """element => {
            const copy = element.cloneNode(true);
            copy.querySelectorAll('.ProseMirror-yjs-cursor').forEach(node => node.remove());
            return copy.textContent.replace(/[\\u2060\\ufeff]/g, '');
        }"""
    )


def _type_line(page: Page, editor: Locator, text: str) -> None:
    editor.click()
    page.keyboard.press("ControlOrMeta+End")
    page.keyboard.type(text)


def _two_writers(page_for, make_user) -> tuple[Actor, Actor, str, Page, Page]:
    owner, writer = make_user(), make_user()
    note_id = _create_shared_note(owner, writer, ["read", "write"])
    owner_page, writer_page = page_for(owner), page_for(writer)
    _open_note(owner_page, note_id)
    _open_note(writer_page, note_id)
    return owner, writer, note_id, owner_page, writer_page


def test_each_writer_sees_the_others_typing_appear(page_for, make_user):
    _, _, _, owner_page, writer_page = _two_writers(page_for, make_user)

    _type_line(owner_page, _note_editor(owner_page), "from the owner")
    expect(_note_editor(writer_page)).to_contain_text("from the owner")
    _type_line(writer_page, _note_editor(writer_page), " and the writer")

    expect(_note_editor(owner_page)).to_contain_text("from the owner and the writer")
    expect(_note_editor(writer_page)).to_contain_text("from the owner and the writer")


def test_both_edits_survive_a_reload_and_are_stored(page_for, make_user):
    owner, _, note_id, owner_page, writer_page = _two_writers(page_for, make_user)
    _type_line(owner_page, _note_editor(owner_page), "owner line")
    expect(_note_editor(writer_page)).to_contain_text("owner line")
    _type_line(writer_page, _note_editor(writer_page), " writer line")
    expect(_note_editor(owner_page)).to_contain_text("owner line writer line")
    _wait_until_stored(owner, note_id, ["owner line writer line"])

    owner_page.reload()
    writer_page.reload()

    expect(_note_editor(owner_page)).to_have_text("owner line writer line")
    expect(_note_editor(writer_page)).to_have_text("owner line writer line")


def test_typing_in_both_pages_at_once_ends_with_the_same_text(page_for, make_user):
    owner, _, note_id, owner_page, writer_page = _two_writers(page_for, make_user)
    owner_editor, writer_editor = _note_editor(owner_page), _note_editor(writer_page)
    owner_editor.click()
    writer_editor.click()
    typed = [f"{who}{index}" for index in range(6) for who in "ow"]

    for index in range(6):
        owner_page.keyboard.type(f"o{index} ")
        writer_page.keyboard.type(f"w{index} ")

    for text in typed:
        expect(owner_editor).to_contain_text(text)
        expect(writer_editor).to_contain_text(text)
    _wait_until_stored(owner, note_id, typed)
    owner_page.reload()
    writer_page.reload()
    expect(owner_editor).to_contain_text(typed[-1])
    expect(writer_editor).to_contain_text(typed[-1])
    assert _document_text(owner_editor) == _document_text(writer_editor)
    assert all(text in _document_text(owner_editor) for text in typed)


def test_a_reader_cannot_type_into_the_note(page_for, make_user):
    owner, reader = make_user(), make_user()
    note_id = _create_shared_note(owner, reader, ["read"])
    page = page_for(reader)

    editor = _open_note(page, note_id, editable=False)
    editor.click()
    page.keyboard.type("let me in")

    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_be_visible()
    expect(editor).to_have_text("")
    assert _stored_markdown(owner, note_id) == ""


def test_a_reader_with_the_note_open_sees_the_writing_after_a_reload(page_for, make_user):
    owner, reader = make_user(), make_user()
    note_id = _create_shared_note(owner, reader, ["read"])
    owner_page, reader_page = page_for(owner), page_for(reader)
    _open_note(owner_page, note_id)
    _open_note(reader_page, note_id, editable=False)

    _type_line(owner_page, _note_editor(owner_page), "written while you watch")
    _wait_until_stored(owner, note_id, ["written while you watch"])
    reader_page.reload()

    reader_editor = _note_editor(reader_page)
    expect(reader_editor).to_have_text("written while you watch")
    expect(reader_editor).to_have_attribute("contenteditable", "false")
