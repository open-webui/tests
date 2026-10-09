"""Journey: reading, comparing and reworking a prompt's saved versions in the prompt editor.

The editor's version picker lists the production version and every older one by its change
description. Picking an old one shows its text read only with its short id, and Production brings
the current text back for editing. Compare to current opens a comparison naming the changed name
and tags with their old and new values and the prompt text as removed and added lines side by
side, and Close returns to the old text. The Unified layout shows those lines in one column and is
kept for the next comparison; a version that differs only in its line endings says so. Edit as new
version makes the old text editable and saves it as a further version that is not production,
leaving what the chat inserts unchanged. Switching versions with unsaved typing asks whether to
discard it first, and so does leaving the editor. Ctrl+S saves the text. A user the prompt is
shared with read only can pick and compare its versions but is offered no way to change them.

Discriminates: passes on dev b130fec73. In a frontend build where an old version's text box takes
typing the read-only test goes red; where the comparison leaves out the prompt text the comparison
test goes red; where Edit as new version saves as production the edit test goes red; where switching
versions never asks about unsaved typing the discard test goes red; where Ctrl+S does nothing the
shortcut test goes red; where leaving the editor never asks the leave test goes red; where an old
version offers its write buttons whatever the access the reader test goes red; where the chosen
layout is not remembered the layout test goes red; and where a change of line endings alone counts
as no change the line endings test goes red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

THREE_LINES = "Summarise this in three lines."
ONE_LINE = "Summarise this in one line."


@pytest.fixture
def writer(make_user):
    """A fresh admin; the prompts it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for prompt in client.get("/api/v1/prompts/").json():
            client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")


def _post(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        response = client.post(path, json=form)
    assert response.status_code == 200, response.text
    return response.json()


def _get(account: Actor, path: str):
    with account.client() as client:
        response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


def _edited_once(account: Actor) -> tuple[dict, dict]:
    """A prompt saved as "First draft" and then renamed and shortened; (first, latest)."""
    command = f"brief{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": f"Brief {command}", "content": THREE_LINES}
    first = _post(
        account,
        "/api/v1/prompts/create",
        {**form, "tags": ["notes"], "commit_message": "First draft"},
    )
    latest = _post(
        account,
        f"/api/v1/prompts/id/{first['id']}/update",
        {
            **form,
            "name": f"Short brief {command}",
            "content": ONE_LINE,
            "tags": ["notes", "short"],
            "commit_message": "Shorter",
        },
    )
    return first, latest


def _content(page: Page) -> Locator:
    return page.get_by_role("textbox", name="Prompt Content")


def _pick_version(page: Page, label: str) -> None:
    page.get_by_label("Select version", exact=True).click()
    page.get_by_role("menuitemradio", name=label).click()


def _change(comparison: Locator, label: str) -> Locator:
    """The old and new value shown under a changed field's label."""
    row = comparison.get_by_text(label, exact=True).locator("xpath=..")
    return row.locator(".grid > div")


def _removed(comparison: Locator) -> Locator:
    """The lines the comparison shows as taken out of the old version."""
    return comparison.locator(".diff-cell.deletion .line-content")


def _added(comparison: Locator) -> Locator:
    return comparison.locator(".diff-cell.addition .line-content")


def _open(page: Page, prompt_id: str, text: str) -> None:
    page.goto(f"/workspace/prompts/{prompt_id}")
    expect(_content(page)).to_have_value(text)


def test_an_old_version_opens_read_only_and_production_brings_back_the_current_text(
    page_for, writer
):
    first, _ = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)

    _pick_version(page, "First draft")
    expect(_content(page)).to_have_value(THREE_LINES)
    expect(_content(page)).not_to_be_editable()
    expect(page.get_by_text(first["version_id"][:7], exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Save", exact=True)).to_have_count(0)

    _pick_version(page, "Production")
    expect(_content(page)).to_have_value(ONE_LINE)
    expect(_content(page)).to_be_editable()
    expect(page.get_by_role("button", name="Save", exact=True)).to_be_visible()


def test_compare_to_current_shows_the_changed_name_tags_and_text(page_for, writer):
    first, latest = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)

    _pick_version(page, "First draft")
    page.get_by_role("button", name="Compare to current").click()
    comparison = page.get_by_role("region", name="Compare to current")
    expect(comparison).to_contain_text(first["version_id"][:7])
    expect(comparison).to_contain_text(latest["version_id"][:7])
    expect(_change(comparison, "Name")).to_have_text([first["name"], latest["name"]])
    expect(_change(comparison, "Tags")).to_have_text(["notes", "notes, short"])
    expect(_removed(comparison)).to_have_text([THREE_LINES])
    expect(_added(comparison)).to_have_text([ONE_LINE])

    comparison.get_by_role("button", name="Close").click()
    expect(comparison).to_have_count(0)
    expect(_content(page)).to_have_value(THREE_LINES)


def test_edit_as_new_version_saves_a_version_that_is_not_production(page_for, writer):
    first, latest = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)

    _pick_version(page, "First draft")
    page.get_by_role("button", name="Edit as new version").click()
    expect(page.get_by_label("Select version", exact=True)).to_contain_text("Editing")
    expect(page.get_by_role("checkbox", name="Set as Production")).not_to_be_checked()
    _content(page).fill("Summarise this in two lines.")
    page.get_by_role("textbox", name="Commit Message").fill("Middle ground")
    page.get_by_role("button", name="Save", exact=True).click()

    expect(page.get_by_label("Select version", exact=True)).to_have_text("Middle ground")
    expect(_content(page)).to_have_value("Summarise this in two lines.")
    expect(_content(page)).not_to_be_editable()
    stored = _get(writer, f"/api/v1/prompts/id/{first['id']}")
    assert stored["version_id"] == latest["version_id"]
    assert stored["content"] == ONE_LINE
    history = _get(writer, f"/api/v1/prompts/id/{first['id']}/history")
    saved = [entry for entry in history if entry["commit_message"] == "Middle ground"]
    assert [entry["snapshot"]["content"] for entry in saved] == ["Summarise this in two lines."]


def test_switching_versions_with_unsaved_typing_asks_to_discard_it(page_for, writer):
    first, _ = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)
    _content(page).fill("Summarise this in one word.")

    _pick_version(page, "First draft")
    confirm = page.get_by_role("dialog", name="Discard unsaved changes?")
    confirm.get_by_role("button", name="Cancel").click()
    expect(confirm).to_have_count(0)
    expect(_content(page)).to_have_value("Summarise this in one word.")

    _pick_version(page, "First draft")
    confirm.get_by_role("button", name="Discard").click()
    expect(_content(page)).to_have_value(THREE_LINES)
    assert _get(writer, f"/api/v1/prompts/id/{first['id']}")["content"] == ONE_LINE


def test_ctrl_s_saves_the_prompt_text(page_for, writer):
    first, _ = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)

    _content(page).fill("Summarise this in one word.")
    _content(page).press("Control+s")

    expect(page.get_by_text("Prompt updated successfully")).to_be_visible()
    stored = _get(writer, f"/api/v1/prompts/id/{first['id']}")
    assert stored["content"] == "Summarise this in one word."


def test_leaving_the_editor_with_unsaved_typing_asks_first(page_for, writer):
    first, _ = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)
    _content(page).fill("Summarise this in one word.")
    asked: list[str] = []

    def answer(accept: bool):
        def handle(dialog):
            asked.append(dialog.message)
            dialog.accept() if accept else dialog.dismiss()

        return handle

    page.once("dialog", answer(False))
    page.get_by_role("button", name="Back", exact=True).click()
    expect(_content(page)).to_have_value("Summarise this in one word.")
    assert asked == ["Discard unsaved changes?"]

    page.once("dialog", answer(True))
    page.get_by_role("button", name="Back", exact=True).click()
    expect(page).to_have_url(re.compile(r"/workspace/prompts$"))
    assert asked == ["Discard unsaved changes?"] * 2
    assert _get(writer, f"/api/v1/prompts/id/{first['id']}")["content"] == ONE_LINE


def test_a_reader_can_compare_versions_but_not_change_them(page_for, writer, admin, make_user):
    first, _ = _edited_once(writer)
    reader = make_user()
    make_group(admin, [reader], {"workspace": {"prompts": True}})
    _post(
        writer,
        f"/api/v1/prompts/id/{first['id']}/access/update",
        {"access_grants": [grant("user", reader.id, "read")]},
    )
    page = page_for(reader)
    _open(page, first["id"], ONE_LINE)

    _pick_version(page, "First draft")
    expect(_content(page)).to_have_value(THREE_LINES)
    page.get_by_role("button", name="Compare to current").click()
    expect(_added(page.get_by_role("region", name="Compare to current"))).to_have_text([ONE_LINE])
    expect(page.get_by_role("button", name="Edit as new version")).to_have_count(0)
    expect(page.get_by_role("button", name="Set as Production")).to_have_count(0)
    expect(page.get_by_role("button", name="Save", exact=True)).to_have_count(0)


def _compare(page: Page, label: str) -> Locator:
    _pick_version(page, label)
    page.get_by_role("button", name="Compare to current").click()
    return page.get_by_role("region", name="Compare to current")


def test_the_unified_layout_shows_the_change_in_one_column_and_is_kept(page_for, writer):
    first, _ = _edited_once(writer)
    page = page_for(writer)
    _open(page, first["id"], ONE_LINE)

    comparison = _compare(page, "First draft")
    layout = comparison.get_by_label("Diff layout", exact=True)
    expect(layout).to_have_text("Split")
    expect(comparison.locator(".split-row")).not_to_have_count(0)
    layout.click()
    page.get_by_role("button", name="Unified", exact=True).click()
    expect(layout).to_have_text("Unified")
    expect(comparison.locator(".split-row")).to_have_count(0)
    expect(comparison.locator(".unified-row.deletion .line-content")).to_have_text([THREE_LINES])
    expect(comparison.locator(".unified-row.addition .line-content")).to_have_text([ONE_LINE])

    _open(page, first["id"], ONE_LINE)
    comparison = _compare(page, "First draft")
    expect(comparison.get_by_label("Diff layout", exact=True)).to_have_text("Unified")
    expect(comparison.locator(".split-row")).to_have_count(0)


def test_a_version_that_differs_only_in_line_endings_says_so(page_for, writer):
    command = f"crlf{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": f"Endings {command}"}
    first = _post(
        writer,
        "/api/v1/prompts/create",
        {**form, "content": "Line one\nLine two", "commit_message": "Unix endings"},
    )
    _post(
        writer,
        f"/api/v1/prompts/id/{first['id']}/update",
        {**form, "content": "Line one\r\nLine two", "commit_message": "Windows endings"},
    )
    page = page_for(writer)
    _open(page, first["id"], "Line one\nLine two")

    comparison = _compare(page, "Unix endings")
    expect(comparison).to_contain_text("Only line endings changed")
    expect(comparison).not_to_contain_text("No differences")
