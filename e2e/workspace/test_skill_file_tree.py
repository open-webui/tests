"""Journey: a skill's file tree in the skill editor, from the first file to a saved edit.

A new skill is written in Workspace > Skills > Create: SKILL.md in the code editor and a second
file made with the file list's Actions > New File; Save & Create stores both and opens the saved
skill. Opening a saved skill lists its files with folders collapsed; a person edits a file, renames
one from its row menu, deletes another after confirming and saves with a description of the
change, which the server keeps as a new version with that message. A picture added with Actions >
Upload is saved with its bytes and shown as a picture. In SKILL.md's preview a link to another
file of the skill opens that file. When the skill was saved in another tab meanwhile, Save warns
that a newer version exists and keeps the draft, and Reload latest shows the other tab's work.

Discriminates: passes on the dev 178de3666 build. In a frontend copy, an editor that sends only
SKILL.md on create turns the create test red; one whose save keeps a renamed file under its old
name as well, or one that does not send deletions, turns the rename and delete test red; an upload
sent as text turns the upload test red; a preview that leaves relative links alone turns the
preview test red; and an editor that does not show a version conflict turns the conflict test
red.
"""

from __future__ import annotations

import base64
import re

import pytest
from playwright.sync_api import Page, expect

from harness.skill_files import (
    create,
    current,
    delete_skills_of,
    files,
    history,
    new_id,
    read,
    save,
    skill_md,
)
from utils.skill_editor import actions, code_editor, file_row, open_row, replace_text, row_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _open_skill(page: Page, skill_id: str) -> None:
    page.goto(f"/workspace/skills/edit?id={skill_id}")
    expect(code_editor(page)).to_be_visible()


def _save(page: Page, message: str = "") -> None:
    if message:
        page.get_by_role("textbox", name="Commit message").fill(message)
    page.get_by_role("button", name="Save", exact=True).click()


def _tide_skill(owner) -> dict:
    skill_id = new_id("tides")
    return create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Read [the tables](references/tables.md) first."),
            "references/tables.md": "High water at noon.\n",
            "references/charts.md": "Charts live here.\n",
        },
        skill_id=skill_id,
        name=f"Tides {skill_id}",
    )


def test_a_new_skill_is_saved_with_a_second_file_made_in_the_editor(page_for, owner):
    skill_id = new_id("knots")
    page = page_for(owner)
    page.goto("/workspace/skills/create")
    editor = page.get_by_role("main")
    editor.get_by_placeholder("Skill Name").fill(skill_id)
    expect(code_editor(page)).to_be_visible()
    replace_text(page, skill_md(skill_id, "Use the knot book."))

    actions(page, "New File")
    page.get_by_role("textbox", name="File name").fill("references/bowline.md")
    page.get_by_role("textbox", name="File name").press("Enter")
    expect(page.get_by_title("references/bowline.md", exact=True)).to_be_visible()
    replace_text(page, "Rabbit out of the hole.\n")
    page.get_by_role("button", name="Save & Create").click()

    expect(page).to_have_url(re.compile(rf"/workspace/skills/edit\?id={skill_id}$"))
    assert set(files(owner, skill_id)) == {"SKILL.md", "references/bowline.md"}
    assert read(owner, skill_id, "references/bowline.md") == b"Rabbit out of the hole.\n"
    assert b"Use the knot book." in read(owner, skill_id, "SKILL.md")


def test_an_edit_a_rename_and_a_delete_are_saved_as_one_version(page_for, owner):
    skill = _tide_skill(owner)
    page = page_for(owner)
    _open_skill(page, skill["id"])

    open_row(page, "references")
    open_row(page, "tables.md")
    expect(page.get_by_title("references/tables.md", exact=True)).to_be_visible()
    expect(code_editor(page)).to_contain_text("High water at noon.")
    replace_text(page, "High water at one.\n")
    row_menu(page, "tables.md", "Rename")
    rename = page.get_by_role("main").locator("input:focus")
    expect(rename).to_have_value("tables.md")
    rename.fill("tide-tables.md")
    rename.press("Enter")
    row_menu(page, "charts.md", "Delete")
    page.get_by_role("button", name="Confirm").click()
    expect(file_row(page, "charts.md")).to_have_count(0)
    _save(page, "Tidy the references")

    expect(page.get_by_text("Skill updated successfully")).to_be_visible()
    assert set(files(owner, skill["id"])) == {"SKILL.md", "references/tide-tables.md"}
    assert read(owner, skill["id"], "references/tide-tables.md") == b"High water at one.\n"
    versions = history(owner, skill["id"])
    assert len(versions) == 2
    assert "Tidy the references" in {entry["commit_message"] for entry in versions}


def test_an_uploaded_picture_is_saved_with_its_bytes_and_shown(page_for, owner):
    skill = _tide_skill(owner)
    page = page_for(owner)
    _open_skill(page, skill["id"])

    page.get_by_role("main").get_by_label("Actions", exact=True).click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Upload", exact=True).click()
    chooser.value.set_files(files=[{"name": "flag.png", "mimeType": "image/png", "buffer": PNG}])
    open_row(page, "flag.png")
    expect(page.get_by_role("img", name="flag.png")).to_be_visible()
    _save(page)

    expect(page.get_by_text("Skill updated successfully")).to_be_visible()
    assert read(owner, skill["id"], "flag.png") == PNG


def test_a_link_in_the_preview_opens_the_linked_file(page_for, owner):
    skill = _tide_skill(owner)
    page = page_for(owner)
    _open_skill(page, skill["id"])

    page.get_by_role("main").get_by_role("button", name="Preview").click()
    page.get_by_role("main").get_by_role("link", name="the tables").click()

    expect(page.get_by_title("references/tables.md", exact=True)).to_be_visible()
    expect(code_editor(page)).to_contain_text("High water at noon.")


def test_a_save_after_another_tab_saved_keeps_the_draft_and_offers_the_latest(page_for, owner):
    skill = _tide_skill(owner)
    page = page_for(owner)
    _open_skill(page, skill["id"])
    # another tab saves first
    other = save(
        owner,
        skill,
        [{"op": "put", "path": "references/tables.md", "content": "From the other tab.\n"}],
    )
    assert other.status_code == 200, other.text

    open_row(page, "references")
    open_row(page, "tables.md")
    replace_text(page, "From this tab.\n")
    _save(page)

    expect(page.get_by_role("alert")).to_contain_text("This skill has a newer version")
    expect(code_editor(page)).to_contain_text("From this tab.")
    assert read(owner, skill["id"], "references/tables.md") == b"From the other tab.\n"
    page.get_by_role("button", name="Reload latest").click()
    page.get_by_role("button", name="Confirm").click()
    open_row(page, "references")
    open_row(page, "tables.md")
    expect(code_editor(page)).to_contain_text("From the other tab.")
    assert current(owner, skill["id"])["version_id"] == other.json()["version_id"]
