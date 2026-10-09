"""Journey: a skill's version history in the editor, and skills brought in and sent out as files.

The skill editor's version picker lists the saved versions by their change description. Picking an
old one shows its files read only with Compare to current and Set as Production; the old version
takes no typing and opens no rename field; the comparison names each changed file with how it
changed and opens the first one's line diff, and setting it as production brings back its files,
which a `$` mention then sends to the model. An old version is deleted from its row in the version
picker after a confirmation, and the current version offers no delete. On the Skills page, Import
takes a ZIP of skill folders and opens a dialog listing each skill with its file count; Import
selected saves them with all their files. A skill that already exists is listed to be skipped, and
choosing Replace saves the upload as a new version of it. A row's Export ZIP downloads the skill's
folder with every file, and Clone opens a new skill holding the same files.

Discriminates: passes on the dev 178de3666 build. In a frontend copy, a code editor that ignores
the read-only flag (an old version takes typing) or a file row that starts a rename on double
click whatever it allows turns the version picker test red; a restore that
sends the current version in place of the picked one turns the restore test red; an import
dialog that offers no Replace turns the replace test red; and a row's Export ZIP that downloads
JSON turns the export ZIP test red. In a backend copy, a comparison that reports every file as
modified turns the version picker test red, a ZIP import that keeps only SKILL.md turns the
import test red, a clone that copies an empty SKILL.md alone turns the clone test red and an
export that leaves out the files turns the export JSON test red. Retargeted for 37138282f and
b130fec73, where the comparison became a panel of its own showing removed and added lines side by
side: the version picker test passes on b130fec73. Retargeted for 24ee1cb16, where an old version is
set as production in place of being restored as a copy: the set production test passes there and
goes red in a frontend build that sets the current version in place of the picked one; the version
delete test goes red in a build whose rows offer no delete.
"""

from __future__ import annotations

import io
import json
import re
import uuid
import zipfile

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
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
from utils.chat_ui import chat_input, expect_reply
from utils.skill_editor import code_editor, open_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


def _zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, text in entries.items():
            archive.writestr(path, text)
    return buffer.getvalue()


def _edited_twice(owner) -> tuple[dict, dict]:
    """A skill saved first as a draft and then rewritten; (first version, latest version)."""
    skill_id = new_id("tides")
    first = create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Read the tide table first."),
            "references/tables.md": "High water at noon.\n",
        },
        skill_id=skill_id,
        name=f"Tides {skill_id}",
        commit_message="First draft",
    )
    latest = save(
        owner,
        first,
        [
            {"op": "put", "path": "SKILL.md", "content": skill_md(skill_id, "Ignore the tides.")},
            {"op": "delete", "path": "references"},
        ],
        commit_message="Drop the tables",
    )
    assert latest.status_code == 200, latest.text
    return first, latest.json()


def _pick_version(page: Page, label: str) -> None:
    page.get_by_label("Select version", exact=True).click()
    page.get_by_role("menuitemradio", name=label).click()


def _skills_page_menu(page: Page, item: str) -> None:
    page.goto("/workspace/skills")
    page.get_by_label("Open create menu").click()
    page.get_by_role("button", name=item, exact=True).click()


def _row_menu(page: Page, name: str, item: str) -> None:
    page.goto("/workspace/skills")
    page.get_by_role("textbox", name="Search Skills").fill(name)
    row = page.get_by_role("main").get_by_role("button").filter(has_text=name)
    expect(row).to_be_visible()
    row.hover()
    row.get_by_role("button", name="Skill Menu").last.click()
    page.get_by_role("button", name=item, exact=True).click()


def test_the_version_picker_lists_saved_versions_and_compares_one_to_the_current(page_for, owner):
    first, _ = _edited_twice(owner)
    page = page_for(owner)
    page.goto(f"/workspace/skills/edit?id={first['id']}")
    expect(code_editor(page)).to_contain_text("Ignore the tides.")

    _pick_version(page, "First draft")
    expect(code_editor(page)).to_contain_text("Read the tide table first.")
    expect(page.get_by_role("main").get_by_text("Read Only", exact=True).first).to_be_visible()
    # an old version takes no typing and no renames
    code_editor(page).click()
    page.keyboard.type("Sneaky edit.")
    expect(code_editor(page)).not_to_contain_text("Sneaky edit.")
    open_row(page, "references")
    page.get_by_role("main").locator("li[data-file-row]").filter(has_text="tables.md").get_by_role(
        "button"
    ).first.dblclick()
    expect(code_editor(page)).to_contain_text("High water at noon.")
    expect(page.get_by_role("main").locator("li[data-file-row] input")).to_have_count(0)
    page.get_by_role("button", name="Compare to current").click()
    comparison = page.get_by_role("region", name="Compare to current")
    expect(comparison.get_by_role("button", name="references/tables.md")).to_contain_text("Deleted")
    # the first changed file opens on its own
    skill_file = comparison.get_by_role("button", name="SKILL.md")
    expect(skill_file).to_contain_text("Modified")
    expect(skill_file).to_have_attribute("aria-expanded", "true")

    # the changed line shows as removed and added, side by side
    removed = comparison.locator(".diff-cell.deletion .line-content")
    added = comparison.locator(".diff-cell.addition .line-content")
    expect(removed.filter(has_text="Read the tide table first.")).to_have_count(1)
    expect(added.filter(has_text="Ignore the tides.")).to_have_count(1)


def test_setting_an_old_version_as_production_brings_back_its_files_for_the_next_chat(
    page_for, owner, upstream
):
    first, latest = _edited_twice(owner)
    page = page_for(owner)
    page.goto(f"/workspace/skills/edit?id={first['id']}")
    expect(code_editor(page)).to_contain_text("Ignore the tides.")

    _pick_version(page, "First draft")
    page.get_by_role("button", name="Set as Production").click()

    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(page.get_by_role("button", name="Set as Production")).to_have_count(0)
    assert current(owner, first["id"])["version_id"] == first["version_id"]
    assert set(files(owner, first["id"])) == {"SKILL.md", "references/tables.md"}
    assert len(history(owner, first["id"])) == 2

    question = f"when is high water? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("At noon.", match=reply.answering(question)))
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("$tides")
    page.get_by_role("button").filter(has_text=first["name"]).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "At noon.")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    assert "Read the tide table first." in system
    assert "Ignore the tides." not in system


def test_a_zip_of_skill_folders_is_imported_through_the_dialog(page_for, owner):
    first, second = new_id("knots"), new_id("sails")
    archive = _zip(
        {
            f"{first}/SKILL.md": skill_md(first, "Tie a bowline."),
            f"{first}/references/bowline.md": "Rabbit out of the hole.\n",
            f"{second}/SKILL.md": skill_md(second, "Reef early."),
        }
    )
    page = page_for(owner)

    page.goto("/workspace/skills")
    page.get_by_label("Open create menu").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Import", exact=True).click()
    chooser.value.set_files(
        files=[{"name": "skills.zip", "mimeType": "application/zip", "buffer": archive}]
    )
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("textbox", name="Skill ID")).to_have_count(2)
    expect(dialog.get_by_text("2 files")).to_be_visible()
    dialog.get_by_role("button", name="Import selected").click()

    expect(dialog.get_by_text("saved")).to_have_count(2)
    assert set(files(owner, first)) == {"SKILL.md", "references/bowline.md"}
    assert read(owner, second, "SKILL.md") == skill_md(second, "Reef early.").encode()


def test_importing_an_existing_skill_offers_skip_and_replace_saves_a_new_version(page_for, owner):
    skill_id = new_id("moor")
    original = create(
        owner, {"SKILL.md": skill_md(skill_id, "Single lines.")}, skill_id=skill_id, name=skill_id
    )
    archive = _zip({f"{skill_id}/SKILL.md": skill_md(skill_id, "Double the lines.")})
    page = page_for(owner)

    page.goto("/workspace/skills")
    page.get_by_label("Open create menu").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Import", exact=True).click()
    chooser.value.set_files(
        files=[{"name": "moor.zip", "mimeType": "application/zip", "buffer": archive}]
    )
    dialog = page.get_by_role("dialog")
    action = dialog.get_by_role("combobox", name="Import action")
    expect(action).to_have_value("skip")
    expect(dialog.get_by_text("ID or name already exists")).to_be_visible()
    expect(dialog.get_by_role("button", name="Import selected")).to_be_disabled()
    action.select_option("replace")
    dialog.get_by_role("button", name="Import selected").click()

    expect(dialog.get_by_text("saved")).to_be_visible()
    assert b"Double the lines." in read(owner, skill_id, "SKILL.md")
    assert current(owner, skill_id)["version_id"] != original["version_id"]
    assert len(history(owner, skill_id)) == 2


def test_a_rows_export_zip_holds_the_skills_folder_with_every_file(page_for, owner):
    skill_id = new_id("charts")
    create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Plot the course."),
            "references/symbols.md": "Rocks are crosses.\n",
        },
        skill_id=skill_id,
        name=f"Charts {skill_id}",
    )
    page = page_for(owner)

    with page.expect_download() as download:
        _row_menu(page, f"Charts {skill_id}", "Export ZIP")
    with zipfile.ZipFile(download.value.path()) as archive:
        names = set(archive.namelist())
        symbols = archive.read(f"{skill_id}/references/symbols.md")

    assert names == {f"{skill_id}/SKILL.md", f"{skill_id}/references/symbols.md"}
    assert symbols == b"Rocks are crosses.\n"


def test_a_rows_export_json_holds_the_skill_with_its_files(page_for, owner):
    skill_id = new_id("charts")
    create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Plot the course."),
            "references/symbols.md": "Rocks are crosses.\n",
        },
        skill_id=skill_id,
        name=f"Charts {skill_id}",
    )
    page = page_for(owner)

    with page.expect_download() as download:
        _row_menu(page, f"Charts {skill_id}", "Export JSON")
    with open(download.value.path()) as saved:
        exported = json.load(saved)

    assert exported["id"] == skill_id
    assert {entry["path"]: entry["content"] for entry in exported["files"]}[
        "references/symbols.md"
    ] == ("Rocks are crosses.\n")


def test_clone_opens_a_new_skill_with_the_same_files(page_for, owner):
    skill_id = new_id("charts")
    create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Plot the course."),
            "references/symbols.md": "Rocks are crosses.\n",
        },
        skill_id=skill_id,
        name=f"Charts {skill_id}",
    )
    page = page_for(owner)

    _row_menu(page, f"Charts {skill_id}", "Clone")

    expect(page).to_have_url(re.compile(rf"/workspace/skills/edit\?id={skill_id}-\w+$"))
    clone_id = page.url.split("id=")[1]
    assert set(files(owner, clone_id)) == {"SKILL.md", "references/symbols.md"}
    assert read(owner, clone_id, "references/symbols.md") == b"Rocks are crosses.\n"


def test_an_old_version_is_deleted_from_the_version_picker(page_for, owner):
    first, latest = _edited_twice(owner)
    page = page_for(owner)
    page.goto(f"/workspace/skills/edit?id={first['id']}")
    expect(code_editor(page)).to_contain_text("Ignore the tides.")

    page.get_by_label("Select version", exact=True).click()
    current_row = page.get_by_role("menuitemradio", name="Current").locator("xpath=..")
    expect(current_row.get_by_label("More Options")).to_have_count(0)
    old_row = page.get_by_role("menuitemradio", name="First draft").locator("xpath=..")
    old_row.get_by_label("More Options").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("dialog", name="Delete Version").get_by_role("button", name="Delete").click()
    expect(page.get_by_text("Version deleted")).to_be_visible()

    assert [entry["id"] for entry in history(owner, first["id"])] == [latest["version_id"]]
    page.get_by_label("Select version", exact=True).click()
    expect(page.get_by_role("menuitemradio", name="First draft")).to_have_count(0)
