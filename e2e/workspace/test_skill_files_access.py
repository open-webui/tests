"""Journey: the skill file tree and version history, as a reader, writer and stranger meet them.

A skill with SKILL.md and references/checklist.md has two versions. A reader (read grant only)
opens its editor, sees the tree, opens the checklist and reads it, but is told it is read only:
no Actions menu with New File, New Folder or Upload, no Rename or Delete on a file, a file
editor that takes no typing, and no "Restore as new version" on an older version. A writer adds
a file with New File and the server lists it, and restores an older version so the checklist is
back to its first text. A stranger who opens the edit URL sees neither the instructions nor the
file names. On the Skills list, Import and Export entries follow the skills_import and
skills_export permissions, in the list's create menu and in a skill's menu.

Discriminates: passes on dev 178de3666. In a frontend copy where SkillEditor passes
`readOnly={false}` to SkillFiles and always offers "Restore as new version", the reader test
goes red (Actions menu, typing and Restore appear). Where the edit page renders the skill
whatever the server answers, the stranger test goes red. Where the import entries are visible
to everyone, the import test goes red, and where the export entries are visible to everyone,
the export test goes red. In a backend copy whose restore keeps the current files, the writer
test goes red (the checklist keeps its second text).
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

INSTRUCTIONS = "Review the change before approving it."
FIRST_CHECKLIST = "first checklist text"
SECOND_CHECKLIST = "second checklist text"


@pytest.fixture
def grant_permissions(admin) -> Iterator[Callable[..., None]]:
    """`grant_permissions(account, **workspace)` puts the account in a group with those rights."""
    made: list[str] = []

    def apply(account: Actor, **workspace: bool) -> None:
        permissions = {"workspace": {"skills": True, **workspace}}
        made.append(make_group(admin, [account], permissions=permissions))

    yield apply
    with admin.client() as client:
        for group_id in made:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


@pytest.fixture
def skill_owner(make_user, grant_permissions) -> Iterator[Actor]:
    """A user who may write skills; the skills it made are deleted afterwards."""
    account = make_user()
    grant_permissions(account)
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _skill_with_two_versions(owner: Actor, shared_with: list[tuple[Actor, list[str]]]) -> dict:
    """A skill with two versions, shared as given; returns its id, name and first version."""
    skill_id = f"review-{uuid.uuid4().hex[:8]}"
    name = f"Review {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": name,
                "description": "reviewing changes",
                "files": [
                    {"path": "SKILL.md", "content": INSTRUCTIONS},
                    {"path": "references/checklist.md", "content": FIRST_CHECKLIST},
                ],
            },
        )
        assert created.status_code == 200, created.text
        first_version = created.json()["version_id"]
        updated = client.post(
            f"/api/v1/skills/id/{skill_id}/update",
            json={
                "id": skill_id,
                "name": name,
                "expected_version_id": first_version,
                "operations": [
                    {"op": "put", "path": "references/checklist.md", "content": SECOND_CHECKLIST}
                ],
            },
        )
        assert updated.status_code == 200, updated.text
        grants = [
            grant("user", account.id, permission)
            for account, permissions in shared_with
            for permission in permissions
        ]
        shared = client.post(
            f"/api/v1/skills/id/{skill_id}/access/update", json={"access_grants": grants}
        )
        assert shared.status_code == 200, shared.text
    return {"id": skill_id, "name": name, "first_version": first_version}


def _server_checklist(actor: Actor, skill_id: str) -> str:
    with actor.client() as client:
        listed = client.get(f"/api/v1/skills/id/{skill_id}/files").json()
        response = client.get(
            f"/api/v1/skills/id/{skill_id}/files/content",
            params={"path": "references/checklist.md", "version_id": listed["version_id"]},
        )
    assert response.status_code == 200, response.text
    return response.text


def _server_paths(actor: Actor, skill_id: str) -> set[str]:
    with actor.client() as client:
        response = client.get(f"/api/v1/skills/id/{skill_id}/files")
    assert response.status_code == 200, response.text
    return {entry["path"] for entry in response.json()["files"]}


def _open_checklist(page: Page, skill: dict) -> Locator:
    page.goto(f"/workspace/skills/edit?id={skill['id']}")
    main = page.get_by_role("main")
    main.get_by_role("button", name=re.compile(r"^references")).click()
    main.get_by_role("button", name=re.compile(r"^checklist\.md")).first.click()
    return main


def _choose_first_version(page: Page, skill: dict) -> None:
    main = page.get_by_role("main")
    main.get_by_role("button", name="Select version").first.click()
    page.get_by_role("menuitemradio").filter(has_text=skill["first_version"][:7]).click()
    expect(main.get_by_text("Compare to current")).to_be_visible()


def test_a_reader_reads_the_files_but_meets_a_read_only_editor(
    page_for, skill_owner, make_user, grant_permissions
):
    reader = make_user()
    grant_permissions(reader)
    skill = _skill_with_two_versions(skill_owner, [(reader, ["read"])])
    page = page_for(reader)

    main = _open_checklist(page, skill)

    expect(main.get_by_text("Read Only", exact=True)).to_be_visible()
    expect(main.get_by_text(SECOND_CHECKLIST)).to_be_visible()
    expect(main.get_by_role("button", name=re.compile(r"^SKILL\.md"))).to_be_visible()
    expect(main.get_by_label("Actions")).to_have_count(0)
    expect(main.get_by_text("New File")).to_have_count(0)
    expect(main.get_by_role("button", name="Save", exact=True)).to_have_count(0)

    row = main.get_by_role("listitem").filter(has_text="checklist.md")
    row.get_by_role("button", name="More").first.click()
    expect(page.get_by_role("button", name="Rename")).to_be_disabled()
    expect(page.get_by_role("button", name="Delete")).to_be_disabled()
    page.keyboard.press("Escape")

    editor = main.get_by_role("textbox").filter(has_text=SECOND_CHECKLIST)
    editor.click()
    page.keyboard.type("sneaky addition")
    page.keyboard.press("ControlOrMeta+s")
    expect(main.get_by_text("sneaky addition")).to_have_count(0)
    assert _server_checklist(skill_owner, skill["id"]) == SECOND_CHECKLIST

    _choose_first_version(page, skill)
    expect(main.get_by_text("Restore as new version")).to_have_count(0)


def test_a_writer_adds_a_file_and_restores_an_older_version(
    page_for, skill_owner, make_user, grant_permissions
):
    writer = make_user()
    grant_permissions(writer)
    skill = _skill_with_two_versions(skill_owner, [(writer, ["read", "write"])])
    page = page_for(writer)
    main = _open_checklist(page, skill)

    expect(main.get_by_text("Read Only", exact=True)).to_have_count(0)
    main.get_by_label("Actions").click()
    page.get_by_role("button", name="New File").click()
    main.get_by_role("textbox", name="File name").fill("notes.md")
    main.get_by_role("textbox", name="File name").press("Enter")
    expect(main.get_by_role("button", name=re.compile(r"^notes\.md"))).to_be_visible()
    main.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Skill updated successfully")).to_be_visible()
    assert "references/notes.md" in _server_paths(skill_owner, skill["id"])

    _choose_first_version(page, skill)
    main.get_by_text("Restore as new version").click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    assert _server_checklist(skill_owner, skill["id"]) == FIRST_CHECKLIST


def test_a_stranger_sees_nothing_of_the_skill(page_for, skill_owner, make_user, grant_permissions):
    stranger = make_user()
    grant_permissions(stranger)
    skill = _skill_with_two_versions(skill_owner, [])
    page = page_for(stranger)

    page.goto(f"/workspace/skills/edit?id={skill['id']}")

    expect(page).to_have_url(re.compile(r"/workspace/skills$"))
    expect(page.get_by_text(INSTRUCTIONS)).to_have_count(0)
    expect(page.get_by_text("checklist.md")).to_have_count(0)
    expect(page.get_by_text("references")).to_have_count(0)


def _create_menu_entries(page: Page) -> Locator:
    page.goto("/workspace/skills")
    expect(
        page.get_by_role("main").get_by_role("button", name="Create", exact=True)
    ).to_be_visible()
    return page.get_by_label("Open create menu")


def _open_create_menu(page: Page) -> None:
    _create_menu_entries(page).click()


def test_import_entries_follow_the_skills_import_permission(
    page_for, admin, make_user, grant_permissions
):
    importer, plain = make_user(), make_user()
    grant_permissions(importer, skills_import=True)
    grant_permissions(plain)

    importer_page = page_for(importer)
    _open_create_menu(importer_page)
    expect(importer_page.get_by_text("Import", exact=True)).to_be_visible()
    expect(importer_page.get_by_text("Import folder")).to_be_visible()

    plain_page = page_for(plain)
    expect(_create_menu_entries(plain_page)).to_have_count(0)
    expect(plain_page.get_by_text("Import")).to_have_count(0)

    admin_page = page_for(admin)
    _open_create_menu(admin_page)
    expect(admin_page.get_by_text("Import", exact=True)).to_be_visible()
    expect(admin_page.get_by_text("Import folder")).to_be_visible()


def test_export_entries_follow_the_skills_export_permission(
    page_for, admin, make_user, grant_permissions
):
    exporter, plain = make_user(), make_user()
    grant_permissions(exporter, skills_export=True)
    grant_permissions(plain)

    for account, allowed in ((exporter, True), (plain, False), (admin, True)):
        skill = _skill_with_two_versions(account, [])
        expected = 1 if allowed else 0
        try:
            page = page_for(account)
            _create_menu_entries(page)
            if allowed:
                page.get_by_label("Open create menu").click()
            expect(page.get_by_text("Export ZIP")).to_have_count(expected)
            expect(page.get_by_text("Export JSON")).to_have_count(expected)
            if allowed:
                page.keyboard.press("Escape")

            page.get_by_placeholder("Search Skills").fill(skill["name"])
            expect(page.get_by_text(skill["name"])).to_be_visible()
            page.get_by_label("Skill Menu").click()
            expect(page.get_by_role("button", name="Clone")).to_be_visible()
            expect(page.get_by_role("button", name="Export ZIP")).to_have_count(expected)
            expect(page.get_by_role("button", name="Export JSON")).to_have_count(expected)
        finally:
            with account.client() as client:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")
