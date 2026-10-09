"""Journey: the Access entry in a skill, tool or note's "..." menu shares the item from its list.

The entry opens an Access dialog that loads the item's grants and saves every change at once with
a "Saved" toast. For a skill an owner adds a person at Read, and the server then lets them read it
and refuses their update; raising the row to Write lets them update it, and the row's X takes both
away again. For a tool the same goes through a group. For a note the owner shares from the Notes
list and the person opens it over the API until the X removes them; a person given write access
to a note is not offered Access in its menu. Visibility offers Public only to an account with the
public sharing permission for that kind: without it the choice is missing, with it the skill
becomes readable by an unrelated account. A reader of a skill or tool has no menu on the row, and
a writer is offered Access.

Discriminates: passes on the dev 206bf9723 build; in a frontend copy the skill and note dialogs
saving nothing turn the skill test (the person still cannot read it), the note test and the public
test red (nothing is stored); Tools.svelte passing no accessHandler turns the tool test and the
tools case of the reader test red (no Access entry); Notes.svelte offering Access to every note
turns the write-access note test red (the entry is shown); sharePublic always true turns the
no-public test red (Public is offered); Skills.svelte showing the menu to every row turns the
skills case of the reader test red (a reader has a menu).
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'
SKILLS_WORKSPACE = {"workspace": {"skills": True}}
WORKSPACE_USER = {"workspace": {"skills": True, "tools": True}}


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def owner(make_user) -> Iterator[Actor]:
    """A fresh admin; its skills, tools and notes are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")
        for note in client.get("/api/v1/notes/").json():
            if note["user_id"] == account.id:
                client.delete(f"/api/v1/notes/{note['id']}/delete")


@pytest.fixture
def group_name_of(admin) -> Iterator[Callable[..., tuple[str, str]]]:
    """`group_name_of(*members)` is a new group's (id, name); the groups are deleted afterwards."""
    made: list[str] = []

    def create(*members: Actor) -> tuple[str, str]:
        group_id = make_group(admin, list(members), permissions=WORKSPACE_USER)
        made.append(group_id)
        with admin.client() as client:
            name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
        return group_id, name

    yield create
    with admin.client() as client:
        for group_id in made:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _new_skill(account: Actor, instructions: str = "Tie a bowline.") -> dict:
    skill = {"id": f"menu-skill-{uuid.uuid4().hex[:8]}", "name": _unique("Menu skill")}
    with account.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={**skill, "description": "knots", "content": instructions, "meta": {}},
        )
    assert created.status_code == 200, created.text
    return skill


def _new_tool(account: Actor) -> dict:
    tool = {"id": f"menu_tool_{uuid.uuid4().hex[:8]}", "name": _unique("Menu tool")}
    with account.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={**tool, "content": TOOL_SOURCE, "meta": {"description": "pings"}},
        )
    assert created.status_code == 200, created.text
    return tool


def _new_note(account: Actor, title: str) -> str:
    with account.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": "agenda"}}, "access_grants": []},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _status(account: Actor, method: str, path: str, body: dict | None = None) -> int:
    with account.client() as client:
        return client.request(method, path, json=body).status_code


def _read_skill(account: Actor, skill: dict) -> int:
    return _status(account, "GET", f"/api/v1/skills/id/{skill['id']}")


def _update_skill(account: Actor, skill: dict, instructions: str) -> int:
    body = {**skill, "content": instructions, "meta": {}}
    return _status(account, "POST", f"/api/v1/skills/id/{skill['id']}/update", body)


def _read_tool(account: Actor, tool: dict) -> int:
    return _status(account, "GET", f"/api/v1/tools/id/{tool['id']}")


def _update_tool(account: Actor, tool: dict, source: str) -> int:
    body = {**tool, "content": source, "meta": {"description": "pings"}}
    return _status(account, "POST", f"/api/v1/tools/id/{tool['id']}/update", body)


def _read_note(account: Actor, note_id: str) -> int:
    return _status(account, "GET", f"/api/v1/notes/{note_id}")


def _list_row(page: Page, section: str, name: str) -> Locator:
    page.goto(f"/workspace/{section}")
    page.get_by_role("textbox", name=f"Search {section.title()}").fill(name)
    row = page.get_by_role("main").get_by_role("button").filter(has_text=name)
    expect(row).to_be_visible()
    return row


def _open_row_menu(row: Locator, trigger: str) -> None:
    row.hover()
    row.get_by_role("button", name=trigger).last.click()


def _open_access(page: Page, row: Locator, trigger: str) -> Locator:
    _open_row_menu(row, trigger)
    page.get_by_role("button", name="Access", exact=True).click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    expect(dialog).to_be_visible()
    return dialog


def _note_row(page: Page, title: str) -> Locator:
    page.goto("/notes")
    row = page.get_by_role("main").get_by_role("button", name="Open note").filter(has_text=title)
    expect(row).to_be_visible()
    return row


def _add_access(page: Page, dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(name)
    picker.get_by_role("button", name=name).last.click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    expect(dialog.get_by_text(name)).to_be_visible()


def _principal_row(dialog: Locator, name: str) -> Locator:
    return dialog.get_by_role("combobox", name="Access level").locator(
        f"xpath=ancestor::div[contains(@class, 'justify-between')][.//*[contains(., '{name}')]][1]"
    )


def _set_level(page: Page, dialog: Locator, name: str, level: str) -> None:
    _principal_row(dialog, name).get_by_role("combobox").select_option(level)
    expect(page.get_by_text("Saved").last).to_be_visible()


def _remove_row(page: Page, dialog: Locator, name: str) -> None:
    _principal_row(dialog, name).get_by_role("button").last.click()
    expect(page.get_by_text("Saved").last).to_be_visible()
    expect(dialog.get_by_text("No access grants. Private to you.")).to_be_visible()


def _choose_visibility(page: Page, dialog: Locator, option: str) -> None:
    dialog.get_by_label("Visibility").click()
    page.get_by_role("menuitemradio", name=option).click()


def test_a_skill_shared_from_its_menu_is_readable_then_writable_then_withdrawn(
    page_for, owner, make_user
):
    person = make_user()
    skill = _new_skill(owner)
    page = page_for(owner)
    assert _read_skill(person, skill) != 200

    dialog = _open_access(page, _list_row(page, "skills", skill["name"]), "Skill Menu")
    _add_access(page, dialog, person.name)
    expect(dialog.get_by_role("combobox", name="Access level")).to_have_value("read")
    assert _read_skill(person, skill) == 200
    assert _update_skill(person, skill, "Tie a reef knot.") in (401, 403, 404)

    _set_level(page, dialog, person.name, "write")
    assert _update_skill(person, skill, "Tie a sheet bend.") == 200
    with owner.client() as client:
        stored = client.get(f"/api/v1/skills/id/{skill['id']}").json()
    assert stored["content"] == "Tie a sheet bend."

    _remove_row(page, dialog, person.name)
    assert _read_skill(person, skill) != 200
    assert _update_skill(person, skill, "Tie a clove hitch.") != 200


def test_a_tool_shared_with_a_group_from_its_menu_reaches_the_member(
    page_for, owner, make_user, group_name_of
):
    member = make_user()
    _, group_name = group_name_of(member)
    tool = _new_tool(owner)
    page = page_for(owner)
    assert _read_tool(member, tool) != 200

    dialog = _open_access(page, _list_row(page, "tools", tool["name"]), "Tool Menu")
    _add_access(page, dialog, group_name)
    assert _read_tool(member, tool) == 200
    assert _update_tool(member, tool, TOOL_SOURCE + "\n# edited") != 200

    _set_level(page, dialog, group_name, "write")
    assert _update_tool(member, tool, TOOL_SOURCE + "\n# edited") == 200

    _remove_row(page, dialog, group_name)
    assert _read_tool(member, tool) != 200
    assert _update_tool(member, tool, TOOL_SOURCE + "\n# again") != 200


def test_a_note_shared_from_the_notes_list_opens_for_the_person_until_removed(
    page_for, make_user, admin
):
    sharer = make_user()
    make_group(admin, [sharer], {"sharing": {"notes": True}})
    person = make_user()
    title = _unique("Agenda")
    note_id = _new_note(sharer, title)
    page = page_for(sharer)
    assert _read_note(person, note_id) != 200

    row = _note_row(page, title)
    row.get_by_label("Note Menu").click()
    page.get_by_role("button", name="Access", exact=True).click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    expect(dialog).to_be_visible()
    _add_access(page, dialog, person.name)
    assert _read_note(person, note_id) == 200

    _remove_row(page, dialog, person.name)
    assert _read_note(person, note_id) != 200


def test_a_person_with_write_access_to_a_note_is_not_offered_access_in_its_menu(
    page_for, make_user
):
    author, writer = make_user(), make_user()
    title = _unique("Shared draft")
    note_id = _new_note(author, title)
    with author.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update",
            json={
                "access_grants": [
                    {"principal_type": "user", "principal_id": writer.id, "permission": "read"},
                    {"principal_type": "user", "principal_id": writer.id, "permission": "write"},
                ]
            },
        )
    assert shared.status_code == 200, shared.text
    page = page_for(writer)

    row = _note_row(page, title)
    row.get_by_label("Note Menu").click()

    expect(page.get_by_role("button", name="Delete", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Access", exact=True)).to_have_count(0)


def test_without_the_public_sharing_permission_visibility_has_no_public_choice(
    page_for, make_user, admin
):
    sharer = make_user()
    make_group(admin, [sharer], {**SKILLS_WORKSPACE, "sharing": {"skills": True}})
    skill = _new_skill(sharer)
    page = page_for(sharer)

    dialog = _open_access(page, _list_row(page, "skills", skill["name"]), "Skill Menu")

    dialog.get_by_label("Visibility").click()
    expect(page.get_by_role("menuitemradio", name="Private")).to_be_visible()
    expect(page.get_by_role("menuitemradio", name="Public")).to_have_count(0)
    with sharer.client() as client:
        client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def test_with_the_public_sharing_permission_a_skill_made_public_reaches_anyone(
    page_for, make_user, admin
):
    sharer, stranger = make_user(), make_user()
    make_group(
        admin, [sharer], {**SKILLS_WORKSPACE, "sharing": {"skills": True, "public_skills": True}}
    )
    skill = _new_skill(sharer)
    page = page_for(sharer)
    assert _read_skill(stranger, skill) != 200

    try:
        dialog = _open_access(page, _list_row(page, "skills", skill["name"]), "Skill Menu")
        _choose_visibility(page, dialog, "Public")
        expect(page.get_by_text("Saved").first).to_be_visible()
        assert _read_skill(stranger, skill) == 200

        _choose_visibility(page, dialog, "Private")
        expect(page.get_by_text("Saved").last).to_be_visible()
        assert _read_skill(stranger, skill) != 200
    finally:
        with sharer.client() as client:
            client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


@pytest.mark.parametrize(
    ("section", "trigger", "made"),
    [("skills", "Skill Menu", _new_skill), ("tools", "Tool Menu", _new_tool)],
)
def test_a_reader_has_no_menu_and_a_writer_is_offered_access(
    page_for, owner, make_user, admin, section, trigger, made
):
    reader, writer = make_user(), make_user()
    make_group(admin, [reader, writer], WORKSPACE_USER)
    item = made(owner)
    grants = [
        {"principal_type": "user", "principal_id": reader.id, "permission": "read"},
        {"principal_type": "user", "principal_id": writer.id, "permission": "read"},
        {"principal_type": "user", "principal_id": writer.id, "permission": "write"},
    ]
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/{section}/id/{item['id']}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text

    reader_page = page_for(reader)
    reader_row = _list_row(reader_page, section, item["name"])
    reader_row.hover()
    expect(reader_row.get_by_role("button", name=trigger)).to_have_count(0)

    writer_page = page_for(writer)
    _open_row_menu(_list_row(writer_page, section, item["name"]), trigger)
    expect(writer_page.get_by_role("button", name="Clone", exact=True)).to_be_visible()
    expect(writer_page.get_by_role("button", name="Access", exact=True)).to_be_visible()
