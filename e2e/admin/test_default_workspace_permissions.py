"""Journey: the workspace and sharing switches of the default permissions reach a user's workspace.

Admin Panel > Users > Groups > Default permissions holds a Workspace block and a Sharing block
whose switches are all off by default (only Allow Sharing With Users / Groups start on), so a
fresh user has no workspace and nothing to share with. An admin switches them on in the dialog
and each shows where a person meets it:

- Models, Knowledge, Prompts, Tools and Skills Access: the section opens and its Create button
  is there and opens the create form; switched back off, /workspace/<section> sends the user
  home with no Create button.
- Import and Export of Models, Prompts, Tools and Skills: the section's create menu offers
  Import JSON (Import for skills) or Export JSON, each by its own switch and none when both are
  off.
- Sharing of Models, Prompts, Knowledge, Tools, Skills and Notes: the item's Access dialog offers
  Add Access; off, the dialog has no Add Access button, however Public Sharing is set.
- Public Sharing of the same six: the Access dialog's visibility list offers Public next to
  Private; off, it lists Private only.
- Allow Sharing With Users / Groups: the Add Access picker lists the users or the groups.

Discriminates: passes on dev 30f3f6a8f. A frontend build that always grants (workspace pages never
redirect, every import and export entry shown, Add Access and Public always offered, the picker
always listing users and groups) turns red the "switched back off" and "both switched off" tests
and the "adds only its menu entry" and "but not public" tests, and the picker removal tests; one
that never grants (every workspace page redirects, no import or export entry, no Add Access, no
Public, an empty picker) turns red every "switched on" test, the "offers its create button" ones
and the default picker test. Retargeted for 9bbb95048, which renamed the skills entry to Import;
the import and export tests pass on dev 178de3666.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = "class Tools:\n    def ping(self) -> str:\n        return 'pong'\n"

# section: the switch that opens it
SECTION_ACCESS = {
    "models": "Models Access",
    "knowledge": "Knowledge Access",
    "prompts": "Prompts Access",
    "tools": "Tools Access",
    "skills": "Skills Access",
}

# the other sections open their create form in a dialog over the list
CREATE_OPENS_A_PAGE = ("models", "tools", "skills")

# section: (Import switch, Export switch); Knowledge has neither
SECTION_IO = {
    "models": ("Import Models", "Export Models"),
    "prompts": ("Import Prompts", "Export Prompts"),
    "tools": ("Import Tools", "Export Tools"),
    "skills": ("Import Skills", "Export Skills"),
}


def save_default_permissions(page: Page, switches: dict[str, bool]) -> None:
    """Set each switch in the Default permissions dialog, in order, and save."""
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    for switch, turn_on in switches.items():
        target = dialog.get_by_role("switch", name=switch, exact=True)
        if (target.get_attribute("aria-checked") == "true") != turn_on:
            target.click()
        expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def open_section(page: Page, section: str) -> None:
    page.goto(f"/workspace/{section}")


@pytest.fixture
def switch_defaults(page_for, admin, preserve) -> Callable[[dict[str, bool]], None]:
    preserve("permissions")
    admin_page = page_for(admin)

    def switch(switches: dict[str, bool]) -> None:
        save_default_permissions(admin_page, switches)

    return switch


@pytest.fixture
def notes_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post("/api/v1/auths/admin/config", json={**current, "ENABLE_NOTES": True})
    saved.raise_for_status()


@pytest.mark.parametrize("section", SECTION_ACCESS)
def test_a_section_switched_on_offers_its_create_button(
    section, switch_defaults, page_for, make_user
):
    switch_defaults({SECTION_ACCESS[section]: True})
    page = page_for(make_user())

    open_section(page, section)

    page.get_by_role("button", name="Create", exact=True).click()
    if section in CREATE_OPENS_A_PAGE:
        expect(page).to_have_url(re.compile(rf"/workspace/{section}/create$"))
    else:
        expect(page.get_by_role("dialog")).to_be_visible()


@pytest.mark.parametrize("section", SECTION_ACCESS)
def test_a_section_switched_back_off_is_closed_and_has_no_create_button(
    section, switch_defaults, page_for, make_user
):
    switch_defaults({SECTION_ACCESS[section]: True})
    switch_defaults({SECTION_ACCESS[section]: False})
    page = page_for(make_user())

    open_section(page, section)

    page.wait_for_url(lambda url: "/workspace" not in url)
    expect(page.get_by_role("button", name="Create", exact=True)).to_have_count(0)


def create_menu_entries(page: Page, section: str) -> tuple[Locator, Locator]:
    open_section(page, section)
    page.get_by_label("Open create menu").click()
    menu = page.get_by_role("menu")
    # skills import JSON, ZIP or Markdown files, so their entry is just Import
    import_label = "Import" if section == "skills" else "Import JSON"
    return (
        menu.get_by_role("button", name=import_label, exact=True),
        menu.get_by_role("button", name="Export JSON"),
    )


IO_CASES = [(section, kind) for section in SECTION_IO for kind in (0, 1)]
IO_IDS = [f"{section}-{('import', 'export')[kind]}" for section, kind in IO_CASES]


@pytest.mark.parametrize(("section", "kind"), IO_CASES, ids=IO_IDS)
def test_an_import_or_export_switched_on_adds_only_its_menu_entry(
    section, kind, switch_defaults, page_for, make_user
):
    switch_defaults({SECTION_ACCESS[section]: True, SECTION_IO[section][kind]: True})
    page = page_for(make_user())

    entries = create_menu_entries(page, section)

    expect(entries[kind]).to_be_visible()
    expect(entries[1 - kind]).to_have_count(0)


@pytest.mark.parametrize(("section", "kind"), IO_CASES, ids=IO_IDS)
def test_an_import_or_export_switched_back_off_leaves_the_menu(
    section, kind, switch_defaults, page_for, make_user
):
    both = {SECTION_ACCESS[section]: True, **{name: True for name in SECTION_IO[section]}}
    switch_defaults(both)
    switch_defaults({SECTION_IO[section][kind]: False})
    page = page_for(make_user())

    entries = create_menu_entries(page, section)

    expect(entries[1 - kind]).to_be_visible()
    expect(entries[kind]).to_have_count(0)


@pytest.mark.parametrize("section", SECTION_IO)
def test_with_both_switched_off_the_create_menu_is_just_a_button(
    section, switch_defaults, page_for, make_user
):
    switch_defaults({SECTION_ACCESS[section]: True})
    page = page_for(make_user())

    open_section(page, section)

    expect(page.get_by_role("button", name="Create", exact=True)).to_be_visible()
    expect(page.get_by_label("Open create menu")).to_have_count(0)


def _create(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _model(owner: Actor) -> str:
    model_id = f"shared-{uuid.uuid4().hex[:8]}"
    form = {"id": model_id, "base_model_id": MOCK_MODEL_ID, "name": model_id, "meta": {}}
    _create(owner, "/api/v1/models/create", {**form, "params": {}})
    return f"/workspace/models/edit?id={model_id}"


def _prompt(owner: Actor) -> str:
    suffix = uuid.uuid4().hex[:8]
    form = {"command": f"shared{suffix}", "name": f"Shared {suffix}", "content": "Say hello."}
    return f"/workspace/prompts/{_create(owner, '/api/v1/prompts/create', form)['id']}"


def _knowledge(owner: Actor) -> str:
    form = {"name": f"Shared {uuid.uuid4().hex[:8]}", "description": ""}
    return f"/workspace/knowledge/{_create(owner, '/api/v1/knowledge/create', form)['id']}"


def _tool(owner: Actor) -> str:
    tool_id = f"shared_tool_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": tool_id, "content": TOOL_SOURCE, "meta": {"description": "x"}}
    _create(owner, "/api/v1/tools/create", form)
    return f"/workspace/tools/edit?id={tool_id}"


def _skill(owner: Actor) -> str:
    skill_id = f"shared-skill-{uuid.uuid4().hex[:8]}"
    form = {"id": skill_id, "name": skill_id, "content": "Be brief.", "meta": {}, "is_active": True}
    _create(owner, "/api/v1/skills/create", form)
    return f"/workspace/skills/edit?id={skill_id}"


def _note(owner: Actor) -> str:
    form = {"title": f"Shared {uuid.uuid4().hex[:8]}", "data": {"content": {"md": "Berth 4."}}}
    return f"/notes/{_create(owner, '/api/v1/notes/create', form)['id']}"


# kind: (create an item and name its page, section switch or None, Sharing switch, Public switch)
SHARED_KINDS = {
    "models": (_model, "Models Access", "Models Sharing", "Models Public Sharing"),
    "prompts": (_prompt, "Prompts Access", "Prompts Sharing", "Prompts Public Sharing"),
    "knowledge": (_knowledge, "Knowledge Access", "Knowledge Sharing", "Knowledge Public Sharing"),
    "tools": (_tool, "Tools Access", "Tools Sharing", "Tools Public Sharing"),
    "skills": (_skill, "Skills Access", "Skills Sharing", "Skills Public Sharing"),
    "notes": (_note, None, "Notes Sharing", "Notes Public Sharing"),
}


def open_access_dialog(page: Page, owner: Actor, kind: str) -> Locator:
    """The Access dialog of a fresh item `owner` owns, opened from the item's own page."""
    page.goto(SHARED_KINDS[kind][0](owner))
    page.get_by_role("button", name="Access", exact=True).click()
    return page.get_by_role("dialog").filter(has_text="Access Control")


def visibility_choices(dialog: Locator) -> Locator:
    return dialog.get_by_role("combobox").locator("option")


def sharing_switches(kind: str, *names: int) -> dict[str, bool]:
    """The Workspace Access switch (when the kind has one) and the Sharing switches, all on."""
    _, section, sharing, public = SHARED_KINDS[kind]
    switches = {section: True} if section else {}
    return {**switches, **{(sharing, public)[name]: True for name in names}}


@pytest.fixture
def owner_after(switch_defaults, make_user, notes_on):
    """`owner_after(switches)` saves the switches in the dialog and returns a fresh user."""

    def make(switches: dict[str, bool]) -> Actor:
        switch_defaults(switches)
        return make_user()

    return make


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_sharing_switched_on_offers_add_access_but_not_public(kind, owner_after, page_for):
    owner = owner_after(sharing_switches(kind, 0))

    dialog = open_access_dialog(page_for(owner), owner, kind)

    expect(dialog.get_by_role("button", name="Add Access")).to_be_visible()
    expect(visibility_choices(dialog)).to_have_text(["Private"])


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_public_sharing_switched_on_offers_public(kind, owner_after, page_for):
    owner = owner_after(sharing_switches(kind, 0, 1))

    dialog = open_access_dialog(page_for(owner), owner, kind)

    expect(dialog.get_by_role("button", name="Add Access")).to_be_visible()
    expect(visibility_choices(dialog)).to_have_text(["Private", "Public"])


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_public_sharing_switched_back_off_withdraws_public_only(
    kind, owner_after, switch_defaults, page_for
):
    owner = owner_after(sharing_switches(kind, 0, 1))
    switch_defaults({SHARED_KINDS[kind][3]: False})

    dialog = open_access_dialog(page_for(owner), owner, kind)

    expect(dialog.get_by_role("button", name="Add Access")).to_be_visible()
    expect(visibility_choices(dialog)).to_have_text(["Private"])


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_sharing_switched_back_off_withdraws_add_access_and_public(
    kind, owner_after, switch_defaults, page_for
):
    owner = owner_after(sharing_switches(kind, 0, 1))
    switch_defaults({SHARED_KINDS[kind][2]: False})

    dialog = open_access_dialog(page_for(owner), owner, kind)

    expect(visibility_choices(dialog)).to_have_text(["Private"])
    expect(dialog.get_by_role("button", name="Add Access")).to_have_count(0)


GRANT_SWITCHES = {"users": "Allow Sharing With Users", "groups": "Allow Sharing With Groups"}


@pytest.fixture
def picker_for(owner_after, page_for, admin, make_user):
    """`picker_for(switches)` opens a prompt's Add Access picker as a fresh user.

    The picker is searched for a token that one group and one user both carry in their names.
    """

    def open_picker(switches: dict[str, bool]) -> Locator:
        group_id = make_group(admin, [])
        with admin.client() as client:
            group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
            member = make_user(name=f"Member {group_name.split()[-1]}")
            owner = owner_after({**sharing_switches("prompts", 0), **switches})
            # the picker lists the groups its user belongs to
            members = {"user_ids": [member.id, owner.id]}
            client.post(f"/api/v1/groups/id/{group_id}/users/add", json=members).raise_for_status()
        dialog = open_access_dialog(page_for(owner), owner, "prompts")
        dialog.get_by_role("button", name="Add Access").click()
        picker = dialog.page.get_by_role("dialog").filter(has_text="Add Access").last
        token = group_name.split()[-1]
        with dialog.page.expect_response(lambda r: "/users/search" in r.url and token in r.url):
            picker.get_by_placeholder("Search").fill(token)
        return picker

    return open_picker


def test_by_default_the_add_access_picker_lists_users_and_groups(picker_for):
    picker = picker_for({})

    expect(picker.get_by_text("Member", exact=False)).to_be_visible()
    expect(picker.get_by_text("Users", exact=True)).to_be_visible()
    expect(picker.get_by_text("Groups", exact=True)).to_be_visible()


@pytest.mark.parametrize(
    ("switch", "withdrawn", "kept"), [("users", "Users", "Groups"), ("groups", "Groups", "Users")]
)
def test_a_grant_switch_switched_off_removes_that_list_from_the_picker(
    switch, withdrawn, kept, picker_for
):
    picker = picker_for({GRANT_SWITCHES[switch]: False})

    expect(picker.get_by_text(kept, exact=True)).to_be_visible()
    expect(picker.get_by_text(withdrawn, exact=True)).to_have_count(0)
