"""Journey: the permissions an admin sets on a group in its editor, as members and others see them.

In Admin Panel > Users > Groups an admin opens a group, switches permissions in its Permissions tab
and saves. Each section reaches the group's members and nobody else: Prompts Access with Import
Prompts gives a member the Prompts workspace with Import JSON while an outsider is sent home;
Notes Sharing with Notes Public Sharing offers a member Add Access and Public on their own note
while an outsider's note stays Private; Allow Chat Controls gives a member the Controls button
the defaults withdraw; Notes under Features reaches a member of a granting group even when their
other group leaves it off. A group cannot take away what the defaults give: switching Allow File
Upload off notes that the default stays enabled, and the member still attaches a file.

Changes reach a member's open tab without a reload, as the groups docs promise: Knowledge Access
switched on puts Workspace in their user menu and switched off takes it away again (and a reload
agrees), Reset to Defaults takes back what the group granted, and unticking them in the Users tab
takes both the group's permission and a model shared only with the group out of the open tab.
Default models set in the group's General tab start the member's next new chat on that model.
The sidebar's Workspace entry does not follow: it stays missing until a reload, so that test is
red on dev.

Discriminates: passes on dev ebc6add67 except the sidebar test, which a frontend copy whose
sidebar re-checks its entries when the account changes turns green. In a backend copy, groups
combining to the later group's value turns the two-group and file upload tests red; group
permissions left out of the account's permissions turns the four section tests, the open-tab
Workspace tests and the reset test red; no access refresh sent to the members turns the open-tab,
reset and default model tests red. In a frontend copy whose access refresh leaves the account and
config as they were, the open-tab, reset and default model tests go red, and without the "will
remain enabled" note the file upload test does; Reset to Defaults keeping the switches as they
were turns the reset test red.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.model_selector import model_options, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DEFAULTS = "/api/v1/users/default/permissions"
ADMIN_CONFIG = "/api/v1/auths/admin/config"
NO_WORKSPACE = dict.fromkeys(["models", "knowledge", "prompts", "tools", "skills"], False)
KEPT_BY_DEFAULT = "This is a default user permission and will remain enabled."


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


@pytest.fixture
def set_defaults(admin, preserve) -> Callable[..., None]:
    """`set_defaults(section={key: value})` saves default permissions; restored afterwards."""
    preserve("permissions", "admin_config")
    with admin.client() as client:
        enabled = client.post(
            ADMIN_CONFIG, json={**client.get(ADMIN_CONFIG).json(), "ENABLE_NOTES": True}
        )
    enabled.raise_for_status()

    def save(**sections: dict[str, bool]) -> None:
        with admin.client() as client:
            current = client.get(DEFAULTS).json()
            changed = {name: {**current[name], **sections.get(name, {})} for name in current}
            client.post(DEFAULTS, json=changed).raise_for_status()

    return save


@pytest.fixture
def make_named_group(admin) -> Iterator[Callable[..., str]]:
    """`make_named_group(members)` adds a group the way New Group does and returns its name.

    New Group starts from the default permissions as they stand, so the group grants nothing more.
    """
    ids: list[str] = []
    with admin.client() as client:

        def create(members: list[Actor], **permissions: dict[str, bool]) -> str:
            name = f"Crew {uuid.uuid4().hex[:8]}"
            current = client.get(DEFAULTS).json()
            starting = {key: {**current[key], **permissions.get(key, {})} for key in current}
            form = {"name": name, "description": "", "permissions": starting}
            created = client.post("/api/v1/groups/create", json=form)
            assert created.status_code == 200, created.text
            ids.append(created.json()["id"])
            added = client.post(
                f"/api/v1/groups/id/{ids[-1]}/users/add",
                json={"user_ids": [member.id for member in members]},
            )
            assert added.status_code == 200, added.text
            return name

        yield create
        for group_id in ids:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def open_group(page: Page, name: str, tab: str) -> Locator:
    page.goto("/admin/users/groups")
    page.get_by_role("main").get_by_role("textbox", name="Search Groups").fill(name)
    page.get_by_role("button", name=re.compile(rf"^{name} \d+ direct members")).click()
    editing = page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name=tab, exact=True).click()
    return editing


def flip(editing: Locator, switch: str, turn_on: bool) -> None:
    target = editing.get_by_role("switch", name=switch, exact=True)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")


def save_group(page: Page, editing: Locator) -> None:
    editing.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Group updated successfully")).to_be_visible()


def save_group_switches(page: Page, name: str, switches: dict[str, bool]) -> None:
    """Set each switch in the group's Permissions tab, in order, and save."""
    editing = open_group(page, name, "Permissions")
    for switch, turn_on in switches.items():
        flip(editing, switch, turn_on)
    save_group(page, editing)


def ready(page: Page) -> Page:
    expect(chat_input(page)).to_be_visible()
    return page


def sidebar_entry(page: Page, name: str) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("link", name="New Chat")).to_be_visible()
    return sidebar.get_by_role("link", name=name, exact=True)


def user_menu_entry(page: Page, name: str) -> Locator:
    """The user menu's link `name`; the menu stays open, so it follows changes as they land."""
    ready(page).get_by_role("button", name="User menu").first.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Settings")).to_be_visible()
    return menu.get_by_role("link", name=name, exact=True)


# each section of the Permissions tab reaches the group's members only


def test_prompts_access_with_import_reaches_the_members_and_not_an_outsider(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(workspace={"prompts": False, "prompts_import": False})
    member, outsider = make_user(), make_user()
    group = make_named_group([member])

    save_group_switches(admin_page, group, {"Prompts Access": True, "Import Prompts": True})

    member_page = page_for(member)
    member_page.goto("/workspace/prompts")
    member_page.get_by_label("Open create menu").click()
    import_entry = member_page.get_by_role("menu").get_by_role("button", name="Import JSON")
    expect(import_entry).to_be_visible()
    outsider_page = page_for(outsider)
    outsider_page.goto("/workspace/prompts")
    outsider_page.wait_for_url(lambda url: "/workspace" not in url)


def _note_access_dialog(page: Page, owner: Actor) -> Locator:
    form = {"title": f"Berths {uuid.uuid4().hex[:6]}", "data": {"content": {"md": "Berth 4."}}}
    with owner.client() as client:
        created = client.post("/api/v1/notes/create", json=form)
    assert created.status_code == 200, created.text
    page.goto(f"/notes/{created.json()['id']}")
    page.get_by_role("button", name="Access", exact=True).click()
    return page.get_by_role("dialog").filter(has_text="Access Control")


def test_notes_sharing_with_public_reaches_the_members_and_not_an_outsider(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(sharing={"notes": False, "public_notes": False})
    member, outsider = make_user(), make_user()
    group = make_named_group([member])

    save_group_switches(admin_page, group, {"Notes Sharing": True, "Notes Public Sharing": True})

    member_dialog = _note_access_dialog(page_for(member), member)
    expect(member_dialog.get_by_role("button", name="Add Access")).to_be_visible()
    expect(member_dialog.get_by_role("combobox").locator("option")).to_have_text(
        ["Private", "Public"]
    )
    outsider_dialog = _note_access_dialog(page_for(outsider), outsider)
    expect(outsider_dialog.get_by_role("combobox").locator("option")).to_have_text(["Private"])
    expect(outsider_dialog.get_by_role("button", name="Add Access")).to_have_count(0)


def test_chat_controls_withdrawn_by_default_come_back_for_the_members(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(chat={"controls": False})
    member, outsider = make_user(), make_user()
    group = make_named_group([member])

    save_group_switches(admin_page, group, {"Allow Chat Controls": True})

    controls = ready(page_for(member)).get_by_role("button", name="Controls", exact=True)
    expect(controls).to_be_visible()
    outsider_page = ready(page_for(outsider))
    expect(outsider_page.get_by_role("button", name="Controls", exact=True)).to_have_count(0)


def test_a_member_of_two_groups_gets_notes_from_the_one_that_grants_it(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(features={"notes": False})
    in_both, in_denying_only = make_user(), make_user()
    granting = make_named_group([in_both])
    make_named_group([in_both, in_denying_only])

    save_group_switches(admin_page, granting, {"Notes": True})

    expect(user_menu_entry(page_for(in_both), "Notes")).to_be_visible()
    expect(user_menu_entry(page_for(in_denying_only), "Notes")).to_have_count(0)


def test_a_group_switching_file_upload_off_leaves_the_default_in_place(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(chat={"file_upload": True})
    member = make_user()
    group = make_named_group([member])

    editing = open_group(admin_page, group, "Permissions")
    flip(editing, "Allow File Upload", turn_on=False)
    upload_row = editing.get_by_role("switch", name="Allow File Upload").locator(
        "xpath=ancestor::div[contains(@class, 'flex-col')][1]"
    )
    expect(upload_row).to_contain_text(KEPT_BY_DEFAULT)
    save_group(admin_page, editing)

    member_page = ready(page_for(member))
    member_page.get_by_role("button", name="More", exact=True).last.click()
    with member_page.expect_file_chooser() as chooser:
        member_page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files(
        files=[{"name": "manifest.txt", "mimeType": "text/plain", "buffer": b"Berth 4"}]
    )
    expect(member_page.get_by_text("manifest.txt")).to_be_visible()


# changes reach a member's open tab without a reload


def test_an_open_tab_gains_and_loses_the_workspace_as_the_group_changes(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(workspace=NO_WORKSPACE)
    member = make_user()
    group = make_named_group([member])
    member_page = page_for(member)
    workspace = user_menu_entry(member_page, "Workspace")
    expect(workspace).to_have_count(0)

    save_group_switches(admin_page, group, {"Knowledge Access": True})
    expect(workspace).to_be_visible()

    save_group_switches(admin_page, group, {"Knowledge Access": False})
    expect(workspace).to_have_count(0)
    member_page.reload()
    expect(user_menu_entry(member_page, "Workspace")).to_have_count(0)


def test_reset_to_defaults_takes_back_what_the_group_granted(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(workspace=NO_WORKSPACE)
    member = make_user()
    group = make_named_group([member], workspace={"knowledge": True})
    workspace = user_menu_entry(ready(page_for(member)), "Workspace")
    expect(workspace).to_be_visible()

    editing = open_group(admin_page, group, "Permissions")
    editing.get_by_role("button", name="Reset to Defaults").click()
    admin_page.get_by_role("dialog").filter(has_text="Reset to Defaults").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(editing.get_by_role("switch", name="Knowledge Access")).to_have_attribute(
        "aria-checked", "false"
    )
    save_group(admin_page, editing)

    expect(workspace).to_have_count(0)


def test_the_sidebar_gains_the_workspace_entry_without_a_reload(
    admin_page, page_for, make_user, set_defaults, make_named_group
):
    set_defaults(workspace=NO_WORKSPACE)
    member = make_user()
    group = make_named_group([member])
    member_page = ready(page_for(member))
    expect(sidebar_entry(member_page, "Workspace")).to_have_count(0)

    save_group_switches(admin_page, group, {"Knowledge Access": True})

    expect(user_menu_entry(member_page, "Workspace")).to_be_visible()
    member_page.keyboard.press("Escape")
    stale = "the user menu offers Workspace at once, the sidebar only after a reload"
    expect(sidebar_entry(member_page, "Workspace"), stale).to_be_visible()


@pytest.fixture
def group_model(admin) -> Iterator[Callable[[str], dict]]:
    """`group_model(group_name)` adds a preset of the scripted model readable by that group only."""
    ids: list[str] = []
    with admin.client() as client:

        def create(group_name: str) -> dict:
            group_id = next(
                group["id"]
                for group in client.get("/api/v1/groups/").json()
                if group["name"] == group_name
            )
            suffix = uuid.uuid4().hex[:6]
            form = {
                "id": f"crew-{suffix}",
                "base_model_id": MOCK_MODEL_ID,
                "name": f"Crew model {suffix}",
                "meta": {},
                "params": {},
                "access_grants": [grant("group", group_id, "read")],
            }
            created = client.post("/api/v1/models/create", json=form)
            assert created.status_code == 200, created.text
            ids.append(form["id"])
            return form

        yield create
        for model_id in ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_unticking_a_member_takes_the_groups_permission_and_model_from_their_open_tab(
    admin_page, page_for, make_user, set_defaults, make_named_group, group_model
):
    set_defaults(workspace=NO_WORKSPACE)
    member = make_user()
    group = make_named_group([member], workspace={"knowledge": True})
    model = group_model(group)
    member_page = ready(page_for(member))
    expect(model_options(member_page, model["name"])).to_have_count(1)
    member_page.keyboard.press("Escape")
    workspace = user_menu_entry(member_page, "Workspace")
    expect(workspace).to_be_visible()

    editing = open_group(admin_page, group, "Users")
    editing.get_by_role("textbox", name="Search").fill(member.name)
    with admin_page.expect_response(lambda response: response.url.endswith("/users/remove")):
        editing.get_by_role("checkbox", name=member.name).click()

    expect(workspace).to_have_count(0)
    member_page.keyboard.press("Escape")
    expect(model_options(member_page, MOCK_MODEL_ID)).to_have_count(1)
    expect(model_options(member_page, model["name"])).to_have_count(0)


def test_default_models_set_in_the_editor_start_the_members_next_new_chat(
    admin_page, page_for, make_user, make_named_group, group_model
):
    member, outsider = make_user(), make_user()
    group = make_named_group([member])
    model = group_model(group)
    member_page = ready(page_for(member))
    expect(member_page.get_by_role("button", name=re.compile("^Selected model"))).to_be_visible()

    editing = open_group(admin_page, group, "General")
    editing.get_by_role("button", name="Inherit", exact=True).click()
    select_model(admin_page, model["name"])
    editing.get_by_text("Edit User Group").click()
    with member_page.expect_response(lambda response: response.url.endswith("/api/config")):
        save_group(admin_page, editing)

    member_page.get_by_role("link", name="New Chat").click()
    expect(
        member_page.get_by_role("button", name=f"Selected model: {model['name']}")
    ).to_be_visible()
    outsider_page = ready(page_for(outsider))
    expect(
        outsider_page.get_by_role("button", name=f"Selected model: {model['name']}")
    ).to_have_count(0)
