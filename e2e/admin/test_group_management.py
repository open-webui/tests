"""Journey: an admin manages a group's members and permissions in Admin Panel > Users > Groups.

Unticking a member in the group's Users tab takes them out: the group counts one member fewer after
a reload. Switching a permission on in the group's Permissions tab reaches its members: Models
Access opens the Models workspace for a member, while an account outside the group is still refused
it. Since dev da850248d the refused page keeps its address and shows a permission message in place
of the workspace; it sent the account back to the chat before. Deleting the group from its General
tab takes it off the list and out of the member's groups in the Edit User dialog. Each test works as
a fresh admin on a group of its own.

Discriminates: passes on dev 176d31d1d; in a frontend copy, unticking a member calling the add
route turns the member test red (no removal is sent) and the delete confirmation only closing the
dialog turns the delete test red (the group stays); in a backend copy, the effective permissions
leaving out the member's groups turns the permission test red (the member is refused).
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.page_tabs import expect_no_permission

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


@pytest.fixture
def member(make_user) -> Actor:
    return make_user()


@pytest.fixture
def group_name(admin, member) -> Iterator[str]:
    """A group holding `member`, deleted afterwards unless the test deleted it."""
    name = f"Crew {uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [member.id]}
        )
        assert added.status_code == 200, added.text
    yield name
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _search_groups(page: Page, name: str) -> Locator:
    groups = page.get_by_role("main")
    groups.get_by_role("textbox", name="Search Groups").fill(name)
    return groups


def _group_row(page: Page, name: str, members: int) -> Locator:
    groups = _search_groups(page, name)
    return groups.get_by_role("button", name=re.compile(rf"^{name} {members} direct members"))


def _open_group(page: Page, name: str, members: int) -> Locator:
    page.goto("/admin/users/groups")
    _group_row(page, name, members).click()
    return page.get_by_role("dialog").filter(has_text="Edit User Group")


def _edit_member(page: Page, member: Actor) -> Locator:
    """The member's Edit User dialog, once it has loaded the member's groups."""
    page.goto("/admin/users")
    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(member.email)
    edit = users.get_by_role("row").filter(has_text=member.email)
    with page.expect_response(lambda response: response.url.endswith(f"/{member.id}/groups")):
        edit.get_by_role("button", name="Edit User").click()
    return page.get_by_role("dialog").filter(has_text="Edit User")


def _expect_models_workspace(page: Page, opens: bool) -> None:
    page.goto("/workspace/models")
    if opens:
        models_tab = page.get_by_role("link", name=re.compile("^Models"))
        expect(models_tab).to_have_attribute("aria-current", "page")
        expect(page).to_have_url(re.compile(r"/workspace/models$"))
    else:
        expect_no_permission(page, "/workspace/models")
        expect(page.get_by_role("link", name=re.compile("^Models"))).to_have_count(0)


def test_unticking_a_member_takes_them_out_of_the_group(admin_page, member, group_name):
    editing = _open_group(admin_page, group_name, members=1)
    editing.get_by_role("button", name="Users", exact=True).click()
    editing.get_by_role("textbox", name="Search").fill(member.name)
    member_box = editing.get_by_role("checkbox", name=member.name)
    expect(member_box).to_have_attribute("aria-checked", "true")

    with admin_page.expect_response(lambda response: response.url.endswith("/users/remove")):
        member_box.click()
    expect(member_box).to_have_attribute("aria-checked", "false")

    admin_page.reload()
    expect(_group_row(admin_page, group_name, members=0)).to_be_visible()


def test_a_permission_switched_on_for_the_group_reaches_its_members(
    admin_page, page_for, make_user, member, group_name
):
    member_page = page_for(member)
    outsider_page = page_for(make_user())
    _expect_models_workspace(member_page, opens=False)

    editing = _open_group(admin_page, group_name, members=1)
    editing.get_by_role("button", name="Permissions", exact=True).click()
    models_access = editing.get_by_role("switch", name="Models Access")
    expect(models_access).to_have_attribute("aria-checked", "false")
    models_access.click()
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()
    expect(_group_row(admin_page, group_name, members=1)).to_contain_text("Custom permissions")

    _expect_models_workspace(member_page, opens=True)
    _expect_models_workspace(outsider_page, opens=False)


def test_a_deleted_group_leaves_the_list_and_the_members_groups(admin_page, member, group_name):
    expect(_edit_member(admin_page, member).get_by_role("link", name=group_name)).to_be_visible()

    editing = _open_group(admin_page, group_name, members=1)
    editing.get_by_role("button", name="Delete", exact=True).click()
    admin_page.get_by_role("button", name="Confirm").click()
    expect(admin_page.get_by_text("Group deleted successfully")).to_be_visible()
    expect(_group_row(admin_page, group_name, members=1)).to_have_count(0)

    admin_page.reload()
    expect(_search_groups(admin_page, group_name).get_by_text("No groups found")).to_be_visible()
    member_groups = _edit_member(admin_page, member).get_by_role("link", name=group_name)
    expect(member_groups).to_have_count(0)
