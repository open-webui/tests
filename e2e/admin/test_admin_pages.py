"""Journey: an admin adds a connection, an account and a group from the admin panel.

On Connections the admin adds an OpenAI connection pointing at a local listener, and the
listener's model shows in the chat's model selector. On Users the admin adds an account through
the form, which can then sign in, and changes its role; the new role is there after a reload. On
Groups the admin creates a group and ticks a member, and the group counts that member after a
reload. Each test works as a fresh admin.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `/openai/config/update` storing
only the first base URL the connection test fails (the selector never offers the listener's
model), with the user update route dropping `role` the users test fails after the reload, and
with a group's add-users route adding nobody the group test fails (0 members).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import sign_in
from harness.listener import json_answer
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OPENAI_CONFIG = ("/openai/config", "/openai/config/update")
LISTENER_MODEL = "harbor-model"


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


def _admin_main(page: Page, path: str) -> Locator:
    page.goto(path)
    return page.get_by_role("main")


def test_an_added_connection_offers_its_models_in_the_selector(admin_page, listener, preserve):
    preserve(OPENAI_CONFIG)
    models = {"object": "list", "data": [{"id": LISTENER_MODEL, "object": "model"}]}
    listener.route("GET", "/v1/models", json_answer(models))

    admin_page.goto("/admin/settings/connections")
    settings = admin_page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Connections", exact=True)).to_be_visible()
    tooltip_button(settings, "Add Connection").click()
    adding = admin_page.get_by_role("dialog").filter(has_text="Add Connection")
    adding.get_by_role("combobox", name="URL").fill(f"{listener.base_url}/v1")
    adding.get_by_role("textbox", name="API Key").fill("sk-local")
    adding.get_by_role("button", name="Save").click()
    expect(settings.get_by_role("textbox", name="API Base URL")).to_have_count(2)

    admin_page.goto("/")
    admin_page.get_by_role("button", name=re.compile("^Selected model")).click()
    available = admin_page.get_by_role("listbox", name="Available models")
    expect(available.get_by_role("option", name=f"Select {LISTENER_MODEL} model")).to_be_visible()
    assert listener.requests_to("/v1/models")


def test_an_added_account_signs_in_and_keeps_a_changed_role(admin_page, instance):
    suffix = uuid.uuid4().hex[:8]
    email, password = f"added-{suffix}@example.com", "addedpassword123"
    users = _admin_main(admin_page, "/admin/users")
    users.get_by_role("button", name="Add User").click()
    form = admin_page.get_by_role("dialog").filter(has_text="Add User")
    form.get_by_role("textbox", name="Name").fill(f"Added {suffix}")
    form.get_by_role("textbox", name="Email").fill(email)
    form.get_by_role("textbox", name="Enter Your Password").fill(password)
    form.get_by_role("button", name="Save").click()

    users.get_by_role("textbox", name="Search").fill(email)
    row = users.get_by_role("row").filter(has_text=email)
    expect(row.get_by_role("button", name="Change User Role")).to_have_text("user")
    assert sign_in(instance, email, password)

    row.get_by_role("button", name="Change User Role").click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User")
    editing.get_by_role("combobox", name="Role").select_option(label="Pending")
    editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    admin_page.reload()
    users.get_by_role("textbox", name="Search").fill(email)
    expect(row.get_by_role("button", name="Change User Role")).to_have_text("pending")


@pytest.fixture
def group_name(admin):
    name = f"Crew {uuid.uuid4().hex[:6]}"
    yield name
    with admin.client() as client:
        for group in client.get("/api/v1/groups/").json():
            if group["name"] == name:
                client.delete(f"/api/v1/groups/id/{group['id']}/delete")


def test_a_new_group_keeps_the_member_ticked_in_it(admin_page, make_user, group_name):
    member = make_user()
    groups = _admin_main(admin_page, "/admin/users/groups")
    groups.get_by_role("button", name="New Group").click()
    creating = admin_page.get_by_role("dialog").filter(has_text="Add User Group")
    creating.get_by_role("textbox", name="Group Name").fill(group_name)
    creating.get_by_role("button", name="Save").click()

    groups.get_by_role("textbox", name="Search Groups").fill(group_name)
    groups.get_by_role("button", name=re.compile(rf"^{group_name} 0 members")).click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name="Users").click()
    editing.get_by_role("textbox", name="Search").fill(member.name)
    with admin_page.expect_response(lambda response: response.url.endswith("/users/add")):
        editing.get_by_role("checkbox", name=member.name).click()

    admin_page.reload()
    groups.get_by_role("textbox", name="Search Groups").fill(group_name)
    expect(
        groups.get_by_role("button", name=re.compile(rf"^{group_name} 1 members"))
    ).to_be_visible()
