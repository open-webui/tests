"""Journey: what an admin sets on a model's row in Admin Settings > Models reaches every user.

Each row carries a switch that enables the model, and its More menu sets it as a selected model for
new chats, pins it to everyone's sidebar, makes it private and hides it. A user who opens the app
afterwards finds the model gone from the selector once it is switched off, made private or hidden,
finds a new chat starting on it once it is a selected model, and finds it pinned in the sidebar
once it is a pinned model.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose row actions save the model as
it was (the enable switch skipping its toggle request, hide and privacy sending the stored meta
and grants, the selected and pinned saves sending the previous lists), every "switched off",
"private", "hidden", "selected" and "pinned" test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.model_selector import SELECTOR_BUTTON, model_options
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


@pytest.fixture
def preset(admin, preserve):
    """A public preset on the scripted model, under a name no other test uses."""
    preserve(MODELS_CONFIG)
    suffix = uuid.uuid4().hex[:8]
    form = {
        "id": f"atlas-{suffix}",
        "name": f"Atlas {suffix}",
        "base_model_id": "mock-model",
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    yield form
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": form["id"]})


def model_row(page: Page, name: str) -> Locator:
    page.goto("/admin/settings/models")
    settings = page.get_by_role("dialog")
    settings.get_by_role("textbox", name="Search Models").fill(name)
    row = settings.locator("#model-list > div").filter(has_text=name)
    expect(row).to_have_count(1)
    return row


def choose_from_menu(page: Page, row: Locator, entry: str) -> None:
    tooltip_button(row, "More").click()
    page.get_by_role("menu").get_by_role("button", name=entry).click()


def offered_to(page: Page, name: str) -> Locator:
    expect(page.get_by_role("button", name=SELECTOR_BUTTON)).to_be_visible()
    return model_options(page, name)


def test_a_model_is_offered_to_a_user_by_default(page_for, make_user, preset):
    page = page_for(make_user())

    expect(offered_to(page, preset["name"])).to_have_count(1)


def test_a_model_switched_off_in_its_row_leaves_the_users_selector(
    page_for, admin, make_user, preset
):
    row = model_row(page_for(admin), preset["name"])
    row.get_by_role("switch").click()
    expect(row.get_by_role("switch")).to_have_attribute("aria-checked", "false")

    page = page_for(make_user())

    expect(page.get_by_role("button", name=SELECTOR_BUTTON)).to_be_visible()
    expect(offered_to(page, preset["name"])).to_have_count(0)


def test_a_model_made_private_leaves_the_users_selector(page_for, admin, make_user, preset):
    admin_page = page_for(admin)
    row = model_row(admin_page, preset["name"])
    choose_from_menu(admin_page, row, "Make Private")
    expect(admin_page.get_by_text("Model is now private")).to_be_visible()

    page = page_for(make_user())

    expect(offered_to(page, preset["name"])).to_have_count(0)


def test_a_hidden_model_leaves_the_users_selector(page_for, admin, make_user, preset):
    admin_page = page_for(admin)
    row = model_row(admin_page, preset["name"])
    choose_from_menu(admin_page, row, "Hide Model")
    expect(admin_page.get_by_text(f"Model {preset['id']} is now hidden")).to_be_visible()

    page = page_for(make_user())

    expect(offered_to(page, preset["name"])).to_have_count(0)


def test_a_selected_model_is_where_a_new_users_chat_starts(page_for, admin, make_user, preset):
    admin_page = page_for(admin)
    row = model_row(admin_page, preset["name"])
    choose_from_menu(admin_page, row, "Set as Selected Model")
    expect(admin_page.get_by_text("Model added to selected models")).to_be_visible()

    page = page_for(make_user())

    expect(page.get_by_role("button", name=f"Selected model: {preset['name']}")).to_be_visible()


def test_a_pinned_model_shows_in_a_new_users_sidebar(page_for, admin, make_user, preset):
    admin_page = page_for(admin)
    row = model_row(admin_page, preset["name"])
    choose_from_menu(admin_page, row, "Set as Pinned Model")
    expect(admin_page.get_by_text("Model added to pinned models")).to_be_visible()

    page = page_for(make_user())
    page.get_by_role("button", name="Open Sidebar", exact=True).click()

    pinned = page.get_by_role("navigation", name="Chat history").get_by_role(
        "link", name=preset["name"]
    )
    expect(pinned).to_be_visible()
    expect(pinned).to_have_attribute("href", re.compile(f"model={preset['id']}"))
