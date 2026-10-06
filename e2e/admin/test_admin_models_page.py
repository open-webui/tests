"""Journey: what an admin sets on a model's row in Admin Settings > Models reaches every user.

Each row carries a switch that enables the model, and its More menu sets it as a selected model for
new chats, pins it to everyone's sidebar, makes it private and hides it. A user who opens the app
afterwards finds the model gone from the selector once it is switched off, made private or hidden,
finds a new chat starting on it once it is a selected model, and finds it pinned in the sidebar
once it is a pinned model. Disable All and Hide All in the Actions menu take only the models the
search shows out of the selector, Enable All and Show All bring them back, and a model exported
with Export and deleted comes back for users through Import.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose row actions save the model as
it was (the enable switch skipping its toggle request, hide and privacy sending the stored meta
and grants, the selected and pinned saves sending the previous lists), every "switched off",
"private", "hidden", "selected" and "pinned" test fails. In one whose Disable All sends nothing,
whose Export saves an empty list and whose Import sends an empty list, the Disable All and the
export tests fail; in one whose Enable All, Hide All and Show All send nothing, their tests fail.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.model_selector import SELECTOR_BUTTON, model_options
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


def create_preset(admin, label: str) -> dict:
    suffix = uuid.uuid4().hex[:8]
    form = {
        "id": f"{label.lower()}-{suffix}",
        "name": f"{label} {suffix}",
        "base_model_id": "mock-model",
        "meta": {"description": f"{label} for the harbour"},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    return form


def delete_preset(admin, form: dict) -> None:
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": form["id"]})


@pytest.fixture
def preset(admin, preserve):
    """A public preset on the scripted model, under a name no other test uses."""
    preserve(MODELS_CONFIG)
    form = create_preset(admin, "Atlas")
    yield form
    delete_preset(admin, form)


@pytest.fixture
def second_preset(admin):
    form = create_preset(admin, "Boreas")
    yield form
    delete_preset(admin, form)


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


def choose_action(page: Page, action: str) -> None:
    settings = page.get_by_role("dialog")
    settings.get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("button", name=action, exact=True).click()


def test_disable_all_switches_off_only_the_searched_models(
    page_for, admin, make_user, preset, second_preset
):
    admin_page = page_for(admin)
    row = model_row(admin_page, preset["name"])
    choose_action(admin_page, "Disable All")
    expect(row.get_by_role("switch")).to_have_attribute("aria-checked", "false")

    page = page_for(make_user())

    expect(offered_to(page, preset["name"])).to_have_count(0)
    expect(offered_to(page, second_preset["name"])).to_have_count(1)


def test_an_exported_model_comes_back_through_import(page_for, admin, make_user, preset):
    admin_page = page_for(admin)
    model_row(admin_page, preset["name"])
    with admin_page.expect_download() as downloaded:
        choose_action(admin_page, "Export")
    exported = json.loads(Path(downloaded.value.path()).read_text())
    [saved] = [entry for entry in exported if entry["id"] == preset["id"]]
    delete_preset(admin, preset)

    admin_page.goto("/admin/settings/models")
    with admin_page.expect_file_chooser() as chooser:
        choose_action(admin_page, "Import")
    chooser.value.set_files(
        files=[
            {
                "name": "models.json",
                "mimeType": "application/json",
                "buffer": json.dumps([saved]).encode(),
            }
        ]
    )
    expect(admin_page.get_by_text("Models imported successfully")).to_be_visible()

    page = page_for(make_user())
    expect(offered_to(page, preset["name"])).to_have_count(1)
    with admin.client() as client:
        restored = client.get("/api/v1/models/model", params={"id": preset["id"]}).json()
    assert restored["meta"]["description"] == "Atlas for the harbour"


def put_away(admin, preset: dict, how: str) -> None:
    """Switch the preset off or hide it over the API, the way the row and its menu do."""
    with admin.client() as client:
        if how == "disabled":
            switched = client.post("/api/v1/models/model/toggle", params={"id": preset["id"]})
            switched.raise_for_status()
            return
        stored = client.get("/api/v1/models/model", params={"id": preset["id"]}).json()
        stored["meta"] = {**stored["meta"], "hidden": True}
        updated = client.post(
            "/api/v1/models/model/update", params={"id": preset["id"]}, json=stored
        )
        updated.raise_for_status()


@pytest.mark.parametrize(("action", "how"), [("Enable All", "disabled"), ("Show All", "hidden")])
def test_a_bulk_action_brings_back_the_searched_model(
    action, how, page_for, admin, make_user, preset
):
    put_away(admin, preset, how)
    expect(offered_to(page_for(make_user()), preset["name"])).to_have_count(0)
    admin_page = page_for(admin)
    model_row(admin_page, preset["name"])

    choose_action(admin_page, action)

    expect(offered_to(page_for(make_user()), preset["name"])).to_have_count(1)


def test_hide_all_takes_the_searched_model_out_of_the_selector(
    page_for, admin, make_user, preset, second_preset
):
    admin_page = page_for(admin)
    model_row(admin_page, preset["name"])

    choose_action(admin_page, "Hide All")
    expect(admin_page.get_by_text("All models are now hidden")).to_be_visible()

    page = page_for(make_user())
    expect(offered_to(page, preset["name"])).to_have_count(0)
    expect(offered_to(page, second_preset["name"])).to_have_count(1)
