"""Journey: an admin clones, filters, republishes and exports models in Admin Settings > Models.

A row's More menu clones the model: the clone opens in the model editor under the name with
"(Clone)" added, and once saved it is a second model that users pick next to the first. The Select
view filter lists only the matching models: Hidden shows the hidden one, Visible the others, and
Workspace Models the presets without the base models. The list opens on Available (461cc7aff):
the models a connection offers and the presets, while settings saved for a model no connection
offers any more wait under Unavailable. Make Public on a private model brings it back to the
users' selector. Export in a row's More menu downloads that one model, not the list. Each test
works as a fresh admin on presets of its own.

Discriminates: passes on dev ebc6add67; in a frontend copy, the clone entry opening the editor
under the original name turns the clone test red, the Hidden and Workspace Models views listing
every model turn the filter test red, Make Public saving the grants it had turns the republish test
red (users still do not see the model) and Export saving an empty list turns the export test red.
With 461cc7aff reverted in a frontend build (the list opens on All, with no Available or
Unavailable view) the Available test and the view filter test are red, the other three pass.
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


def _create_preset(
    admin,
    label: str,
    public: bool = True,
    hidden: bool = False,
    base_model_id: str | None = "mock-model",
) -> dict:
    suffix = uuid.uuid4().hex[:8]
    form = {
        "id": f"{label.lower()}-{suffix}",
        "name": f"{label} {suffix}",
        "base_model_id": base_model_id,
        "meta": {"description": f"{label} for the harbour", "hidden": hidden},
        "params": {},
        "access_grants": [EVERYONE_READS] if public else [],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    return form


@pytest.fixture
def presets(admin, preserve):
    """Presets made by a test, deleted afterwards along with any clone of them."""
    preserve(MODELS_CONFIG)
    made: list[dict] = []

    def create(label: str, **options) -> dict:
        made.append(_create_preset(admin, label, **options))
        return made[-1]

    yield create
    with admin.client() as client:
        for form in made:
            client.post("/api/v1/models/model/delete", json={"id": form["id"]})
            client.post("/api/v1/models/model/delete", json={"id": f"{form['id']}-clone"})


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


def _model_row(page: Page, name: str) -> Locator:
    page.goto("/admin/settings/models")
    settings = page.get_by_role("dialog")
    settings.get_by_role("textbox", name="Search Models").fill(name)
    row = settings.locator("#model-list > div").filter(has_text=name)
    expect(row).to_have_count(1)
    return row


def _more_menu_entry(page: Page, row: Locator, entry: str) -> Locator:
    tooltip_button(row, "More").click()
    return page.get_by_role("menu").get_by_role("button", name=entry)


def _offered_to(page: Page, name: str) -> Locator:
    expect(page.get_by_role("button", name=SELECTOR_BUTTON)).to_be_visible()
    return model_options(page, name)


def _choose_view(page: Page, current: str, view: str) -> None:
    page.get_by_role("dialog").get_by_role("button", name=current, exact=True).click()
    page.get_by_role("button", name=view, exact=True).click()


def test_a_cloned_model_is_a_second_model_users_can_pick(admin_page, page_for, make_user, presets):
    preset = presets("Atlas")
    clone_name = f"{preset['name']} (Clone)"
    row = _model_row(admin_page, preset["name"])

    _more_menu_entry(admin_page, row, "Clone").click()

    editor = admin_page.get_by_role("main")
    expect(editor.get_by_role("textbox", name="Model Name")).to_have_value(clone_name)
    editor.get_by_role("button", name="Save & Create").click()
    expect(admin_page.get_by_text("Model created successfully!")).to_be_visible()

    page = page_for(make_user())
    expect(_offered_to(page, clone_name)).to_have_count(1)
    expect(_offered_to(page, preset["name"])).to_have_count(1)


def test_the_view_filter_lists_only_the_matching_models(admin_page, presets):
    shown = presets("Atlas")
    hidden = presets("Boreas", hidden=True)
    admin_page.goto("/admin/settings/models")
    rows = admin_page.get_by_role("dialog").locator("#model-list > div")
    expect(rows.filter(has_text=shown["name"])).to_have_count(1)
    mine = rows.filter(has_text=re.compile(f"{shown['name']}|{hidden['name']}"))

    _choose_view(admin_page, "Available", "Hidden")
    expect(rows.filter(has_text=hidden["name"])).to_have_count(1)
    expect(rows.filter(has_text=shown["name"])).to_have_count(0)

    _choose_view(admin_page, "Hidden", "Visible")
    expect(rows.filter(has_text=shown["name"])).to_have_count(1)
    expect(rows.filter(has_text=hidden["name"])).to_have_count(0)

    _choose_view(admin_page, "Visible", "Base Models")
    expect(rows.first).to_be_visible()
    expect(mine).to_have_count(0)

    _choose_view(admin_page, "Base Models", "Workspace Models")
    expect(mine).to_have_count(2)
    expect(rows.filter(has_text="mock-model")).to_have_count(0)


def test_the_list_opens_on_available_models_and_keeps_unavailable_ones_apart(admin_page, presets):
    preset = presets("Atlas")
    # Settings saved the way the panel saves a base model's, for a model no connection offers.
    retired = presets("Retired", base_model_id=None)
    admin_page.goto("/admin/settings/models")
    settings = admin_page.get_by_role("dialog")
    rows = settings.locator("#model-list > div")

    expect(settings.get_by_role("button", name="Available", exact=True)).to_be_visible()
    expect(rows.filter(has_text=preset["name"])).to_have_count(1)
    expect(rows.filter(has_text="mock-model").first).to_be_visible()
    expect(rows.filter(has_text=retired["name"])).to_have_count(0)

    _choose_view(admin_page, "Available", "Unavailable")
    expect(rows.filter(has_text=retired["name"])).to_have_count(1)
    expect(rows.filter(has_text=preset["name"])).to_have_count(0)
    expect(rows.filter(has_text="mock-model")).to_have_count(0)

    _choose_view(admin_page, "Unavailable", "All")
    expect(rows.filter(has_text=retired["name"])).to_have_count(1)
    expect(rows.filter(has_text=preset["name"])).to_have_count(1)


def test_a_private_model_made_public_is_offered_to_users_again(
    admin_page, page_for, make_user, presets
):
    preset = presets("Atlas", public=False)
    expect(_offered_to(page_for(make_user()), preset["name"])).to_have_count(0)
    row = _model_row(admin_page, preset["name"])

    with admin_page.expect_response(
        lambda response: response.request.method == "POST" and "/models" in response.url
    ):
        _more_menu_entry(admin_page, row, "Make Public").click()
    expect(admin_page.get_by_text("Model is now public")).to_be_visible()

    expect(_offered_to(page_for(make_user()), preset["name"])).to_have_count(1)


def test_a_rows_export_downloads_that_one_model(admin_page, presets):
    preset = presets("Atlas")
    presets("Boreas")
    row = _model_row(admin_page, preset["name"])

    with admin_page.expect_download() as downloaded:
        _more_menu_entry(admin_page, row, "Export").click()

    exported = json.loads(Path(downloaded.value.path()).read_text())
    assert [entry["id"] for entry in exported] == [preset["id"]]
    assert exported[0]["name"] == preset["name"]
