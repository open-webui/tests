"""Journey: the model editor's JSON Preview can be copied, and the copy is what the preview shows.

A builder opens one of their models in the workspace editor and presses Copy beside the JSON
Preview heading: the page confirms "Copied to clipboard" and the clipboard holds the model's JSON.
The copy is the JSON the preview shows, edited name included. Copy used to put the editor's
unedited starting state on the clipboard (capabilities empty, an edited name missing) while the
preview showed the current form (open-webui/open-webui#31955), fixed in dev 106aae70e.

Discriminates: in a frontend build without the Copy button the copy tests go red (no button). The
two tests that compare the copy with the preview pass on dev ebc6add67 and failed on dev 30f3f6a8f,
before 106aae70e (the unedited starting state was copied).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def builder(make_user):
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def own_model(builder) -> dict:
    model = {"id": f"quay-{uuid.uuid4().hex[:8]}", "name": f"Quay notes {uuid.uuid4().hex[:6]}"}
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={**model, "base_model_id": MOCK_MODEL_ID, "meta": {}, "params": {}},
        )
    assert created.status_code == 200, created.text
    return model


def open_editor_with_preview(page: Page, model: dict):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    beside_copy = editor.get_by_role("button", name="Copy", exact=True).locator("xpath=..")
    beside_copy.get_by_role("button", name="Show").click()
    return editor, editor.locator("textarea[readonly]")


def copied(page: Page) -> str:
    return page.evaluate("() => navigator.clipboard.readText()")


def test_the_copy_button_puts_the_models_json_on_the_clipboard(page_for, builder, own_model):
    page = page_for(builder)
    editor, preview = open_editor_with_preview(page, own_model)

    editor.get_by_role("button", name="Copy", exact=True).click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    on_clipboard = json.loads(copied(page))
    assert on_clipboard["id"] == own_model["id"]
    assert on_clipboard["name"] == own_model["name"]
    assert on_clipboard["base_model_id"] == MOCK_MODEL_ID


def test_the_copy_is_the_json_the_preview_shows(page_for, builder, own_model):
    page = page_for(builder)
    editor, preview = open_editor_with_preview(page, own_model)

    editor.get_by_role("button", name="Copy", exact=True).click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    assert json.loads(copied(page)) == json.loads(preview.input_value()), (
        "Copy put different JSON on the clipboard than the JSON Preview beside it shows (#31955)"
    )


def test_the_copy_matches_the_preview_after_the_name_is_edited(page_for, builder, own_model):
    page = page_for(builder)
    editor, preview = open_editor_with_preview(page, own_model)
    renamed = f"{own_model['name']} renamed"
    editor.get_by_placeholder("Model Name").fill(renamed)
    expect(preview).to_have_value(re.compile(re.escape(renamed)))

    editor.get_by_role("button", name="Copy", exact=True).click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    on_clipboard = json.loads(copied(page))
    assert on_clipboard["name"] == renamed, (
        f"Copy put the unedited name {on_clipboard['name']!r} on the clipboard while the "
        f"JSON Preview shows {renamed!r} (#31955)"
    )
