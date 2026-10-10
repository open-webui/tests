"""Regression: the model editor's JSON Preview showed the model as last saved, not as edited.

Issue open-webui/open-webui#31848, fix `571dafae3` (PR open-webui/open-webui#31851). The editor
keeps its edits in separate fields and only folded them into the model object when Save & Update
ran, so the JSON Preview under the form showed the stored model whatever had been typed. The
preview now builds the object from the live fields: name, system prompt, advanced params and
capabilities appear as they are edited, before anything is saved.

The Copy button beside the preview is covered by test_model_editor_json_copy.py.

Discriminates: passes on the dev b859124f9 build, fails on that build with 571dafae3 reverted (the
preview still shows the old name, no system prompt, no params and the old capabilities).
Retargeted for d4879a98b, whose JSON Preview and Advanced Params headings are the buttons that
show them: passes on dev 0401b7522 (3 of 3) and fails in a build of it whose preview leaves out
the edited params.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def preset(builder) -> dict:
    model = {
        "id": f"preview-{uuid.uuid4().hex[:8]}",
        "name": f"Harbour guide {uuid.uuid4().hex[:6]}",
    }
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    return model


def _open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def _preview(editor: Locator) -> dict:
    """The model object the JSON Preview shows right now."""
    textarea = editor.locator("textarea[readonly]")
    return json.loads(textarea.input_value())


def _show_preview(editor: Locator) -> None:
    editor.get_by_role("button", name="JSON Preview").click()
    expect(editor.locator("textarea[readonly]")).to_be_visible()


def _stored_name(account, model: dict) -> str:
    with account.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model["id"]})
    assert stored.status_code == 200, stored.text
    return stored.json()["name"]


def test_the_json_preview_shows_unsaved_changes_of_the_editor(page_for, builder, preset):
    page = page_for(builder)
    editor = _open_editor(page, preset)
    _show_preview(editor)
    assert _preview(editor)["name"] == preset["name"]

    new_name = f"Lighthouse guide {uuid.uuid4().hex[:6]}"
    system_prompt = "Answer as the lighthouse keeper."
    editor.get_by_placeholder("Model Name").fill(new_name)
    editor.get_by_role("textbox", name=re.compile("^Write your model system prompt")).fill(
        system_prompt
    )
    editor.get_by_role("button", name="Advanced Params").click()
    editor.get_by_text("Temperature", exact=True).locator("xpath=..").get_by_role(
        "button", name="Default"
    ).click()
    editor.get_by_role("spinbutton", name="Temperature").fill("0.3")
    editor.get_by_role("checkbox", name="File Upload", exact=True).click()

    expect(editor.locator("textarea[readonly]")).to_have_value(re.compile(re.escape(new_name)))
    shown = _preview(editor)
    assert shown["name"] == new_name
    assert shown["params"]["system"] == system_prompt
    assert shown["params"]["temperature"] == 0.3
    assert shown["meta"]["capabilities"]["file_upload"] is False
    assert _stored_name(builder, preset) == preset["name"], "the preview saved the model"
