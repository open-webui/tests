"""Saving a workspace model from its editor wiped a base model the editor could not list.

Fix `54f31c630` (open-webui 0.10.2), frontend half. When the model's base model was missing from
the model list the editor loads (retired, hidden or not visible to the editor), `ModelEditor`
nulled `base_model_id` in edit mode too, so the next save either detached the model or was refused
with "Base Model is required". It now keeps the stored value while editing.

Twin of unit/models/test_model_edit_preserves_base_model.py.

Discriminates: passes on upstream dev `bbfa876af`; with `ModelEditor.svelte` back on
`else if (!(edit && model.base_model_id === model.id))` the save is refused and the page stays
in the editor.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def model_on_a_retired_base(admin):
    """A workspace model whose base model is no longer offered by any connection."""
    model_id = f"retired-base-edit-{uuid.uuid4().hex[:8]}"
    base_model_id = f"retired-base-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": "Retired base",
                "base_model_id": base_model_id,
                "meta": {"description": "before"},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
        yield model_id, base_model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_saving_a_new_description_keeps_the_base_model(page_for, admin, model_on_a_retired_base):
    model_id, base_model_id = model_on_a_retired_base
    page = page_for(admin)
    page.goto(f"/workspace/models/edit?id={model_id}")

    description = page.get_by_placeholder("Add a short description about what this model does")
    expect(description).to_have_value("before")
    description.fill("after")
    page.get_by_role("button", name="Save & Update").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))

    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model_id}).json()
    assert stored["meta"]["description"] == "after"
    assert stored["base_model_id"] == base_model_id
