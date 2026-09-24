"""Editing a workspace model wiped its base model when the edit omitted `base_model_id`.

Fix `54f31c630` (open-webui 0.10.2). `ModelForm.base_model_id` defaults to `None`, and the model
writer stores every column, so an update payload without the field detached the model from the
provider model it proxies. The router now restores the stored value when the payload did not set
the field, while an explicit value, including an explicit `null`, still wins.

Twin of unit/models/test_model_edit_preserves_base_model.py.

Discriminates: passes on upstream dev `bbfa876af`; with the router's `model_fields_set` guard
removed, `test_an_edit_without_base_model_id_keeps_it` fails (the GET returns a null base model).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import httpx
import pytest

from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def admin_client(admin) -> Iterator[httpx.Client]:
    with admin.client() as client:
        yield client


@pytest.fixture
def workspace_model(admin_client) -> Iterator[str]:
    """A workspace model on top of the provider's model, removed afterwards."""
    model_id = f"edit-keeps-base-{uuid.uuid4().hex[:8]}"
    created = admin_client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": "Before",
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"description": "before"},
            "params": {},
        },
    )
    assert created.status_code == 200, created.text
    yield model_id
    admin_client.post("/api/v1/models/model/delete", json={"id": model_id})


def _update(client: httpx.Client, model_id: str, **fields) -> httpx.Response:
    payload = {"id": model_id, "name": "After", "meta": {"description": "after"}, "params": {}}
    return client.post("/api/v1/models/model/update", json={**payload, **fields})


def _stored(client: httpx.Client, model_id: str) -> dict:
    response = client.get("/api/v1/models/model", params={"id": model_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_an_edit_without_base_model_id_keeps_it(admin_client, workspace_model):
    assert _update(admin_client, workspace_model).status_code == 200

    stored = _stored(admin_client, workspace_model)
    assert stored["meta"]["description"] == "after"
    assert stored["base_model_id"] == MOCK_MODEL_ID, (
        "an edit that did not send base_model_id detached the model from its base model"
    )


def test_an_edit_naming_another_base_model_switches_to_it(admin_client, workspace_model):
    assert _update(admin_client, workspace_model, base_model_id="another-base").status_code == 200

    assert _stored(admin_client, workspace_model)["base_model_id"] == "another-base"


def test_an_explicit_null_still_detaches_the_model(admin_client, workspace_model):
    assert _update(admin_client, workspace_model, base_model_id=None).status_code == 200

    assert _stored(admin_client, workspace_model)["base_model_id"] is None
