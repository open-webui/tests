"""Regression: the workspace model editor let a user save a model under a provider model's id.

open-webui 0.11.1, fix `ea55d3879` (maintainer refac, no PR): a user with `workspace.models` could
create a workspace model whose id is one a connection serves, which replaces the provider's model
for everyone. The editor only refuses ids already in the user's own model list, and a provider
model the user cannot see is not in it, so the create went through. The fix refuses the id on the
server and the editor shows that refusal.

Twin of unit/security/test_workspace_model_shadowing.py.

Discriminates: passes on bbfa876af, fails with ea55d3879 reverted (the editor reports the model
created and the provider id is taken over).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _refresh_models(admin) -> list[str]:
    with admin.client() as client:
        listed = client.get("/api/models", params={"refresh": "true"})
    assert listed.status_code == 200, listed.text
    return [entry["id"] for entry in listed.json()["data"]]


@pytest.fixture
def provider_model_id(upstream, admin):
    """A model the scripted provider serves that the user cannot see and no row claims."""
    model_id = f"provider-only-{uuid.uuid4().hex[:8]}"
    upstream.models.append(model_id)
    assert model_id in _refresh_models(admin)
    yield model_id
    upstream.models.remove(model_id)
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})
    _refresh_models(admin)


@pytest.fixture
def workspace_user(admin, preserve, make_user):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["workspace"]["models"] = True
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    return make_user()


def test_the_editor_refuses_a_provider_model_id(page_for, admin, workspace_user, provider_model_id):
    page = page_for(workspace_user)
    page.goto("/workspace/models/create")

    page.get_by_placeholder("Model Name").fill("Hijack")
    page.get_by_placeholder("Model ID").fill(provider_model_id)
    page.get_by_role("button", name=re.compile("Select a base model")).click()
    page.get_by_role("option", name=f"Select {MOCK_MODEL_ID} model").click()
    page.get_by_role("button", name="Save & Create").click()

    outcome = page.get_by_text(re.compile("already registered|Model created successfully"))
    expect(outcome).to_be_visible()
    assert "already registered" in outcome.inner_text(), (
        "the editor created a workspace model under a provider model's id (ea55d3879)"
    )
    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": provider_model_id})
    assert stored.status_code == 404, (
        f"the editor saved a workspace model under the provider id {provider_model_id} (ea55d3879)"
    )
