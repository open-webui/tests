"""Regression: a non-admin could store a workspace model under a provider model's id.

open-webui 0.11.1, fix `ea55d3879` (maintainer refac, no PR): `POST /api/v1/models/create`,
`/import` and `/model/update` let any holder of `workspace.models` store an entry whose id is a
model a connection serves, or one with no base model. Such an entry is no preset; it replaces the
provider's model for everyone, so the user's system prompt reached every chat anyone started with
what looked like the stock model. The fix requires a base model on non-admin writes, refuses an id
a connection serves (an Ollama id also without its tag), with 401 on create and a skipped entry on
import, and refuses clearing a base model on update. Admins keep the unrestricted path.

Each test serves a fresh id from the scripted provider or an Ollama listener, so no workspace row
claims it yet (the harness gives `mock-model` a row when it publishes it).

Twin of unit/security/test_workspace_model_shadowing.py.

Discriminates: passes on bbfa876af, fails with ea55d3879 reverted (create and import store the
shadowing entries, update clears the base model and the admin's chat with the provider model
carries the user's system prompt); the nearby tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

from harness.chat import ask
from harness.listener import json_answer
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PLANTED_PROMPT = "planted-system-prompt"
OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")


def _fresh_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _model(model_id: str, base_model_id: str | None, name: str = "Hijack") -> dict:
    return {
        "id": model_id,
        "base_model_id": base_model_id,
        "name": name,
        "meta": {},
        "params": {"system": PLANTED_PROMPT},
    }


def _write(actor, route: str, model: dict):
    """One write the way the workspace does it: the editor form, a model file or an edit."""
    path, body = {
        "create": ("/api/v1/models/create", model),
        "import": ("/api/v1/models/import", {"models": [model]}),
        "update": ("/api/v1/models/model/update", model),
    }[route]
    with actor.client() as client:
        return client.post(path, json=body)


def _refreshed_models(admin) -> dict[str, dict]:
    """The admin's model picker, rebuilt from the connections and the workspace rows."""
    with admin.client() as client:
        listed = client.get("/api/models", params={"refresh": "true"})
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry for entry in listed.json()["data"]}


def _stored(admin, model_id: str) -> dict | None:
    with admin.client() as client:
        found = client.get("/api/v1/models/model", params={"id": model_id})
    if found.status_code == 404:
        return None
    assert found.status_code == 200, found.text
    return found.json()


def _delete(admin, *model_ids: str) -> None:
    with admin.client() as client:
        for model_id in model_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def provider_model_id(instance, admin):
    """A model the scripted provider serves and no workspace row claims."""
    model_id = _fresh_id("provider-only")
    instance.upstream.models.append(model_id)
    assert model_id in _refreshed_models(admin)
    yield model_id
    instance.upstream.models.remove(model_id)
    _delete(admin, model_id)
    _refreshed_models(admin)


@pytest.fixture
def ollama_model_id(admin, preserve, listener):
    """A tagged model an Ollama connection serves, e.g. `llama-1a2b3c4d:latest`."""
    model_id = f"{_fresh_id('llama')}:latest"
    listener.route(
        "GET", "/api/tags", json_answer({"models": [{"name": model_id, "model": model_id}]})
    )
    preserve(OLLAMA_CONFIG)
    config = {
        "ENABLE_OLLAMA_API": True,
        "OLLAMA_BASE_URLS": [listener.base_url],
        "OLLAMA_API_CONFIGS": {},
    }
    with admin.client() as client:
        assert client.post(OLLAMA_CONFIG[1], json=config).status_code == 200
    assert model_id in _refreshed_models(admin)
    yield model_id
    _delete(admin, model_id, model_id.split(":")[0])


@pytest.fixture
def workspace_user(admin, preserve, make_user):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["workspace"].update(models=True, models_import=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    return make_user()


@pytest.fixture
def preset_id(workspace_user, admin):
    """A preset the workspace user owns, built on the published model."""
    model_id = _fresh_id("preset")
    assert _write(workspace_user, "create", _model(model_id, MOCK_MODEL_ID)).status_code == 200
    yield model_id
    _delete(admin, model_id)


# narrow: the three writes that took over a provider model


@pytest.mark.parametrize("base_model_id", [None, MOCK_MODEL_ID], ids=["no-base", "with-base"])
def test_create_refuses_a_provider_model_id(
    workspace_user, admin, provider_model_id, base_model_id
):
    created = _write(workspace_user, "create", _model(provider_model_id, base_model_id))

    assert created.status_code == 401, (
        f"a non-admin created a workspace model under the provider id {provider_model_id} "
        f"(HTTP {created.status_code}), taking it over for everyone (ea55d3879)"
    )
    assert _stored(admin, provider_model_id) is None


def test_import_skips_a_provider_model_id_and_keeps_the_rest(
    workspace_user, admin, provider_model_id
):
    helper_id = _fresh_id("helper")
    models = [_model(provider_model_id, MOCK_MODEL_ID), _model(helper_id, MOCK_MODEL_ID, "Helper")]
    with workspace_user.client() as client:
        imported = client.post("/api/v1/models/import", json={"models": models})

    assert imported.status_code == 200, imported.text
    assert _stored(admin, provider_model_id) is None, (
        "importing a model file stored an entry under a provider model id (ea55d3879)"
    )
    assert _stored(admin, helper_id)["base_model_id"] == MOCK_MODEL_ID
    _delete(admin, helper_id)


def test_update_refuses_clearing_the_base_model(workspace_user, admin, preset_id):
    updated = _write(workspace_user, "update", _model(preset_id, None))

    assert updated.status_code == 401, (
        f"a non-admin cleared the base model of their preset (HTTP {updated.status_code}), "
        "turning it into a standalone entry (ea55d3879)"
    )
    assert _stored(admin, preset_id)["base_model_id"] == MOCK_MODEL_ID


def test_import_does_not_clear_the_base_model(workspace_user, admin, preset_id):
    _write(workspace_user, "import", _model(preset_id, None))

    assert _stored(admin, preset_id)["base_model_id"] == MOCK_MODEL_ID, (
        "re-importing an owned preset with an empty base model cleared it (ea55d3879)"
    )


# broad: no user write claims a connection's model or skips the base model


@pytest.mark.parametrize("route", ["create", "import"])
@pytest.mark.parametrize("tag", ["tagged", "untagged"])
def test_no_user_write_claims_an_ollama_model_id(
    workspace_user, admin, ollama_model_id, route, tag
):
    model_id = ollama_model_id if tag == "tagged" else ollama_model_id.split(":")[0]

    _write(workspace_user, route, _model(model_id, MOCK_MODEL_ID))

    assert _stored(admin, model_id) is None, (
        f"{route} stored {model_id!r}, which resolves to the Ollama model {ollama_model_id!r} "
        "(ea55d3879)"
    )


@pytest.mark.parametrize("route", ["create", "import"])
def test_no_user_write_stores_an_entry_without_a_base_model(workspace_user, admin, route):
    standalone_id = _fresh_id("standalone")

    _write(workspace_user, route, _model(standalone_id, None))

    assert _stored(admin, standalone_id) is None, (
        f"{route} stored a model with no base model, which claims the id ahead of any "
        "provider that serves it (ea55d3879)"
    )


def test_no_user_write_plants_a_prompt_in_the_provider_model(
    workspace_user, admin, provider_model_id, upstream
):
    for route in ("create", "import"):
        for base_model_id in (None, MOCK_MODEL_ID):
            _write(workspace_user, route, _model(provider_model_id, base_model_id))
    picker_entry = _refreshed_models(admin)[provider_model_id]

    with admin.client() as client:
        ask(client, "hello", model=provider_model_id)

    assert picker_entry["name"] == provider_model_id, "the picker shows the user's entry"
    assert PLANTED_PROMPT not in str(upstream.chat_requests()[-1]["messages"]), (
        "a user's system prompt reached the admin's chat with a provider model (ea55d3879)"
    )


# nearby: ordinary workspace models and admin writes keep working


def test_user_can_create_a_preset_on_a_free_id(workspace_user, admin):
    model_id = _fresh_id("preset")

    created = _write(workspace_user, "create", _model(model_id, MOCK_MODEL_ID))

    assert created.status_code == 200, created.text
    assert created.json()["id"] == model_id
    _delete(admin, model_id)


def test_user_cannot_take_another_users_workspace_id(workspace_user, admin):
    model_id = _fresh_id("team")
    assert _write(admin, "create", _model(model_id, MOCK_MODEL_ID, "Team")).status_code == 200

    taken = _write(workspace_user, "create", _model(model_id, MOCK_MODEL_ID))

    assert taken.status_code == 401
    assert _stored(admin, model_id)["name"] == "Team"
    _delete(admin, model_id)


@pytest.mark.parametrize("route", ["update", "import"])
def test_user_can_edit_a_preset_that_keeps_its_base_model(workspace_user, admin, preset_id, route):
    written = _write(workspace_user, route, _model(preset_id, MOCK_MODEL_ID, "Renamed"))

    assert written.status_code == 200, written.text
    assert _stored(admin, preset_id)["name"] == "Renamed"


def test_user_can_edit_a_writable_entry_that_never_had_a_base_model(workspace_user, admin):
    model_id = _fresh_id("house")
    assert _write(admin, "create", _model(model_id, None)).status_code == 200
    grants = [
        {"principal_type": "user", "principal_id": workspace_user.id, "permission": permission}
        for permission in ("read", "write")
    ]
    with admin.client() as client:
        shared = client.post(
            "/api/v1/models/model/access/update", json={"id": model_id, "access_grants": grants}
        )
    assert shared.status_code == 200, shared.text

    updated = _write(workspace_user, "update", _model(model_id, None, "Renamed"))

    assert updated.status_code == 200, updated.text
    assert _stored(admin, model_id)["name"] == "Renamed"
    _delete(admin, model_id)


@pytest.mark.parametrize("route", ["create", "import", "update"])
def test_admin_writes_are_not_restricted(admin, provider_model_id, route):
    if route == "update":
        assert _write(admin, "create", _model(provider_model_id, MOCK_MODEL_ID)).status_code == 200

    written = _write(admin, route, _model(provider_model_id, None, "Override"))

    assert written.status_code == 200, written.text
    stored = _stored(admin, provider_model_id)
    assert stored["name"] == "Override"
    assert stored["base_model_id"] is None
