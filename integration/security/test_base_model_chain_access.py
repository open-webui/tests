"""Regression: a shared workspace model must not be a ladder to a base model the user cannot use.

open-webui 0.11.0 fix `fe4b31942` (#26905, issue #26900): `has_base_model_access` walked the
`base_model_id` chain and treated a base with no row in the `model` table (a raw provider
model) as unrestricted. Everywhere else such a model is admin-only: users do not see it and a
direct chat with it is refused. So a preset the admin shared, built on a raw provider model,
gave every user the underlying model. The fix makes that hop admin-only.

Twin of unit/security/test_base_model_chain_access.py.

Discriminates: passes on dev bbfa876af; with fe4b31942 reverted the user's chat through the
shared preset is answered by the raw provider model.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RAW_MODEL = "raw-provider-model"
PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


def _complete(client, model_id: str):
    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    return client.post("/api/chat/completions", json=body)


def _create_model(client, model_id: str, base_model_id: str | None, access_grants: list[dict]):
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": base_model_id,
            "name": f"Preset on {base_model_id}",
            "meta": {},
            "params": {},
            "access_grants": access_grants,
        },
    )
    assert created.status_code == 200, created.text


@pytest.fixture
def raw_model(admin, upstream):
    """A second provider model with no workspace entry, so only admins may use it.

    Yields `share(base_model_id, access_grants, model_id)`, which adds a workspace model on it.
    """
    upstream.models.append(RAW_MODEL)
    created: list[str] = []
    with admin.client() as client:
        client.get("/api/models?refresh=true").raise_for_status()

        def share(base_model_id=RAW_MODEL, access_grants=PUBLIC_READ, model_id=None) -> str:
            created.append(model_id or f"preset-{uuid.uuid4().hex[:8]}")
            _create_model(client, created[-1], base_model_id, access_grants)
            client.get("/api/models?refresh=true").raise_for_status()
            return created[-1]

        yield share
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id}).raise_for_status()
        upstream.models.remove(RAW_MODEL)
        client.get("/api/models?refresh=true")


def _models_asked_for(upstream) -> list[str]:
    return [request["model"] for request in upstream.chat_requests()]


def test_a_shared_preset_does_not_reach_an_unregistered_base_for_a_user(
    raw_model, make_user, upstream
):
    preset_id = raw_model()
    with make_user().client() as client:
        response = _complete(client, preset_id)

    assert response.status_code == 400 and "Model not found" in response.text, (
        f"a user reached the admin-only raw model {RAW_MODEL} through a shared preset "
        f"(#26900): HTTP {response.status_code} {response.text[:200]}"
    )
    assert RAW_MODEL not in _models_asked_for(upstream)


def test_an_admin_still_uses_a_preset_on_an_unregistered_base(raw_model, admin, upstream):
    preset_id = raw_model()
    with admin.client() as client:
        response = _complete(client, preset_id)

    assert response.status_code == 200, response.text
    assert _models_asked_for(upstream) == [RAW_MODEL]


def test_a_user_cannot_use_the_unregistered_model_directly(raw_model, make_user, upstream):
    with make_user().client() as client:
        response = _complete(client, RAW_MODEL)

    assert response.status_code == 400 and "Model not found" in response.text, response.text


def test_a_shared_preset_on_a_readable_registered_base_works_for_a_user(
    raw_model, make_user, upstream
):
    preset_id = raw_model(base_model_id="mock-model")
    with make_user().client() as client:
        response = _complete(client, preset_id)

    assert response.status_code == 200, response.text


def test_a_shared_preset_on_a_private_registered_base_stays_refused(raw_model, make_user, upstream):
    raw_model(base_model_id=None, access_grants=[], model_id=RAW_MODEL)
    preset_id = raw_model()
    with make_user().client() as client:
        response = _complete(client, preset_id)

    assert response.status_code == 400 and "Model not found" in response.text, response.text
