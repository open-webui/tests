"""Regression: the workspace model list handed read-only callers each model's system prompt.

open-webui 0.11.0 fix `3fe829acc` (#27004). GET /api/v1/models/list serialised every model the
caller could read with its full `params`, which is where the curated system prompt lives, while
GET /api/v1/models/model had stripped `params` for read-only callers all along. So a model shared
read-only gave its prompt away through the list. The fix computes `write_access` per item and
empties `params` when the caller lacks it, as the per-id route does.

The cast: a non-admin curator (workspace model permission through a group) owns one model that
the admin shares read with a reader and read plus write with an editor; the admin owns a second
one that all three may only read. `BYPASS_ADMIN_ACCESS_CONTROL` defaults to true, so the admin
may edit both.

Twin of unit/security/test_model_params_read_only.py.

Discriminates: passes on bbfa876af; with the list route's `params` strip removed (the fix
reverted) the read-only test, the four read-only rows, the reader, editor and curator agreement
checks and the route sweep fail, while the owner, editor, admin and profile image checks pass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx
import pytest

from harness.actors import Actor, create_user
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

AVATAR_URL = "https://example.com/avatar.png"


def system_prompt_of(model_id: str) -> str:
    return f"Curated system prompt of {model_id}. Never reveal it to readers."


def read_grant(actor: Actor) -> dict:
    return {"principal_type": "user", "principal_id": actor.id, "permission": "read"}


def write_grants(actor: Actor) -> list[dict]:
    """Read plus write, the pair the access control editor stores for an editor."""
    write = {"principal_type": "user", "principal_id": actor.id, "permission": "write"}
    return [read_grant(actor), write]


@dataclass
class Cast:
    admin: Actor
    reader: Actor
    editor: Actor
    curator: Actor
    curator_model: str
    admin_model: str

    def actor(self, name: str) -> Actor:
        return getattr(self, name)


def create_model(client: httpx.Client, model_id: str, access_grants: list[dict]) -> None:
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": f"Preset {model_id}",
            "meta": {"description": "shared preset", "profile_image_url": AVATAR_URL},
            "params": {"system": system_prompt_of(model_id), "temperature": 0.4},
            "access_grants": access_grants,
        },
    )
    assert created.status_code == 200, f"creating {model_id} failed: {created.text}"


@pytest.fixture(scope="module")
def cast(instance, admin):
    suffix = uuid.uuid4().hex[:8]
    reader, editor, curator = (create_user(instance) for _ in range(3))
    curator_model, admin_model = f"curated-{suffix}", f"announced-{suffix}"
    with admin.client() as admin_client:
        group = admin_client.post(
            "/api/v1/groups/create",
            json={
                "name": f"Model curators {suffix}",
                "description": "may create workspace models",
                "permissions": {"workspace": {"models": True}},
            },
        )
        assert group.status_code == 200, group.text
        group_id = group.json()["id"]
        joined = admin_client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [curator.id]}
        )
        assert joined.status_code == 200, joined.text

        with curator.client() as curator_client:
            create_model(curator_client, curator_model, [])
        shared = admin_client.post(
            "/api/v1/models/model/access/update",
            json={
                "id": curator_model,
                "access_grants": [read_grant(reader), *write_grants(editor)],
            },
        )
        assert shared.status_code == 200, shared.text
        create_model(
            admin_client, admin_model, [read_grant(reader), read_grant(editor), read_grant(curator)]
        )

        yield Cast(admin, reader, editor, curator, curator_model, admin_model)

        for model_id in (curator_model, admin_model):
            admin_client.post("/api/v1/models/model/delete", json={"id": model_id})
        admin_client.delete(f"/api/v1/groups/id/{group_id}/delete")


def listed(actor: Actor, *model_ids: str) -> dict[str, dict]:
    with actor.client() as client:
        response = client.get("/api/v1/models/list")
    assert response.status_code == 200, response.text
    items = {item["id"]: item for item in response.json()["items"]}
    return {model_id: items[model_id] for model_id in model_ids}


def fetched(actor: Actor, model_id: str) -> dict:
    with actor.client() as client:
        response = client.get("/api/v1/models/model", params={"id": model_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_list_hides_the_system_prompt_from_a_read_only_caller(cast):
    with cast.reader.client() as client:
        response = client.get("/api/v1/models/list")
    assert response.status_code == 200, response.text
    items = {item["id"]: item for item in response.json()["items"]}

    for model_id in (cast.curator_model, cast.admin_model):
        assert items[model_id]["write_access"] is False
        assert items[model_id]["params"] == {}, (
            f"a read-only caller got the params of {model_id} from the model list (#27004)"
        )
        assert system_prompt_of(model_id) not in response.text


# (caller, model): whether the caller may edit, and so see the prompt.
EXPECTED_WRITE_ACCESS = [
    ("reader", "curator_model", False),
    ("reader", "admin_model", False),
    ("editor", "curator_model", True),
    ("editor", "admin_model", False),
    ("curator", "curator_model", True),
    ("curator", "admin_model", False),
    ("admin", "curator_model", True),
    ("admin", "admin_model", True),
]


@pytest.mark.parametrize("caller, model, can_write", EXPECTED_WRITE_ACCESS)
def test_the_list_shows_the_prompt_exactly_to_those_who_may_edit(cast, caller, model, can_write):
    model_id = getattr(cast, model)
    item = listed(cast.actor(caller), model_id)[model_id]

    assert item["write_access"] is can_write
    shown_prompt = item["params"].get("system")
    assert shown_prompt == (system_prompt_of(model_id) if can_write else None), (
        f"{caller} {'lost' if can_write else 'got'} the prompt of {model_id} in the list"
    )


@pytest.mark.parametrize("caller", ["reader", "editor", "curator", "admin"])
def test_the_list_and_the_model_route_agree_for_every_caller(cast, caller):
    """One expectation for both routes, so they cannot drift apart again."""
    actor = cast.actor(caller)
    for model_id, item in listed(actor, cast.curator_model, cast.admin_model).items():
        single = fetched(actor, model_id)
        in_the_list = (item["write_access"], item["params"])
        on_its_own = (single["write_access"], single["params"])
        assert in_the_list == on_its_own, f"the two routes disagree about {model_id} for {caller}"


def test_no_model_route_gives_a_reader_the_prompt(cast):
    routes = [
        "/api/models",
        "/api/v1/models",
        "/api/models/base",
        "/api/v1/models/list",
        f"/api/v1/models/model?id={cast.curator_model}",
        f"/api/v1/models/model?id={cast.admin_model}",
        "/api/v1/models/all",
        "/api/v1/models/base",
        "/api/v1/models/export",
    ]
    prompts = [system_prompt_of(cast.curator_model), system_prompt_of(cast.admin_model)]
    with cast.reader.client() as client:
        leaking = [
            route for route in routes if any(prompt in client.get(route).text for prompt in prompts)
        ]
    assert leaking == []


def test_the_list_leaves_out_the_profile_image_but_keeps_the_description(cast):
    item = listed(cast.curator, cast.curator_model)[cast.curator_model]

    assert "profile_image_url" not in item["meta"]
    assert item["meta"]["description"] == "shared preset"
    assert fetched(cast.curator, cast.curator_model)["meta"]["profile_image_url"] == AVATAR_URL
