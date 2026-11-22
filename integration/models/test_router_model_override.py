"""Editing a router model's row in Settings > Models dropped its access and its routes, #29564.

Fix commit `b8de508db` (open-webui/open-webui#31309), made for arena models, which router models
replaced in 67863f310. Saving the model on the admin models page (to give it tools, say) writes
the model row under its id. That row's metadata replaced the settings' wholesale, so the access
grants and the models it picks from were gone: users lost the model even when it was public,
and its chats picked from every model. A router model now keeps its routes in the router
settings, which win over the row on merge, and its access on the row, which an edit keeps.

Discriminates: the arena tests passed on dev efe63bd34 and failed with b8de508db reverted (the
edited model listed without its grants and pool, the user no longer saw it or could chat with
it, and its chats left the pool); the unedited one passed on both. Retargeted to router models
on dev 53d741157, where all five pass.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.router_models import (
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    default_routes,
    save_router_model,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ROUTE_SYSTEM_PROMPT = "You are the only model this router picks."


@pytest.fixture
def routed_model(admin) -> Iterator[str]:
    """A preset readable by everyone, whose system prompt marks requests routed to it."""
    model_id = f"routed-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {"system": ROUTE_SYSTEM_PROMPT},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def router_model(admin, preserve, routed_model) -> str:
    """A public router model whose only route is `routed_model`."""
    preserve(ROUTER_MODELS_CONFIG)
    router_id = f"router-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        save_router_model(
            client,
            router_id,
            f"Router {router_id}",
            routes=default_routes(routed_model),
            access_grants=[EVERYONE_READS],
        )
    return router_id


def _edit_in_settings_models(admin: Actor, router_id: str) -> None:
    """What saving the router model's row on the admin models page stores."""
    with admin.client() as client:
        saved = client.post(
            "/api/v1/models/model/update",
            json={
                "id": router_id,
                "base_model_id": None,
                "name": f"Router {router_id}",
                "meta": {"description": "Edited on the models page", "toolIds": []},
                "params": {},
            },
        )
    assert saved.status_code == 200, saved.text


def _listed(client: httpx.Client, model_id: str) -> dict | None:
    models = client.get("/api/models", params={"refresh": True})
    assert models.status_code == 200, models.text
    return next((model for model in models.json()["data"] if model["id"] == model_id), None)


# ---------------------------------------------------------------- narrow


def test_an_edited_router_model_keeps_its_access_and_routes(admin, router_model, routed_model):
    _edit_in_settings_models(admin, router_model)

    with admin.client() as client:
        listed = _listed(client, router_model)

    assert listed is not None
    info = listed["info"]
    assert info["meta"].get("routes") == default_routes(routed_model), (
        f"the edit dropped the router's routes (#29564): {info['meta'].get('routes')}"
    )
    assert info["meta"].get("strategy") == "random"
    assert info.get("access_grants"), "the edit dropped the router model's access grants (#29564)"
    assert info["meta"].get("description") == "Edited on the models page", "the edit was lost"


def test_a_user_still_sees_and_chats_with_an_edited_public_router_model(
    admin, make_user, upstream, router_model
):
    _edit_in_settings_models(admin, router_model)
    member = make_user()
    upstream.queue(reply.text("from the router"))

    with member.client() as client:
        assert _listed(client, router_model) is not None, "the user lost the router model (#29564)"
        _, answer = ask(client, "hello router", model=router_model)

    assert answer["content"] == "from the router"


def test_chats_on_an_edited_router_model_stay_on_its_routes(
    admin, upstream, router_model, routed_model
):
    _edit_in_settings_models(admin, router_model)
    upstream.reset()

    picked = set()
    with admin.client() as client:
        _listed(client, router_model)
        for turn in range(6):
            _, answer = ask(client, f"router round {turn}", model=router_model)
            picked.add(answer.get("selectedModelId"))

    assert picked == {routed_model}, f"the router picked outside its routes (#29564): {picked}"
    system_prompts = [
        message["content"]
        for request in upstream.chat_requests()
        for message in request["messages"]
        if message["role"] == "system"
    ]
    assert system_prompts.count(ROUTE_SYSTEM_PROMPT) == 6


# ---------------------------------------------------------------- nearby


def test_an_unedited_router_model_is_listed_for_a_user(make_user, router_model, routed_model):
    member = make_user()
    with member.client() as client:
        listed = _listed(client, router_model)

    assert listed is not None
    assert listed["info"]["meta"]["routes"] == default_routes(routed_model)


def test_a_private_router_model_stays_hidden_after_an_edit(admin, make_user, preserve):
    preserve(ROUTER_MODELS_CONFIG)
    router_id = f"router-private-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        save_router_model(
            client, router_id, router_id, routes=default_routes(MOCK_MODEL_ID), access_grants=[]
        )
    _edit_in_settings_models(admin, router_id)

    with make_user().client() as client:
        assert _listed(client, router_id) is None
