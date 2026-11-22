"""A public router model vanished from users' model selector once an admin edited it, #29564.

Fix commit `b8de508db` (open-webui/open-webui#31309), made for arena models, which router models
replaced in 67863f310. Saving the model's row on the admin models page (to give it tools, say)
replaced its metadata with the row's, dropping the access set with the model, so every non-admin
user lost it. The access now sits on the row, which an edit keeps. Twin of
integration/models/test_router_model_override.py.

Discriminates: the arena tests passed on dev efe63bd34; with b8de508db reverted in a backend copy
the user's selector no longer offered the edited model; the unedited one was offered on both.
Retargeted to router models on dev 53d741157, where both pass.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.router_models import (
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    default_routes,
    save_router_model,
)
from harness.upstream import MOCK_MODEL_ID
from utils.model_selector import model_search

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def router(admin, preserve) -> tuple[str, str]:
    """A public router model on the scripted model, named uniquely for the selector search."""
    preserve(ROUTER_MODELS_CONFIG)
    router_id = f"router-{uuid.uuid4().hex[:8]}"
    name = f"Router {router_id[-8:]}"
    with admin.client() as client:
        save_router_model(
            client,
            router_id,
            name,
            routes=default_routes(MOCK_MODEL_ID),
            access_grants=[EVERYONE_READS],
        )
    return router_id, name


def _edit_in_settings_models(admin: Actor, router_id: str, name: str) -> None:
    """What saving the router model's row on the admin models page stores."""
    with admin.client() as client:
        saved = client.post(
            "/api/v1/models/model/update",
            json={
                "id": router_id,
                "base_model_id": None,
                "name": name,
                "meta": {"description": "Edited on the models page", "toolIds": []},
                "params": {},
            },
        )
        assert saved.status_code == 200, saved.text
        client.get("/api/models", params={"refresh": True}).raise_for_status()


def _expect_offered(page: Page, name: str) -> None:
    page.goto("/")
    page.get_by_role("button", name=re.compile("^Selected model")).click()
    model_search(page).fill(name)
    available = page.get_by_role("listbox", name="Available models")
    expect(available.get_by_role("option", name=f"Select {name} model")).to_be_visible()


def test_a_user_is_still_offered_an_edited_public_router_model(page_for, make_user, admin, router):
    router_id, name = router
    _edit_in_settings_models(admin, router_id, name)

    _expect_offered(page_for(make_user()), name)


# ---------------------------------------------------------------- nearby


def test_a_user_is_offered_an_unedited_public_router_model(page_for, make_user, router):
    _, name = router
    _expect_offered(page_for(make_user()), name)
