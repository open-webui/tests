"""Regression: a private router model with no access grants was missing from its admin's selector.

Fix 8c34af303 (PR #31858, issue #30013), made for arena models, which router models replaced in
67863f310. With `BYPASS_ADMIN_ACCESS_CONTROL` off, a model left Private with no grants was
refused to every admin, the one who created it included, so the model selector never offered it.
A router model's access sits on a model row owned by the admin who saved it: that admin is offered
it and can chat with it, a user is not offered it. Twin of
integration/models/test_router_model_owner_only.py, which also covers another admin.

Discriminates: the arena test passed on dev b859124f9 and failed with 8c34af303 reverted (the
admin's selector found no such model). Retargeted to router models on dev 53d741157, where it
passes. The user's selector not offering the model, the security half, is the test's first step;
the admin chatting with it is the second.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.instance import LaunchedInstance
from harness.router_models import ROUTER_MODELS_CONFIG, default_routes, save_router_model
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import expect_reply, send
from utils.model_selector import model_options, select_model

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def restricted(instance_with) -> LaunchedInstance:
    """An instance where admins are held to access grants like everyone else."""
    return instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})


@pytest.fixture
def private_router(restricted, preserve) -> Iterator[str]:
    """A router model left at its default access, named uniquely for the selector search."""
    preserve(ROUTER_MODELS_CONFIG, on=restricted)
    name = f"Router {uuid.uuid4().hex[:8]}"
    with admin_of(restricted).client() as client:
        save_router_model(
            client,
            name.lower().replace(" ", "-"),
            name,
            routes=default_routes(MOCK_MODEL_ID),
            access_grants=[],
        )
    yield name


def test_a_private_router_model_is_offered_to_its_admin_and_not_to_a_user(
    page_for, restricted, private_router
):
    user_page = page_for(create_user(restricted))
    expect(model_options(user_page, MOCK_MODEL_ID)).to_be_visible()
    expect(model_options(user_page, private_router)).to_have_count(0)

    restricted.upstream.queue(reply.text("from the router", match=reply.answering("hello router")))
    page = page_for(admin_of(restricted))

    select_model(page, private_router)
    send(page, "hello router")

    expect_reply(page, "from the router")
