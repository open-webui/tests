"""Regression: a private arena model with no access grants was missing from the admin's selector.

Fix 8c34af303 (PR #31858, issue #30013). With `BYPASS_ADMIN_ACCESS_CONTROL` off, an arena model
left Private with no grants was refused to every admin, the one who created it included, so the
model selector never offered it. It is now admin-only: the admin is offered it and can chat with
it, a user is not offered it. Twin of integration/models/test_arena_model_admin_only.py.

Discriminates: passes on dev b859124f9, fails with 8c34af303 reverted (the admin's selector finds
no such model). The user's selector not offering the model, the security half, is the test's
first step; the admin chatting with it is the second.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.instance import LaunchedInstance
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import expect_reply, send
from utils.model_selector import model_options, select_model

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EVALUATION_CONFIG = ("/api/v1/evaluations/config", "/api/v1/evaluations/config")


@pytest.fixture
def restricted(instance_with) -> LaunchedInstance:
    """An instance where admins are held to access grants like everyone else."""
    return instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})


@pytest.fixture
def private_arena(restricted, preserve) -> Iterator[str]:
    """An arena model left at its default access, named uniquely for the selector search."""
    preserve(EVALUATION_CONFIG, on=restricted)
    name = f"Arena {uuid.uuid4().hex[:8]}"
    arena = {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "meta": {"model_ids": [MOCK_MODEL_ID], "access_grants": []},
    }
    with admin_of(restricted).client() as client:
        saved = client.post(
            "/api/v1/evaluations/config",
            json={"ENABLE_EVALUATION_ARENA_MODELS": True, "EVALUATION_ARENA_MODELS": [arena]},
        )
    assert saved.status_code == 200, saved.text
    yield name


def test_a_private_arena_model_is_offered_to_an_admin_and_not_to_a_user(
    page_for, restricted, private_arena
):
    user_page = page_for(create_user(restricted))
    expect(model_options(user_page, MOCK_MODEL_ID)).to_be_visible()
    expect(model_options(user_page, private_arena)).to_have_count(0)

    restricted.upstream.queue(reply.text("from the arena", match=reply.answering("hello arena")))
    page = page_for(admin_of(restricted))

    select_model(page, private_arena)
    send(page, "hello arena")

    expect_reply(page, "from the arena")
