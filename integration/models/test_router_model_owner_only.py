"""Regression: a private router model with no access grants vanished for its own admin, #30013.

Fix 8c34af303 (PR #31858), made for arena models, which router models replaced in 67863f310.
With `BYPASS_ADMIN_ACCESS_CONTROL` off, a model left at its default access (Private, no grants)
was checked against grants nobody holds, so the model list, the chat request and the profile
image refused it to the admin who set it up, and nobody on the instance could use it. A router
model's access now sits on a model row owned by the admin who saved it, so a grantless one is
that admin's alone: another admin is held to its grants like a user, and a model shared with a
user stays hidden from every admin but its owner.

Discriminates: the arena tests passed on dev b859124f9 and failed with 8c34af303 reverted (the
admin's model list lacked the model, a chat with it was refused and its profile image was the
default one). Retargeted to router models on dev 53d741157, where all six pass.
"""

from __future__ import annotations

import base64
import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.router_models import ROUTER_MODELS_CONFIG, default_routes, save_router_model
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# a 1x1 transparent PNG
PROFILE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PROFILE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(PROFILE_PNG).decode()


@pytest.fixture
def restricted(instance_with) -> LaunchedInstance:
    """An instance where admins are held to access grants like everyone else."""
    return instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})


def save_router(restricted: LaunchedInstance, access_grants: list[dict]) -> str:
    router_id = f"router-{uuid.uuid4().hex[:8]}"
    with admin_of(restricted).client() as client:
        save_router_model(
            client,
            router_id,
            f"Router {router_id}",
            routes=default_routes(MOCK_MODEL_ID),
            access_grants=access_grants,
            meta={"profile_image_url": PROFILE_IMAGE_URL},
        )
    return router_id


@pytest.fixture
def private_router(restricted, preserve) -> Iterator[str]:
    """A router model left at its default access: Private, with no grants."""
    preserve(ROUTER_MODELS_CONFIG, on=restricted)
    yield save_router(restricted, access_grants=[])


def listed_ids(client: httpx.Client) -> set[str]:
    models = client.get("/api/models", params={"refresh": True})
    assert models.status_code == 200, models.text
    return {model["id"] for model in models.json()["data"]}


def profile_image(client: httpx.Client, model_id: str) -> httpx.Response:
    return client.get("/api/v1/models/model/profile/image", params={"id": model_id})


def refused_chat(restricted: LaunchedInstance, client: httpx.Client, model_id: str) -> None:
    """A chat with `model_id` is refused before it reaches the provider."""
    listed_ids(client)
    refused = client.post(
        "/api/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "let me in"}],
            "stream": False,
        },
    )
    assert refused.status_code >= 400, refused.text
    assert not restricted.upstream.chat_requests()


# ---------------------------------------------------------------- narrow


def test_its_admin_sees_a_private_router_model_without_grants(restricted, private_router):
    with admin_of(restricted).client() as client:
        assert private_router in listed_ids(client), "the router model is gone for its admin"


def test_its_admin_chats_with_a_private_router_model_without_grants(restricted, private_router):
    restricted.upstream.queue(reply.text("from the router", match=reply.answering("hello router")))
    with admin_of(restricted).client() as client:
        listed_ids(client)
        _, answer = ask(client, "hello router", model=private_router)

    assert answer["content"] == "from the router"


def test_its_admin_gets_the_profile_image_of_a_private_router_model(restricted, private_router):
    with admin_of(restricted).client() as client:
        image = profile_image(client, private_router)

    assert image.status_code == 200, image.text
    assert image.content == PROFILE_PNG, "the admin got the default image (#30013)"


# ---------------------------------------------------------------- nearby


def test_a_user_does_not_see_a_private_router_model_without_grants(restricted, private_router):
    with create_user(restricted).client() as client:
        assert private_router not in listed_ids(client)
        image = profile_image(client, private_router)

    assert image.content != PROFILE_PNG


def test_a_user_cannot_chat_with_a_private_router_model_without_grants(restricted, private_router):
    with create_user(restricted).client() as client:
        refused_chat(restricted, client, private_router)


def test_a_router_model_shared_with_a_user_is_theirs_and_not_another_admins(restricted, preserve):
    preserve(ROUTER_MODELS_CONFIG, on=restricted)
    member = create_user(restricted)
    other_admin = create_user(restricted, role="admin")
    grant = {"principal_type": "user", "principal_id": member.id, "permission": "read"}
    router_id = save_router(restricted, access_grants=[grant])

    with member.client() as client:
        assert router_id in listed_ids(client), "the user it is shared with does not see it"
    with admin_of(restricted).client() as client:
        assert router_id in listed_ids(client), "the admin who saved it lost it"
    with other_admin.client() as client:
        assert router_id not in listed_ids(client), "an admin it is not shared with sees it"
        refused_chat(restricted, client, router_id)
