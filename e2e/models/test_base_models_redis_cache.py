"""A connection change on one instance left the model selector on another showing the old list.

open-webui v0.11.2 (`6609918bf`): with the base model cache on, each instance of a deployment
kept the provider model list in its own memory, so after an admin changed a connection on one
instance, a person on another still picked from the models it had cached. The list is now
shared through Redis and a connection change deletes it for every instance.

Two instances share one database and one real Redis, with `ENABLE_BASE_MODELS_CACHE` on; the
provider is a `listener` added as one more OpenAI connection, whose models the test changes.

Twin of unit/models/test_base_models_redis_cache.py.

Discriminates: passes on dev ef67cc3fa; with 6609918bf reverted the second instance's selector
never offers the provider's new model; the first instance's own selector, the test's second
step, offers it on both.
"""

from __future__ import annotations

import re
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from harness import backends
from harness.actors import create_user
from harness.listener import json_answer, listening
from harness.second_provider import OPENAI_CONFIG, attach
from utils.model_selector import model_search

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

OLD_MODEL = "old-provider-model"
NEW_MODEL = "new-provider-model"
CACHE_ON = {"ENABLE_BASE_MODELS_CACHE": "true"}


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def provider() -> Iterator[dict]:
    """A listener serving the models in `provider["serving"]` at the time it is asked."""
    state = {"serving": [OLD_MODEL]}
    with listening() as listener:
        state["listener"] = listener
        yield state


def _serve_current_models(provider: dict) -> None:
    def models(_request):
        return json_answer({"object": "list", "data": [{"id": n} for n in provider["serving"]]})

    provider["listener"].route("GET", "/v1/models", models)


@pytest.fixture(scope="module")
def workers(shared_redis, instance_with, provider) -> tuple:
    """Two instances on one database and one Redis, the provider attached through the first."""
    first = instance_with({**CACHE_ON, "REDIS_URL": shared_redis, "WEBUI_NAME": "worker-a"})
    second = instance_with(
        {
            **CACHE_ON,
            "REDIS_URL": shared_redis,
            "DATABASE_URL": first.database_url,
            "WEBUI_NAME": "worker-b",
        }
    )
    with first.client() as client:
        attach(client, provider["listener"], OLD_MODEL)
    _serve_current_models(provider)
    return first, second


def _change_the_connection_on(instance) -> None:
    with instance.client() as client:
        current = client.get(OPENAI_CONFIG[0])
        current.raise_for_status()
        client.post(OPENAI_CONFIG[1], json=current.json()).raise_for_status()


def _expect_offered(page: Page, model_id: str) -> None:
    page.get_by_role("button", name=re.compile("^Selected model")).click()
    model_search(page).fill(model_id)
    available = page.get_by_role("listbox", name="Available models")
    expect(available.get_by_role("option", name=f"Select {model_id} model")).to_be_visible()


@pytest.fixture
def new_model_behind_a_connection_change(workers, provider) -> tuple:
    """Both instances have listed the old model; then the connection changes on the first."""
    first, second = workers
    provider["serving"] = [OLD_MODEL]
    for instance in (first, second):
        with create_user(instance, role="admin").client() as client:
            client.get("/api/models").raise_for_status()
    provider["serving"] = [NEW_MODEL]
    _change_the_connection_on(first)
    return workers


def test_the_other_instances_selector_offers_the_new_model(
    new_model_behind_a_connection_change, page_for
):
    first, second = new_model_behind_a_connection_change

    _expect_offered(page_for(create_user(second, role="admin")), NEW_MODEL)
    _expect_offered(page_for(create_user(first, role="admin")), NEW_MODEL)
