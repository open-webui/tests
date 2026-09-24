"""A model's filters and chat variable fields came out in a per-process order.

Fix `b8f279b8f` (#29264, open-webui 0.11.2) in `utils/models.py` and `utils/chat_variables.py`.
The Redis model registry skips its write when the content signature matches what is stored, but
a model's `filters` and each chat variable field's keys were built by iterating a `set`, whose
order follows the process's hash seed. Every worker serialized identical content differently
and rewrote the whole registry on every refresh. Both are now sorted, which `/api/models` shows.

Twin of unit/models/test_model_registry_signature_stability.py. The skip itself is pinned by
unit/models/test_shared_model_pool_cache.py; one process cannot show a second worker's order.

Discriminates: passes on upstream dev `bbfa876af`; with the `filter_ids.sort()` removed the
filters come out unsorted, and with `_safe_field` iterating the set again the field keys do
(ten ids, or eleven keys, landing sorted by chance is below one in three million).
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import httpx
import pytest

from harness.plugins import installed_function
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOGGLE_FILTER = """
class Filter:
    def __init__(self):
        self.toggle = True

    def inlet(self, body):
        return body
"""

EVERY_FIELD_PROPERTY = (
    'type=text:label="City":placeholder="Berlin":default="Berlin":min=1:max=9'
    ':minlength=2:maxlength=8:step=1:required=true:options=["a","b"]'
)


@pytest.fixture(scope="module")
def toggle_filters(admin) -> Iterator[list[str]]:
    """Ten active, non-global toggle filters, so only a model that names them lists them."""
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(installed_function(admin, TOGGLE_FILTER)) for _ in range(10)]


@contextlib.contextmanager
def workspace_model(client: httpx.Client, *, filter_ids=(), system: str = "") -> Iterator[str]:
    model_id = f"signature-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "name": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "meta": {"filterIds": list(filter_ids)},
            "params": {"system": system},
        },
    )
    assert created.status_code == 200, created.text
    try:
        yield model_id
    finally:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def listed(client: httpx.Client, model_id: str) -> dict:
    models = client.get("/api/models")
    assert models.status_code == 200, models.text
    return next(model for model in models.json()["data"] if model["id"] == model_id)


@pytest.fixture
def admin_client(admin) -> Iterator[httpx.Client]:
    with admin.client() as client:
        yield client


def test_a_models_filters_are_listed_in_sorted_order(admin_client, toggle_filters):
    registered = sorted(toggle_filters, reverse=True)
    with workspace_model(admin_client, filter_ids=registered) as model_id:
        filters = [item["id"] for item in listed(admin_client, model_id)["filters"]]

    assert filters == sorted(toggle_filters), "filters came out in the process's set order"


def test_chat_variable_field_keys_are_listed_in_sorted_order(admin_client):
    system = "{{chat.variables.city|" + EVERY_FIELD_PROPERTY + "}}"
    with workspace_model(admin_client, system=system) as model_id:
        schema = listed(admin_client, model_id)["info"]["meta"]["chat_variables_schema"]

    keys = list(schema["fields"][0])
    assert keys[0] == "key"
    assert len(keys) == 12
    assert keys[1:] == sorted(keys[1:]), "field keys came out in the process's set order"


def test_a_model_naming_no_filters_lists_none(admin_client):
    with workspace_model(admin_client) as model_id:
        assert listed(admin_client, model_id)["filters"] == []


def test_a_bare_chat_variable_keeps_its_defaults(admin_client):
    with workspace_model(admin_client, system="{{chat.variables.city}}") as model_id:
        schema = listed(admin_client, model_id)["info"]["meta"]["chat_variables_schema"]

    assert schema["fields"] == [{"key": "city", "type": "text", "required": False}]
