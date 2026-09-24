"""Regressions for two v0.11.1 repairs of settings persisted in the wrong shape.

1. Default model settings (commit 5c05608e3a). Instances whose `ui.default_models` or
   `ui.default_pinned_models` row was stored as a list instead of the comma-separated string the
   app reads had their defaults silently ignored. The boot's config repair now joins such a row.
   The instance here boots on a data directory whose legacy `config.json` holds the rows: its
   import writes each key verbatim before the repair runs, so the boot meets the old shapes
   (`harness.prepared_data.LEGACY_CONFIG_ROWS`).
2. Connection tags saved as plain strings (commit 8be4c5fa6a, issue #28749). `/api/models` did
   `[tag.get('name') for tag in ...]` inside a bare `try/except`, so a connection whose tags or
   listed `info.meta.tags` were strings lost every tag, and a listed model with `info: null`
   took the whole listing down. The fix normalises both.

Twin of unit/migrations/test_startup_repairs.py.
Discriminates: passes on dev bbfa876af; with the default-model pass of
`Config.repair_config_rows` removed from a copy the list comes back as a list, and with the
`main.py` half of 8be4c5fa6a reverted the string tags come back empty and the `info: null`
model fails the listing.
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness.listener import json_answer
from harness.prepared_data import with_legacy_config
from harness.second_provider import OPENAI_CONFIG, attach
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TAGGED = {"id": "tagged-model", "info": {"meta": {"tags": ["alpha", "beta"]}}}
BARE = {"id": "bare-model", "info": None}
DICT_TAGGED = {
    "id": "dict-tagged-model",
    "info": {"meta": {"tags": [{"name": "delta"}], "profile_image_url": "data:image/png;base64,A"}},
}


@pytest.fixture(scope="module")
def repaired_config(instance_with, tmp_path_factory) -> dict:
    with with_legacy_config(instance_with, tmp_path_factory).client() as client:
        config = client.get("/api/config")
    config.raise_for_status()
    return config.json()


@pytest.mark.slow
def test_a_list_shaped_default_models_row_reads_back_as_the_string(repaired_config):
    assert repaired_config["default_models"] == f"{MOCK_MODEL_ID},second-model", (
        "a list-shaped default models row was not repaired, so the defaults stay ignored"
    )


@pytest.mark.slow
def test_a_string_shaped_default_models_row_is_left_as_it_is(repaired_config):
    assert repaired_config["default_pinned_models"] == f"{MOCK_MODEL_ID},second-model"


@pytest.mark.slow
def test_flattened_permission_rows_are_reassembled(repaired_config):
    permissions = repaired_config["permissions"]
    assert permissions["chat"]["controls"] is False, permissions
    assert permissions["workspace"]["models"] is True, permissions


@pytest.fixture
def connect(admin, listener, preserve):
    """`connect(tags)` adds one more connection with its tags saved as given; returns the
    admin's client."""
    preserve(OPENAI_CONFIG)
    clients = []

    def factory(tags: list) -> httpx.Client:
        client = admin.client()
        clients.append(client)
        attach(client, listener, TAGGED["id"], tags=tags)
        return client

    yield factory
    for client in clients:
        client.close()


def _listed(client, listener, *models: dict) -> dict[str, dict]:
    """`/api/models`, by id, once the connection lists `models`."""
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": list(models)}))
    wanted = {model["id"] for model in models}
    deadline = time.monotonic() + 10
    while True:
        listed = client.get("/api/models", params={"refresh": "true"})
        assert listed.status_code == 200, f"the model listing failed: {listed.text}"
        found = {model["id"]: model for model in listed.json()["data"]}
        if wanted <= found.keys() or time.monotonic() > deadline:
            return found
        time.sleep(0.2)


def _names(tags: list[dict]) -> list[str]:
    return sorted(tag["name"] for tag in tags)


def test_plain_string_tags_come_back_as_tags(connect, listener):
    tagged = _listed(connect(["gamma"]), listener, TAGGED)[TAGGED["id"]]

    assert _names(tagged["tags"]) == ["alpha", "beta", "gamma"], tagged["tags"]
    assert tagged["info"]["meta"]["tags"] == [{"name": "alpha"}, {"name": "beta"}], (
        "the listed tags the connection editor reads back were not normalised (#28749)"
    )


def test_a_model_listed_with_null_info_does_not_break_the_listing(connect, listener):
    listed = _listed(connect([]), listener, BARE)

    assert listed[BARE["id"]]["tags"] == []


def test_tag_objects_and_the_profile_image_strip_still_work(connect, listener):
    listed = _listed(connect([{"name": "gamma"}]), listener, DICT_TAGGED)[DICT_TAGGED["id"]]

    assert _names(listed["tags"]) == ["delta", "gamma"]
    assert "profile_image_url" not in listed["info"]["meta"]
