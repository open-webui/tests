"""The model registry signature differed per worker, open-webui v0.11.2.

Fix commit `b8f279b8f` (#29264) in `backend/open_webui/utils/models.py` and
`backend/open_webui/utils/chat_variables.py`. `RedisDict.set` fingerprints the
serialized model list and skips the write when the fingerprint matches what is
already cached. Two places built that list by iterating a `set`, whose order is
per-process, so every worker produced a different serialization of an identical
list and every refresh rewrote the whole cache. Both now iterate in sorted order.

Discriminates: passes on v0.11.3, fails on v0.11.1 (where a model's `filters`
and a chat variable's field keys come out in set-iteration order, which is not
sorted and differs between worker processes).
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.regression

FILTER_IDS = [
    "zebra_filter",
    "yak_filter",
    "walrus_filter",
    "tapir_filter",
    "rhino_filter",
    "quail_filter",
    "possum_filter",
    "otter_filter",
    "narwhal_filter",
    "marmot_filter",
]

ALL_FIELD_PROPERTIES = (
    'type=text:label="City":placeholder="Berlin":default="Berlin":min=1:max=9'
    ":minlength=2:maxlength=8:step=1:required=true"
    ':options=["a","b"]'
)


@pytest.fixture(scope="session")
def models_util(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.models")


@pytest.fixture(scope="session")
def chat_variables(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.chat_variables")


@pytest.fixture(scope="session")
def socket_utils(owui_module) -> ModuleType:
    return owui_module("open_webui.socket.utils")


# ── driving the real model list build ────────────────────────────────────────


def _base_model(id: str) -> dict:
    return {"id": id, "name": id, "object": "model", "created": 0, "owned_by": "openai"}


def _function_record(id: str):
    return SimpleNamespace(
        id=id,
        name=id,
        meta=SimpleNamespace(description=f"{id} description", manifest={}),
    )


def _filter_module():
    return SimpleNamespace(toggle=True, icon_url=None)


async def _build_model_list(models_util, base_model_ids: list[str], filter_ids: list[str]) -> list:
    """Drive the real `utils.models.get_all_models` with only its I/O stubbed."""
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={"seed": {}}, BASE_MODELS=[])),
        state=SimpleNamespace(),
    )
    config = {
        "models.base_models_cache": False,
        "evaluation.arena.enable": False,
        "models.default_metadata": {},
    }
    active_filters = [(filter_id, True) for filter_id in filter_ids]
    functions = [_function_record(filter_id) for filter_id in filter_ids]
    cache = {filter_id: _filter_module() for filter_id in filter_ids}

    async def active_ids_by_type(function_type):
        return active_filters if function_type == "filter" else []

    with (
        patch.object(models_util.Config, "get_many", AsyncMock(return_value=config)),
        patch.object(models_util, "ENABLE_PLUGINS", True),
        patch.object(
            models_util,
            "get_all_base_models",
            AsyncMock(return_value=[_base_model(id) for id in base_model_ids]),
        ),
        patch.object(models_util.Models, "get_all_models", AsyncMock(return_value=[])),
        patch.object(
            models_util.Functions,
            "get_active_function_ids_by_type",
            AsyncMock(side_effect=active_ids_by_type),
        ),
        patch.object(
            models_util.Functions, "get_functions_by_ids", AsyncMock(return_value=functions)
        ),
        patch.object(
            models_util.Functions, "get_function_valves_by_ids", AsyncMock(return_value={})
        ),
        patch.object(models_util, "get_function_module_from_cache", AsyncMock(return_value=None)),
        patch.object(models_util, "get_functions_cache", MagicMock(return_value=cache)),
    ):
        return await models_util.get_all_models(request)


# ── the real RedisDict against an in-memory backing ──────────────────────────


class _FakeRedis:
    """The Redis surface `RedisDict.set` touches, plus a record of the writes."""

    def __init__(self):
        self.strings = {}
        self.hashes = {}
        self.hset_calls = 0

    def get(self, name):
        return self.strings.get(name)

    def set(self, name, value):
        self.strings[name] = value

    def delete(self, name):
        self.strings.pop(name, None)
        self.hashes.pop(name, None)

    def hkeys(self, name):
        return list(self.hashes.get(name, {}))

    def hset(self, name, mapping=None):
        self.hset_calls += 1
        self.hashes.setdefault(name, {}).update(mapping or {})

    def hdel(self, name, *keys):
        for key in keys:
            self.hashes.get(name, {}).pop(key, None)


def _worker_cache(socket_utils, redis):
    with patch.object(socket_utils, "get_redis_connection", MagicMock(return_value=redis)):
        return socket_utils.RedisDict("models", "redis://x", cache_set_signature=True)


def _as_registry(models: list) -> dict:
    return {model["id"]: model for model in models}


# ═════════════════════════════════════════════════════════════════════════════
# Narrow
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_model_filters_come_out_in_a_stable_order(models_util):
    """Narrow. `filter_ids` came straight off a set union, so the order of a
    model's `filters` was whatever the worker's string hashing produced. Ten ids
    landing in sorted order by chance is a one in 3.6 million event."""
    models = await _build_model_list(models_util, ["gpt-4o"], FILTER_IDS)

    emitted = [item["id"] for item in models[0]["filters"]]
    assert emitted == sorted(FILTER_IDS)


@pytest.mark.asyncio
async def test_filter_order_does_not_follow_the_configured_order(models_util):
    """Narrow. The same filters registered in a different order must serialize
    identically, otherwise the two workers disagree on the signature."""
    forward = await _build_model_list(models_util, ["gpt-4o"], FILTER_IDS)
    reversed_registration = await _build_model_list(
        models_util, ["gpt-4o"], list(reversed(FILTER_IDS))
    )

    assert [item["id"] for item in forward[0]["filters"]] == [
        item["id"] for item in reversed_registration[0]["filters"]
    ]
    assert [item["id"] for item in forward[0]["filters"]] == sorted(FILTER_IDS)


def test_chat_variable_field_keys_come_out_in_a_stable_order(chat_variables):
    """Narrow. `_safe_field` copied allowed keys while iterating a set literal, so
    the field dict's key order, and therefore its JSON bytes, varied per worker."""
    prompt = "{{chat.variables.city|" + ALL_FIELD_PROPERTIES + "}}"

    schema = chat_variables.get_chat_variables_schema(prompt)

    keys = list(schema["fields"][0])
    assert keys[0] == "key"
    assert len(keys) == 12
    assert keys[1:] == sorted(keys[1:])


# ═════════════════════════════════════════════════════════════════════════════
# Broad: an unchanged list writes nothing, a changed one writes
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_two_workers_building_the_same_list_write_the_cache_once(
    models_util, socket_utils
):
    """Broad. Second worker recomputes the same signature, so `set` returns before
    touching the hash."""
    redis = _FakeRedis()
    first = await _build_model_list(models_util, ["gpt-4o", "gpt-4o-mini"], FILTER_IDS)
    second = await _build_model_list(
        models_util, ["gpt-4o-mini", "gpt-4o"], list(reversed(FILTER_IDS))
    )

    _worker_cache(socket_utils, redis).set(_as_registry(first))
    assert redis.hset_calls == 1
    signature = redis.strings["models:signature"]

    _worker_cache(socket_utils, redis).set(_as_registry(second))
    assert redis.strings["models:signature"] == signature
    assert redis.hset_calls == 1
    assert sorted(redis.hashes["models"]) == ["gpt-4o", "gpt-4o-mini"]


@pytest.mark.asyncio
async def test_a_genuinely_changed_list_still_writes(models_util, socket_utils):
    """Broad. The skip is content-addressed, so an added model refreshes the cache."""
    redis = _FakeRedis()
    first = await _build_model_list(models_util, ["gpt-4o"], FILTER_IDS)
    grown = await _build_model_list(models_util, ["gpt-4o", "gpt-4o-mini"], FILTER_IDS)

    _worker_cache(socket_utils, redis).set(_as_registry(first))
    signature = redis.strings["models:signature"]
    _worker_cache(socket_utils, redis).set(_as_registry(grown))

    assert redis.strings["models:signature"] != signature
    assert redis.hset_calls == 2
    assert sorted(redis.hashes["models"]) == ["gpt-4o", "gpt-4o-mini"]


def test_signature_ignores_top_level_key_order(socket_utils):
    """Broad. The fingerprint walks sorted keys, so registry insertion order alone
    never forces a write."""
    redis = _FakeRedis()
    registry = {"a": {"id": "a"}, "b": {"id": "b"}}
    shuffled = {"b": {"id": "b"}, "a": {"id": "a"}}

    _worker_cache(socket_utils, redis).set(registry)
    _worker_cache(socket_utils, redis).set(shuffled)

    assert redis.hset_calls == 1


# ═════════════════════════════════════════════════════════════════════════════
# Nearby: unchanged edges
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_empty_model_list_builds_and_clears_the_cache(models_util, socket_utils):
    """Nearby. No base models means no listing, and an empty mapping clears rather
    than writes."""
    models = await _build_model_list(models_util, [], FILTER_IDS)
    assert models == []

    redis = _FakeRedis()
    redis.hashes["models"] = {"stale": "{}"}
    _worker_cache(socket_utils, redis).set(_as_registry(models))

    assert redis.hset_calls == 0
    assert "models" not in redis.hashes


@pytest.mark.asyncio
async def test_single_model_list_round_trips(models_util, socket_utils):
    """Nearby. One model, one write, one cached entry."""
    models = await _build_model_list(models_util, ["gpt-4o"], FILTER_IDS)
    assert [model["id"] for model in models] == ["gpt-4o"]

    redis = _FakeRedis()
    _worker_cache(socket_utils, redis).set(_as_registry(models))

    assert redis.hset_calls == 1
    assert list(redis.hashes["models"]) == ["gpt-4o"]


@pytest.mark.asyncio
async def test_model_with_no_filters_gets_an_empty_filter_list(models_util):
    """Nearby. No registered filters leaves the key present and empty."""
    models = await _build_model_list(models_util, ["gpt-4o"], [])

    assert models[0]["filters"] == []


def test_bare_chat_variable_still_defaults(chat_variables):
    """Nearby. A variable with no definition keeps its defaulted shape."""
    schema = chat_variables.get_chat_variables_schema("{{chat.variables.city}}")

    field = schema["fields"][0]
    assert field["key"] == "city"
    assert field["type"] == "text"
    assert field["required"] is False
