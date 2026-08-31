"""The base model list moved into Redis, open-webui v0.11.2 (`6609918bf`).

`utils.models.get_all_models` cached the assembled base model list only in `app.state`, so every
worker rebuilt it independently and a config change in one worker left the others serving a stale
list. The list is now read from and written to Redis under `{REDIS_KEY_PREFIX}:models:base`, a
`refresh=True` call deletes that key and empties `app.state.BASE_MODELS`, and both provider
routers delete the same key when their connection config changes (`openai.update_config` now
routes through `clear_openai_model_cache` instead of repeating its body).

Discriminates: passes on v0.11.2, fails on v0.11.1 (pre-fix never reads, writes or deletes the
shared Redis key, so a worker with a cold `app.state` rebuilds from the providers instead).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.regression

CACHE_KEY_SUFFIX = ":models:base"


@pytest.fixture(scope="module")
def models_util(owui_module):
    return owui_module("open_webui.utils.models")


@pytest.fixture(scope="module")
def openai_router(owui_module):
    return owui_module("open_webui.routers.openai")


@pytest.fixture(scope="module")
def ollama_router(owui_module):
    return owui_module("open_webui.routers.ollama")


@pytest.fixture(scope="module")
def cache_key(owui_module):
    return owui_module("open_webui.env").REDIS_KEY_PREFIX + CACHE_KEY_SUFFIX


def _base_model(id: str) -> dict:
    return {"id": id, "name": id, "object": "model", "created": 0, "owned_by": "openai"}


class FakeRedis:
    def __init__(self, values: dict | None = None):
        self.values = dict(values or {})
        self.deletes = []
        self.writes = []

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *args, **kwargs):
        self.values[key] = value
        self.writes.append(key)

    async def delete(self, key):
        self.deletes.append(key)
        self.values.pop(key, None)


def _request(redis=None, **state):
    state.setdefault("MODELS", {})
    state.setdefault("BASE_MODELS", [])
    app_state = SimpleNamespace(redis=redis, **state)
    return SimpleNamespace(
        app=SimpleNamespace(state=app_state),
        state=SimpleNamespace(),
        headers={},
        cookies={},
    )


async def _run(models_util, request, rebuilt: list[dict], cache_enabled=True, refresh=False):
    """Drive the real `utils.models.get_all_models`, counting provider rebuilds."""
    rebuilds = []

    async def get_all_base_models(_request, user=None):
        rebuilds.append(user)
        return [dict(model) for model in rebuilt]

    config = {
        "models.base_models_cache": cache_enabled,
        "evaluation.arena.enable": False,
        "models.default_metadata": {},
    }
    with (
        patch.object(models_util.Config, "get_many", AsyncMock(return_value=config)),
        patch.object(models_util, "get_all_base_models", get_all_base_models),
        patch.object(models_util, "ENABLE_PLUGINS", False),
        patch.object(models_util.Models, "get_all_models", AsyncMock(return_value=[])),
        patch.object(models_util.Functions, "get_functions_by_ids", AsyncMock(return_value=[])),
        patch.object(
            models_util.Functions, "get_function_valves_by_ids", AsyncMock(return_value={})
        ),
        patch.object(models_util, "get_functions_cache", MagicMock(return_value={})),
    ):
        result = await models_util.get_all_models(request, refresh=refresh)
    return result, rebuilds


# ═════════════════════════════════════════════════════════════════════════════
# 6609918bf. The base model list is shared through Redis
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_base_models_are_served_from_the_redis_cache(models_util, cache_key):
    """Narrow. A worker with a cold `app.state` must take the list another worker cached; pre-fix
    it ignored Redis and rebuilt straight from the providers."""
    redis = FakeRedis({cache_key: json.dumps([_base_model("cached-model")])})
    request = _request(redis=redis)

    result, rebuilds = await _run(models_util, request, rebuilt=[_base_model("rebuilt-model")])

    assert [model["id"] for model in result] == ["cached-model"]
    assert rebuilds == []
    assert request.app.state.BASE_MODELS == [_base_model("cached-model")]


@pytest.mark.asyncio
async def test_a_rebuilt_base_model_list_is_written_to_redis(models_util, cache_key):
    """Narrow. Without the write-back the shared cache is never populated."""
    redis = FakeRedis()
    request = _request(redis=redis)

    await _run(models_util, request, rebuilt=[_base_model("rebuilt-model")])

    assert redis.writes == [cache_key]
    assert json.loads(redis.values[cache_key]) == [_base_model("rebuilt-model")]


@pytest.mark.asyncio
async def test_refresh_drops_the_shared_cache_before_rebuilding(models_util, cache_key):
    """Narrow. Pre-fix `refresh=True` only cleared the two provider caches, so the shared key
    kept serving the stale list to every other worker."""
    redis = FakeRedis({cache_key: json.dumps([_base_model("stale-model")])})
    request = _request(redis=redis, BASE_MODELS=[_base_model("stale-model")])

    result, rebuilds = await _run(
        models_util, request, rebuilt=[_base_model("fresh-model")], refresh=True
    )

    assert redis.deletes == [cache_key]
    assert rebuilds != []
    assert [model["id"] for model in result] == ["fresh-model"]


@pytest.mark.asyncio
async def test_clearing_the_openai_model_cache_drops_the_shared_key(openai_router, cache_key):
    """Broad. Every path that invalidates provider models must drop the same key the model
    assembler writes, otherwise a connection change is invisible to the other workers."""
    redis = FakeRedis({cache_key: json.dumps([_base_model("stale-model")])})
    request = _request(redis=redis, OPENAI_MODELS={})

    await openai_router.clear_openai_model_cache(request)

    assert redis.deletes == [cache_key]
    assert cache_key not in redis.values


def test_all_three_modules_agree_on_the_shared_cache_key(
    models_util, openai_router, ollama_router, cache_key
):
    """Broad. A router deleting a different key than the assembler writes is a silent no-op."""
    keys = {
        module.__name__: getattr(module, "BASE_MODELS_CACHE_KEY", None)
        for module in (models_util, openai_router, ollama_router)
    }
    assert set(keys.values()) == {cache_key}, keys


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "ollama"])
async def test_a_connection_config_change_drops_the_shared_key(
    openai_router, ollama_router, cache_key, provider
):
    """Broad. Changing a provider's connections must invalidate the list every worker reads.
    Pre-fix neither handler touched Redis, and openai's inlined its own invalidation body so the
    delete later added to `clear_openai_model_cache` would have missed this path anyway."""
    router = openai_router if provider == "openai" else ollama_router
    form = (
        openai_router.OpenAIConfigForm(
            ENABLE_OPENAI_API=True,
            OPENAI_API_BASE_URLS=[],
            OPENAI_API_KEYS=[],
            OPENAI_API_CONFIGS={},
        )
        if provider == "openai"
        else ollama_router.OllamaConfigForm(
            ENABLE_OLLAMA_API=True, OLLAMA_BASE_URLS=[], OLLAMA_API_CONFIGS={}
        )
    )
    redis = FakeRedis({cache_key: json.dumps([_base_model("stale-model")])})
    request = _request(redis=redis, OPENAI_MODELS={}, OLLAMA_MODELS={})

    with (
        patch.object(router.Config, "upsert", AsyncMock()),
        patch.object(router, "publish_event", AsyncMock()),
    ):
        await router.update_config(request, form, user=SimpleNamespace(id="admin"))

    assert redis.deletes == [cache_key]
    assert cache_key not in redis.values


@pytest.mark.asyncio
async def test_without_redis_the_process_local_state_cache_is_still_used(models_util):
    """Nearby. Single-worker deployments keep the old in-process behaviour."""
    request = _request(MODELS={"seed": {}}, BASE_MODELS=[_base_model("state-model")])

    result, rebuilds = await _run(models_util, request, rebuilt=[_base_model("rebuilt-model")])

    assert [model["id"] for model in result] == ["state-model"]
    assert rebuilds == []


@pytest.mark.asyncio
async def test_cache_disabled_always_rebuilds_and_never_writes(models_util, cache_key):
    """Nearby. The setting still wins over both cache layers."""
    redis = FakeRedis({cache_key: json.dumps([_base_model("cached-model")])})
    request = _request(redis=redis, MODELS={"seed": {}}, BASE_MODELS=[_base_model("state-model")])

    result, rebuilds = await _run(
        models_util, request, rebuilt=[_base_model("rebuilt-model")], cache_enabled=False
    )

    assert [model["id"] for model in result] == ["rebuilt-model"]
    assert rebuilds != []
    assert redis.writes == []


@pytest.mark.asyncio
async def test_an_empty_rebuild_falls_back_to_the_last_known_base_models(models_util):
    """Nearby. Providers that all fail must not wipe the list users are already seeing."""
    request = _request(
        redis=FakeRedis(), MODELS={"seed": {}}, BASE_MODELS=[_base_model("state-model")]
    )

    result, _ = await _run(models_util, request, rebuilt=[], cache_enabled=False)

    assert [model["id"] for model in result] == ["state-model"]
