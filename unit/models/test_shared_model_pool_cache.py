"""Regression: the shared model pool behind Redis refetched and rewrote on every touch.

Two 0.11.4 fixes share the Redis-backed model registry:

* `aceda892b` (PR #28176, issue #28167): with `WEBSOCKET_MANAGER=redis` the
  model pool is a `RedisDict`, and resolving a model id fetched from Redis, so
  a chat request asked about a couple of models by pulling the whole pool out
  and JSON-decoding every entry, several times per request, on the synchronous
  Redis client. The pool is now a `CachedRedisDict` in `socket/main.py`: each
  worker keeps its own copy and refetches only when the signature key changes.
* `649c012e` (a `refac`): `get_all_models` wrote the pool to Redis on every
  refresh, even when nothing changed, because Ollama attaches a countdown
  (`expires_at`) to a model it holds in memory and the fingerprint covered it.
  The registry copy now drops `ollama.expires_at`, so the moving countdown stays
  in the API response but the write is skipped for a genuinely unchanged list.

Discriminates: passes on 344ea5306 (0.11.4), fails on aceda892b^ / 649c012e^
(a read pulls the whole hash every time and a no-op refresh rewrites it).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.regression


class FakeRedis:
    """In-memory stand-in for the sync redis client the pool dict talks to."""

    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}
        self.hgetall_calls = 0
        self.hset_calls = 0

    def hset(self, name, key=None, value=None, mapping=None):
        target = self.hashes.setdefault(name, {})
        if mapping:
            self.hset_calls += 1
            target.update(mapping)
        if key is not None:
            target[key] = value

    def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def hdel(self, name, *keys):
        target = self.hashes.get(name, {})
        return sum(1 for key in keys if target.pop(key, None) is not None)

    def hkeys(self, name):
        return list(self.hashes.get(name, {}).keys())

    def hgetall(self, name):
        self.hgetall_calls += 1
        return dict(self.hashes.get(name, {}))

    def delete(self, name):
        self.hashes.pop(name, None)
        self.strings.pop(name, None)

    def get(self, name):
        return self.strings.get(name)

    def set(self, name, value):
        self.strings[name] = value


@pytest.fixture(scope="module")
def socket_utils(owui_module):
    return owui_module("open_webui.socket.utils")


@pytest.fixture(scope="module")
def models_util(owui_module):
    return owui_module("open_webui.utils.models")


@pytest.fixture(scope="module")
def socket_main_tree(open_webui_backend: Path) -> ast.Module:
    source = (open_webui_backend / "open_webui" / "socket" / "main.py").read_text(encoding="utf-8")
    return ast.parse(source)


def _cached_dict(socket_utils, redis):
    with patch.object(socket_utils, "get_redis_connection", return_value=redis):
        return socket_utils.CachedRedisDict("models", "redis://localhost:6379")


def _signed_dict(socket_utils, redis):
    with patch.object(socket_utils, "get_redis_connection", return_value=redis):
        return socket_utils.RedisDict(
            "models", "redis://localhost:6379", cache_set_signature=True
        )


# --- narrow: #28176, reads come from the per-worker cache ------------------------


def test_a_read_refetches_only_when_the_signature_changed(socket_utils):
    """The whole point of the cache: an unchanged pool costs one GET, not an HGETALL."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}, "b": {"id": "b"}})

    assert models["a"] == {"id": "a"}
    assert models["b"] == {"id": "b"}
    assert "b" in models
    assert len(models) == 2
    assert redis.hgetall_calls == 1, (
        "reading a couple of models pulled the whole shared hash out of Redis again "
        "on every access, so a chat request paid one HGETALL per model lookup (#28176)"
    )


def test_many_lookups_of_one_model_cost_one_hgetall(socket_utils):
    """The access pattern of a real request: resolve the model and read params."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"m": {"id": "m", "info": {"meta": {}}}})

    for _ in range(10):
        models["m"]
    assert redis.hgetall_calls == 1, "repeated lookups of the same pool kept refetching it (#28176)"


def test_a_worker_sees_another_worker_s_change(socket_utils):
    """The cache may only serve what the signature vouches for, never a stale copy."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}})
    assert models["a"] == {"id": "a"}
    reads_before = redis.hgetall_calls

    other = _cached_dict(socket_utils, redis)
    other.set({"a": {"id": "a"}, "c": {"id": "c"}})

    assert "c" in models, (
        "a model another worker added stayed invisible because the reader kept "
        "serving its stale per-worker copy (#28176)"
    )
    assert redis.hgetall_calls == reads_before + 1


def test_a_cleared_signature_forces_a_refetch(socket_utils):
    """A writer clears the signature before rewriting the hash; readers must not trust their
    cached copy once the signature is gone."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}})
    assert models["a"] == {"id": "a"}

    redis.delete("models:signature")
    redis.hset("models", key="a", value='{"id": "a2"}')

    assert models["a"] == {"id": "a2"}, (
        "the reader kept its cached copy while the signature key was absent, serving a "
        "model the writer was replacing (#28176)"
    )


def test_delete_many_is_visible_to_readers(socket_utils):
    """#28176's second hole: delete_many never touched the signature, so readers kept the
    deleted models forever."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}, "b": {"id": "b"}})
    assert models["b"] == {"id": "b"}

    models.delete_many("b")

    assert "b" not in models, "delete_many left the reader serving a model it had removed (#28176)"


def test_items_and_values_read_through_the_cache(socket_utils):
    """Broad: every mapping operation that used to fetch the whole hash serves from the cache."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}, "b": {"id": "b"}})

    assert sorted(models.keys()) == ["a", "b"]
    assert {"a": {"id": "a"}, "b": {"id": "b"}} == dict(models.items())
    assert {"id": "a"} in models.values()
    assert redis.hgetall_calls == 1


def test_a_missing_key_raises_keyerror(socket_utils):
    """Nearby: the cached surface keeps the dict contract."""
    redis = FakeRedis()
    models = _cached_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}})

    with pytest.raises(KeyError):
        models["missing"]


def test_the_shared_pool_is_the_cached_kind_behind_redis(socket_main_tree):
    """The MODELS pool must be a CachedRedisDict in the redis-manager branch, or the cache
    is dead code. Pinned structurally: the branch only exists when WEBSOCKET_MANAGER=redis
    at import time, which this suite's environment does not set."""
    source = ast.unparse(socket_main_tree)
    in_redis_branch = re.search(
        r"if WEBSOCKET_MANAGER == 'redis':(.*)", source, re.DOTALL
    )
    assert in_redis_branch, "the redis websocket-manager branch is gone from socket/main.py"
    branch = in_redis_branch.group(1)
    assignment = re.search(r"MODELS = (\w+)\(", branch)
    assert assignment and assignment.group(1) == "CachedRedisDict", (
        "the shared model pool is not a CachedRedisDict, so every chat request "
        "fetches it from Redis again (#28176)"
    )


# --- narrow: 649c012e, no-op refreshes stop rewriting the registry ----------------


def _ollama_model(name: str, expires_at: int | None) -> dict:
    entry = {"model": name, "name": name}
    if expires_at is not None:
        entry["expires_at"] = expires_at
    return {
        "id": name,
        "name": name,
        "object": "model",
        "created": 0,
        "owned_by": "ollama",
        "ollama": entry,
    }


def _request_with_redis_pool(pool):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(redis=None, MODELS=pool, BASE_MODELS=[])
        )
    )


async def _run_get_all_models(models_util, request, base_models):
    """Drive the real get_all_models with the provider fetch stubbed at the I/O boundary."""

    async def get_all_base_models(request, user=None):
        return [dict(model) for model in base_models]

    config = {
        "models.base_models_cache": False,
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
        return await models_util.get_all_models(request)


@pytest.mark.asyncio
async def test_a_refresh_where_only_the_countdown_moved_does_not_rewrite_the_registry(
    models_util, socket_utils
):
    """Pre-fix the registry fingerprint covered expires_at, so Ollama's moving countdown
    made every refresh rewrite the whole pool to Redis."""
    redis = FakeRedis()
    pool = _signed_dict(socket_utils, redis)
    request = _request_with_redis_pool(pool)

    await _run_get_all_models(models_util, request, [_ollama_model("m1", expires_at=100)])
    writes_after_first = redis.hset_calls

    await _run_get_all_models(models_util, request, [_ollama_model("m1", expires_at=200)])

    assert redis.hset_calls == writes_after_first, (
        "a refresh that changed nothing but Ollama's expires_at countdown rewrote the whole "
        "model registry to Redis (649c012e)"
    )


@pytest.mark.asyncio
async def test_the_registry_copy_carries_no_countdown_but_the_api_response_keeps_it(
    models_util, socket_utils
):
    """Broad: expires_at stays in what the UI reads, but never in what the registry signs."""
    from open_webui.utils.json_codec import JSONCodec

    redis = FakeRedis()
    pool = _signed_dict(socket_utils, redis)
    request = _request_with_redis_pool(pool)

    result = await _run_get_all_models(
        models_util, request, [_ollama_model("m1", expires_at=100)]
    )

    stored = JSONCodec.loads(redis.hashes["models"]["m1"])
    assert "expires_at" not in stored["ollama"], (
        "the moving countdown went into the registry copy, so its signature changed on every "
        "refresh and the write could never be skipped (649c012e)"
    )
    assert result[0]["ollama"]["expires_at"] == 100, (
        "the API response lost the countdown (649c012e)"
    )


@pytest.mark.asyncio
async def test_a_genuinely_changed_model_list_is_still_written(
    models_util, socket_utils
):
    """Nearby: dropping expires_at must not suppress the write when a model is added."""
    redis = FakeRedis()
    pool = _signed_dict(socket_utils, redis)
    request = _request_with_redis_pool(pool)

    await _run_get_all_models(models_util, request, [_ollama_model("m1", expires_at=100)])
    await _run_get_all_models(
        models_util,
        request,
        [_ollama_model("m1", expires_at=100), _ollama_model("m2", expires_at=None)],
    )

    assert redis.hset_calls == 2, "an actually changed model list was not written to Redis"


def test_a_plain_redis_dict_write_still_obeys_the_content_signature(socket_utils):
    """Nearby: the underlying RedisDict skip is keyed on the content digest, not the token."""
    redis = FakeRedis()
    models = _signed_dict(socket_utils, redis)
    models.set({"a": {"id": "a"}})
    writes = redis.hset_calls

    models.set({"a": {"id": "a"}})
    assert redis.hset_calls == writes, "an identical rewrite went through to Redis (aceda892b)"

    redis.strings["models:signature"] = "not-the-content-digest:token"
    models.set({"a": {"id": "a"}})
    assert redis.hset_calls == writes + 1, (
        "a signature from different content suppressed the repairing write, so a diverged "
        "hash could never be repaired (aceda892b)"
    )
