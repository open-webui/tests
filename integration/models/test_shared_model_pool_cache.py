"""Regression: the shared model pool behind Redis was refetched and rewritten on every touch.

Two 0.11.4 fixes to the model registry an instance keeps in Redis with `WEBSOCKET_MANAGER=redis`:

* `aceda892b` (#28176, issue #28167): every read of the pool went to Redis, so resolving the
  models of one chat request cost a round trip per lookup. The pool is now a `CachedRedisDict`
  that keeps a per-worker copy and refetches it with one HGETALL only when the pool's signature
  key changes or is gone (a writer clears it while it rewrites the pool), and a signature left
  by other content never suppresses a write.
* `649c012e`: Ollama reports a moving `expires_at` for a model it holds in memory, and the
  registry's fingerprint covered it, so every model refresh rewrote the whole pool. The
  registry copy now drops it; the API response keeps it.

The instance runs on `HashRedis`, which records every command it is sent and lets the test
change the pool as another worker would. `delete_many`'s signature handling has no caller on
the pool and is not pinned.

Twin of unit/models/test_shared_model_pool_cache.py.

Discriminates: passes on dev bbfa876af; with the pool back on a plain `RedisDict` a chat request
reads the pool six times (five HGET, one HEXISTS) and every refresh rewrites it; with the cache
refetching only when empty, or trusting a cleared signature, another worker's change is not
seen; with any signature suppressing a write the pool is never repaired; and with `expires_at`
kept in the registry copy a refresh in which only the countdown moved rewrites the pool.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json

import pytest

from harness.actors import create_user
from harness.chat import ask
from harness.listener import json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.upstream import MOCK_MODEL_ID
from integration.hash_redis import HashRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MODEL_POOL = "open-webui:models"
SIGNATURE = f"{MODEL_POOL}:signature"
POOL_READS = ("HGET", "HGETALL", "HEXISTS", "HKEYS", "HVALS", "HLEN", "HSCAN")


@pytest.fixture(scope="module")
def pool_redis():
    redis = HashRedis()
    yield redis
    redis.close()


@pytest.fixture(scope="module")
def pooled_instance(instance_with, pool_redis):
    return instance_with(
        {
            "WEBSOCKET_MANAGER": "redis",
            "WEBSOCKET_REDIS_URL": pool_redis.url,
            # the room-channel manager pattern-subscribes, which the stand-in does not serve
            "WEBSOCKET_REDIS_ROOM_CHANNELS": "false",
        }
    )


def _refreshed_models(client) -> list[dict]:
    listed = client.get("/api/models", params={"refresh": "true"})
    assert listed.status_code == 200, listed.text
    return listed.json()["data"]


def test_a_chat_request_reads_the_pool_from_redis_at_most_once(pooled_instance, pool_redis):
    user = create_user(pooled_instance)
    with user.client() as client:
        ask(client, "warm up")
        pool_redis.clear_log()

        _, answer = ask(client, "hello")

    assert answer["done"], answer
    reads = {command: pool_redis.count(command, MODEL_POOL) for command in POOL_READS}
    assert sum(reads.values()) <= 1, (
        f"one chat request read the model pool from Redis {sum(reads.values())} times: {reads}"
    )


def test_a_model_added_to_the_pool_answers_at_once(pooled_instance):
    with pooled_instance.client() as client:
        _refreshed_models(client)
        pooled_instance.upstream.models.append("late-model")
        assert "late-model" in {model["id"] for model in _refreshed_models(client)}

        _, answer = ask(client, "hello", model="late-model")

    assert answer["done"] and not answer.get("error"), (
        f"the worker kept serving its cached copy of the pool without the new model: {answer}"
    )


@pytest.mark.parametrize("signature", ["replaced", "cleared"])
def test_a_change_another_worker_made_to_the_pool_is_seen(pooled_instance, pool_redis, signature):
    upstream = pooled_instance.upstream
    with pooled_instance.client() as client:
        _refreshed_models(client)
        ask(client, "warm up")  # this worker now holds a copy of the pool
        entry = json.loads(pool_redis.fields(MODEL_POOL)[MOCK_MODEL_ID])
        entry["info"]["meta"]["capabilities"] = {"usage": True}
        pool_redis.write_fields(MODEL_POOL, {MOCK_MODEL_ID: json.dumps(entry)})
        # a writer clears the signature before it rewrites the pool, and sets a new one after
        pool_redis.write_value(
            SIGNATURE, "another-worker:token" if signature == "replaced" else None
        )
        upstream.reset()

        ask(client, "hello")

        assert upstream.chat_requests()[-1].get("stream_options") == {"include_usage": True}, (
            "the worker served its cached copy of the pool over the change another worker made"
        )
        pool_redis.clear_log()
        _refreshed_models(client)
    repaired = json.loads(pool_redis.fields(MODEL_POOL)[MOCK_MODEL_ID])
    assert pool_redis.count("HSET", MODEL_POOL) == 1, "a foreign signature stopped the repair"
    assert repaired["info"]["meta"]["capabilities"] != {"usage": True}


@pytest.fixture
def ollama_countdown(pooled_instance, preserve, listener):
    """An Ollama model whose `expires_at` moves on every refresh, as a loaded model's does."""
    preserve(OLLAMA_CONFIG, on=pooled_instance)
    serve_ollama(listener, "llama3:latest")
    ticks = itertools.count()

    def loaded(_request):
        expires_at = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=next(ticks)
        )
        return json_answer(
            {
                "models": [
                    {
                        "name": "llama3:latest",
                        "model": "llama3:latest",
                        "expires_at": expires_at.isoformat(),
                    }
                ]
            }
        )

    listener.route("GET", "/api/ps", loaded)
    with pooled_instance.client() as client:
        connect_ollama(client, listener)
        yield client


def _ollama_expiry(models: list[dict]) -> int:
    return next(model for model in models if model["id"] == "llama3:latest")["ollama"]["expires_at"]


def test_a_refresh_where_only_the_countdown_moved_does_not_rewrite_the_pool(
    ollama_countdown, pool_redis
):
    first_expiry = _ollama_expiry(_refreshed_models(ollama_countdown))
    pool_redis.clear_log()

    second_expiry = _ollama_expiry(_refreshed_models(ollama_countdown))

    assert second_expiry != first_expiry, "the stand-in's countdown did not move"
    stored = json.loads(pool_redis.fields(MODEL_POOL)["llama3:latest"])
    assert "expires_at" not in stored["ollama"]
    assert pool_redis.count("HSET", MODEL_POOL) == 0, (
        "a refresh that changed nothing but Ollama's expires_at rewrote the whole model pool"
    )


def test_a_changed_model_list_is_still_written(ollama_countdown, pool_redis, pooled_instance):
    _refreshed_models(ollama_countdown)
    pool_redis.clear_log()
    pooled_instance.upstream.models.append("another-model")

    listed = {model["id"] for model in _refreshed_models(ollama_countdown)}

    assert "another-model" in listed
    assert pool_redis.count("HSET", MODEL_POOL) == 1
