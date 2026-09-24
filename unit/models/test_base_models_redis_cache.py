"""The base model list moved into Redis, open-webui v0.11.2 (`6609918bf`).

`utils.models.get_all_models` cached the assembled base model list only in `app.state`, so every
worker rebuilt it on its own and a connection change on one worker left the others serving the
old list. The list is now shared through Redis: a worker reads what another built, writes back
what it builds, and a refresh or a connection change on either provider deletes it.

Kept as a unit test: the HTTP form needs two instances on one database and one Redis, and the
suite has neither a real Redis nor a way to boot two instances on one database. Here two
workers are two FastAPI apps with the real provider routers, over one scratch SQLite with the
real tables and one Redis stand-in specced from `redis.asyncio.Redis`. Providers answer from
their configured model ids, so nothing leaves the process; every call comes from a fresh
account, so the providers' per-account one-second cache never answers for another worker.

Discriminates: passes on upstream dev `bbfa876af`; with 6609918bf reverted (the Redis read,
write and deletes removed from `utils/models.py` and both provider routers) the first three
tests (four cases) fail, each second worker serving its own stale list, and the rest pass.
"""

from __future__ import annotations

import pkgutil
import uuid
from unittest.mock import create_autospec

import httpx
import pytest
import pytest_asyncio
import redis.asyncio
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import Request

pytestmark = pytest.mark.regression

PROVIDER_URL = "http://provider.invalid/v1"


def provider_settings(model_ids: list[str], *, enabled: bool = True) -> dict:
    """An OpenAI connection that lists `model_ids` without being asked."""
    return {
        "openai.enable": enabled,
        "openai.api_base_urls": [PROVIDER_URL],
        "openai.api_keys": [""],
        "openai.api_configs": {"0": {"model_ids": model_ids}},
    }


@pytest.fixture(scope="module")
def backend(owui_module):
    """The modules a worker is made of, with every table module imported."""
    table_modules = owui_module("open_webui.models")
    for table_module in pkgutil.iter_modules(table_modules.__path__):
        owui_module(f"open_webui.models.{table_module.name}")
    return {
        "models": owui_module("open_webui.utils.models"),
        "openai": owui_module("open_webui.routers.openai"),
        "ollama": owui_module("open_webui.routers.ollama"),
        "config": owui_module("open_webui.models.config").Config,
        "users": owui_module("open_webui.models.users"),
        "auth": owui_module("open_webui.utils.auth"),
        "db": owui_module("open_webui.internal.db"),
    }


@pytest_asyncio.fixture
async def configure(backend, tmp_path, monkeypatch):
    """A scratch database with every real table; returns a writer for its settings."""
    db_path = tmp_path / "webui.db"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    backend["db"].Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    sessions = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(backend["db"], "AsyncSessionLocal", sessions)

    async def write(settings: dict) -> None:
        await backend["config"].upsert(settings)

    await write(
        {"ollama.enable": False, "evaluation.arena.enable": False, "models.base_models_cache": True}
    )
    yield write
    await async_engine.dispose()


@pytest.fixture
def shared_redis():
    """A Redis stand-in holding strings in a dict, specced so a changed call shape fails."""
    values: dict[str, str] = {}
    client = create_autospec(redis.asyncio.Redis, instance=True)

    async def get(name):
        return values.get(name)

    async def set_value(name, value, *args, **kwargs):
        values[name] = value

    async def delete(*names):
        return sum(values.pop(name, None) is not None for name in names)

    client.get.side_effect = get
    client.set.side_effect = set_value
    client.delete.side_effect = delete
    client.values = values
    return client


class Worker:
    """One Open WebUI worker: its own app state, sharing the database and Redis."""

    def __init__(self, backend, redis_client):
        self.backend = backend
        self.app = FastAPI()
        self.app.state.redis = redis_client
        self.app.state.MODELS = {}
        self.app.state.BASE_MODELS = []
        self.app.state.OPENAI_MODELS = {}
        self.app.state.OLLAMA_MODELS = {}
        self.app.include_router(backend["openai"].router, prefix="/openai")
        self.app.include_router(backend["ollama"].router, prefix="/ollama")
        self.app.dependency_overrides[backend["auth"].get_admin_user] = self.account

    def account(self):
        suffix = uuid.uuid4().hex[:8]
        return self.backend["users"].UserModel(
            id=f"user-{suffix}",
            email=f"{suffix}@example.com",
            name=suffix,
            role="admin",
            last_active_at=0,
            updated_at=0,
            created_at=0,
        )

    async def model_ids(self, *, refresh: bool = False) -> list[str]:
        """What this worker's `/api/models` is built from."""
        request = Request({"type": "http", "app": self.app, "headers": [], "method": "GET"})
        models = await self.backend["models"].get_all_models(
            request, refresh=refresh, user=self.account()
        )
        return [model["id"] for model in models]

    async def post(self, path: str, payload: dict) -> None:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            response = await client.post(path, json=payload)
        assert response.status_code == 200, response.text


@pytest.fixture
def worker(backend):
    return lambda redis_client: Worker(backend, redis_client)


@pytest.mark.asyncio
async def test_a_cold_worker_serves_the_list_another_worker_built(configure, worker, shared_redis):
    await configure(provider_settings(["first-model"]))
    assert await worker(shared_redis).model_ids() == ["first-model"]

    await configure(provider_settings(["second-model"]))

    assert await worker(shared_redis).model_ids() == ["first-model"], (
        "a second worker rebuilt the list instead of taking the one in Redis"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/openai/config/update",
            {
                "ENABLE_OPENAI_API": True,
                "OPENAI_API_BASE_URLS": [PROVIDER_URL],
                "OPENAI_API_KEYS": [""],
                "OPENAI_API_CONFIGS": {"0": {"model_ids": ["second-model"]}},
            },
        ),
        (
            "/ollama/config/update",
            {"ENABLE_OLLAMA_API": False, "OLLAMA_BASE_URLS": [], "OLLAMA_API_CONFIGS": {}},
        ),
    ],
    ids=["openai", "ollama"],
)
async def test_a_connection_change_on_one_worker_reaches_the_other(
    configure, worker, shared_redis, path, payload
):
    await configure(provider_settings(["first-model"]))
    changing, serving = worker(shared_redis), worker(shared_redis)
    assert await changing.model_ids() == await serving.model_ids() == ["first-model"]

    await configure(provider_settings(["second-model"]))
    await changing.post(path, payload)

    assert await serving.model_ids() == ["second-model"], (
        "a connection change on one worker left another serving the old list"
    )


@pytest.mark.asyncio
async def test_a_refresh_on_one_worker_reaches_the_other(configure, worker, shared_redis):
    await configure(provider_settings(["first-model"]))
    refreshing, serving = worker(shared_redis), worker(shared_redis)
    assert await refreshing.model_ids() == await serving.model_ids() == ["first-model"]

    await configure(provider_settings(["second-model"]))

    assert await refreshing.model_ids(refresh=True) == ["second-model"]
    assert await serving.model_ids() == ["second-model"]


@pytest.mark.asyncio
async def test_without_redis_a_worker_keeps_its_own_list(configure, worker):
    alone = worker(None)
    await configure(provider_settings(["first-model"]))
    assert await alone.model_ids() == ["first-model"]

    await configure(provider_settings(["second-model"]))

    assert await alone.model_ids() == ["first-model"]


@pytest.mark.asyncio
async def test_with_the_cache_off_every_call_rebuilds_and_nothing_is_shared(
    configure, worker, shared_redis
):
    await configure({"models.base_models_cache": False, **provider_settings(["first-model"])})
    uncached = worker(shared_redis)
    assert await uncached.model_ids() == ["first-model"]

    await configure(provider_settings(["second-model"]))

    assert await uncached.model_ids() == ["second-model"]
    assert shared_redis.values == {}


@pytest.mark.asyncio
async def test_providers_answering_nothing_keep_the_last_list(configure, worker, shared_redis):
    await configure({"models.base_models_cache": False, **provider_settings(["first-model"])})
    uncached = worker(shared_redis)
    assert await uncached.model_ids() == ["first-model"]

    await configure(provider_settings(["first-model"], enabled=False))

    assert await uncached.model_ids() == ["first-model"]
