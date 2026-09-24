"""Regression: an empty tool server cache is a hit, and an absent one is not an error.

open-webui 0.11.1 `f1a64ccfc2` (issue 28568): `get_tool_servers` and `get_terminal_servers`
decoded the Redis cache unconditionally and rebuilt whenever the result was falsy. An
unpopulated key hit `loads(None)` and logged a decode error for a cache that was merely empty,
and an empty but valid cached list was taken for a miss, so every request fetched every
server's spec again. Only a Redis-backed instance reaches this, so it stays here, driven through
the real loaders with the Redis client and the config store as the only stand-ins; a rebuild
shows as the cache being written again.

The docstring, per-connection cookie and unreachable-server fixes this file covered are pinned
over HTTP by integration/tools/test_tool_server_plumbing.py.

Discriminates: passes on dev `bbfa876af`; with `f1a64ccfc2` reverted the empty-list and
absent-key tests fail for both loaders (the empty list is rebuilt, the absent key logs an error).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, create_autospec, patch

import pytest

pytest.importorskip("fastapi")
redis_asyncio = pytest.importorskip("redis.asyncio")

from fastapi import FastAPI  # noqa: E402
from starlette.requests import Request  # noqa: E402

pytestmark = pytest.mark.regression

LOADERS = ["get_tool_servers", "get_terminal_servers"]


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@contextmanager
def _error_records(module):
    """Records straight off the module logger; caplog cannot see it past loguru's intercept."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                records.append(record)

    handler = Collector()
    previous_level = module.log.level
    module.log.addHandler(handler)
    module.log.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        module.log.removeHandler(handler)
        module.log.setLevel(previous_level)


async def _load(tools_module, loader: str, cached: str | None):
    """Run a loader against a Redis cache holding `cached`, with no connections configured."""
    cache = create_autospec(redis_asyncio.Redis, instance=True)
    # redis-py declares its async commands as plain functions, so autospec cannot see them
    cache.get, cache.set = AsyncMock(return_value=cached), AsyncMock()
    app = FastAPI()
    app.state.redis = cache
    app.state.TOOL_SERVERS = []
    app.state.TERMINAL_SERVERS = []
    request = Request({"type": "http", "app": app, "method": "POST", "path": "/", "headers": []})
    with patch.object(tools_module.Config, "get", AsyncMock(return_value=[])):
        with _error_records(tools_module) as errors:
            result = await getattr(tools_module, loader)(request=request)
    return result, cache.set.await_count, errors


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", LOADERS)
async def test_an_empty_cached_list_is_not_rebuilt(tools_module, loader):
    result, rebuilds, _ = await _load(tools_module, loader, "[]")

    assert result == []
    assert rebuilds == 0, (
        f"{loader} took an empty cached list for a miss and fetched every server's spec again "
        "on every request (#28568)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", LOADERS)
async def test_an_absent_cache_key_rebuilds_without_an_error(tools_module, loader):
    result, rebuilds, errors = await _load(tools_module, loader, None)

    assert result == []
    assert rebuilds == 1
    assert errors == [], f"{loader} logged a decode error for a cache that was only empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", LOADERS)
async def test_a_populated_cache_is_returned_as_is(tools_module, loader):
    cached = '[{"id": "srv", "url": "http://srv.invalid"}]'
    result, rebuilds, _ = await _load(tools_module, loader, cached)

    assert result == [{"id": "srv", "url": "http://srv.invalid"}]
    assert rebuilds == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", LOADERS)
async def test_a_corrupt_cache_is_logged_and_rebuilt(tools_module, loader):
    result, rebuilds, errors = await _load(tools_module, loader, "{not json")

    assert result == []
    assert rebuilds == 1
    assert errors, "a broken cache value is worth an error line"
