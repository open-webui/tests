"""Guard: in-process registries must shrink again once their entries are dead.

Each test feeds one of the backend's in-memory registries many distinct keys, or removes
the thing an entry stands for, and asserts the registry lets go or never grew: expired login
keys, finished tasks, warned-about values, deleted plugins. A failure here is memory only a
restart reclaims.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3), where every case except the control
fails. Each is a strict `xfail` until upstream fixes it, and fails as XPASS when it does.
Unmarked because no issue is filed yet.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope="session")
def rate_limit_module(owui_module):
    return owui_module("open_webui.utils.rate_limit")


@pytest.fixture(scope="session")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


@pytest.fixture(scope="session")
def models_module(owui_module):
    return owui_module("open_webui.models.models")


@pytest.fixture(scope="session")
def tools_router_module(owui_module):
    return owui_module("open_webui.routers.tools")


@pytest.fixture(scope="session")
def functions_router_module(owui_module):
    return owui_module("open_webui.routers.functions")


OWNER = SimpleNamespace(id="u1")
ADMIN = SimpleNamespace(id="a1")
KEYS_PER_WINDOW = 100


async def _noop(): ...


def _request_with_caches(**caches):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**caches)))


@pytest.mark.xfail(
    raises=AssertionError, strict=True, reason="expired outer entries are never dropped"
)
def test_rate_limiter_forgets_keys_once_their_window_has_passed(rate_limit_module, monkeypatch):
    """`RateLimiter._memory_store` is the fallback without Redis, keyed by the login email the
    signin request supplies. A key's expired buckets are pruned only when that key is checked
    again and its outer entry is never removed, so an abandoned email keeps both forever."""
    limiter_class = rate_limit_module.RateLimiter
    monkeypatch.setattr(limiter_class, "_memory_store", type(limiter_class._memory_store)())
    limiter = limiter_class(redis_client=None, limit=5, window=60)
    bucket = 1_000
    monkeypatch.setattr(limiter, "_current_bucket", lambda: bucket)

    for window in range(5):
        for i in range(KEYS_PER_WINDOW):
            limiter.is_limited(f"user{window}-{i}@example.com")
        bucket += 10

    assert len(limiter._memory_store) <= KEYS_PER_WINDOW, (
        "keys from expired windows are still in the store"
    )


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="cleanup_task skips a falsy item id")
@pytest.mark.asyncio
async def test_finished_task_with_an_empty_item_id_is_dropped_from_item_tasks(tasks_module):
    """`create_task` files every task under `item_tasks[id]`, but `cleanup_task` only removes
    it when `id` is truthy. In `socket/main.py`, `ydoc_document_join` accepts an empty document
    id (no `note:` prefix, so no access check) and `yjs_document_update` hands it to
    `create_task` as the item id."""

    task_id, task = await tasks_module.create_task(None, _noop(), id="")
    await task
    await asyncio.sleep(0)
    leftover = tasks_module.item_tasks.pop("", [])

    assert task_id not in leftover


@pytest.mark.asyncio
async def test_finished_task_with_an_item_is_dropped_from_item_tasks(tasks_module):
    """Control: the keyed path is cleaned up today."""

    task_id, task = await tasks_module.create_task(None, _noop(), id="chat-1")
    await task
    await asyncio.sleep(0)

    assert task_id not in tasks_module.tasks
    assert "chat-1" not in tasks_module.item_tasks


@pytest.mark.xfail(
    raises=AssertionError, strict=True, reason="one set entry per distinct invalid value"
)
def test_invalid_profile_image_url_warnings_do_not_accumulate_per_value(models_module, monkeypatch):
    """`_warned_profile_urls` deduplicates a warning per distinct URL value. `ModelForm.meta` is
    validated before the workspace permission check, so any signed-in user grows the set by one
    entry per request; the fix is to stop warning per value, a digest per value still grows."""
    warned = set()
    monkeypatch.setattr(models_module, "_warned_profile_urls", warned)

    for i in range(20):
        meta = models_module.ModelMeta(profile_image_url=f"data:image/svg+xml;base64,{i}")
        if meta.profile_image_url is not None:
            pytest.fail("the validator no longer clears an SVG data URI")

    assert len(warned) <= 1, f"{len(warned)} invalid URL values retained for a log deduplication"


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="TOOL_CONTENTS is never popped")
@pytest.mark.asyncio
async def test_deleting_a_tool_evicts_its_cached_source(tools_router_module, monkeypatch):
    """Delete pops the module from `TOOLS` and must pop its source from `TOOL_CONTENTS` too."""
    tool = SimpleNamespace(id="t1", user_id=OWNER.id, name="t1")
    monkeypatch.setattr(
        tools_router_module,
        "Tools",
        SimpleNamespace(
            get_tool_by_id=AsyncMock(return_value=tool),
            delete_tool_by_id=AsyncMock(return_value=True),
        ),
    )
    monkeypatch.setattr(tools_router_module, "publish_event", AsyncMock())
    request = _request_with_caches(TOOLS={"t1": object()}, TOOL_CONTENTS={"t1": "source"})

    if not await tools_router_module.delete_tools_by_id(request, "t1", user=OWNER, db=None):
        pytest.fail("delete reported failure")
    if "t1" in request.app.state.TOOLS:
        pytest.fail("delete did not pop the module cache")

    assert "t1" not in request.app.state.TOOL_CONTENTS, "source text kept after delete"


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="FUNCTION_CONTENTS is never popped")
@pytest.mark.asyncio
async def test_deleting_a_function_evicts_its_cached_source(functions_router_module, monkeypatch):
    """Same contract for `FUNCTIONS` and `FUNCTION_CONTENTS`."""
    monkeypatch.setattr(
        functions_router_module,
        "Functions",
        SimpleNamespace(delete_function_by_id=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr(functions_router_module, "publish_event", AsyncMock())
    request = _request_with_caches(FUNCTIONS={"f1": object()}, FUNCTION_CONTENTS={"f1": "source"})

    if not await functions_router_module.delete_function_by_id(request, "f1", user=ADMIN, db=None):
        pytest.fail("delete reported failure")
    if "f1" in request.app.state.FUNCTIONS:
        pytest.fail("delete did not pop the module cache")

    assert "f1" not in request.app.state.FUNCTION_CONTENTS, "source text kept after delete"
