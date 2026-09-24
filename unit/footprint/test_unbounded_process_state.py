"""Guard: in-process registries must shrink again once their entries are dead.

Each test feeds one of the backend's in-memory registries many distinct keys and asserts the
registry lets go or never grew: expired login keys, finished tasks, warned-about values. A
failure here is memory only a restart reclaims. Deleted plugins giving their source back is
measured on the server process in `integration/footprint/test_process_memory_stays_bounded.py`;
these registries grow by a few bytes per key, far below what the process size can show.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3); upstream has since fixed the three cases
(#29971, #29977, #29980), so they assert the fixed behaviour. Unmarked because no issue is filed.
Discriminates: in a copy of dev bbfa876af, keying the limiter's store per login key again
(#29977) fails the first test, filing tasks under a falsy item id again (#29980) fails the
second, and a module-level set of rejected URLs (#29971) fails the fourth.
"""

from __future__ import annotations

import asyncio
import time

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


KEYS_PER_WINDOW = 100
EMAIL_DOMAIN = "@example.com"


async def _noop(): ...


def _retained_emails(obj) -> int:
    """Login keys reachable from the limiter's own and its class's attributes."""
    pending, count = [vars(obj), dict(vars(type(obj)))], 0
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            count += value.endswith(EMAIL_DOMAIN)
        elif isinstance(value, dict):
            pending.extend([*value.keys(), *value.values()])
        elif isinstance(value, (list, set, tuple)):
            pending.extend(value)
    return count


def _module_container_sizes(module):
    return {
        name: len(value)
        for name, value in vars(module).items()
        if isinstance(value, (set, dict, list))
    }


@pytest.mark.asyncio
async def test_rate_limiter_forgets_keys_once_their_window_has_passed(
    rate_limit_module, monkeypatch
):
    """Without Redis the limiter keeps its counts in memory, fed the login email the sign-in
    request supplies. Rolling past a window has to drop every key that window held."""
    limiter = rate_limit_module.RateLimiter(limit=5, window=60)
    now = time.time()

    for window in range(5):
        monkeypatch.setattr(time, "time", lambda moment=now + window * 600: moment)
        for i in range(KEYS_PER_WINDOW):
            await limiter.is_limited(redis=None, key=f"user{window}-{i}{EMAIL_DOMAIN}")

    retained = _retained_emails(limiter)
    assert 0 < retained <= KEYS_PER_WINDOW, (
        f"{retained} login keys retained; keys from expired windows must be dropped"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("item_id", ["", None])
async def test_a_finished_task_without_an_item_id_is_not_tracked(tasks_module, item_id):
    """`ydoc_document_join` accepts an empty document id and `yjs_document_update` hands it to
    `create_task` as the item id; cleanup only ever removes tasks filed under a truthy id."""
    task_id, task = await tasks_module.create_task(redis=None, coroutine=_noop(), id=item_id)
    await task

    assert task_id not in await tasks_module.list_task_ids_by_item_id(redis=None, id=item_id)


@pytest.mark.asyncio
async def test_a_finished_task_with_an_item_is_dropped_from_its_item(tasks_module):
    """Control: the keyed path is cleaned up."""
    task_id, task = await tasks_module.create_task(redis=None, coroutine=_noop(), id="chat-1")
    assert task_id in await tasks_module.list_task_ids_by_item_id(redis=None, id="chat-1")
    await task
    for _ in range(3):  # the done callback schedules the cleanup as a task of its own
        await asyncio.sleep(0)

    assert task_id not in await tasks_module.list_tasks(redis=None)
    assert task_id not in await tasks_module.list_task_ids_by_item_id(redis=None, id="chat-1")


def test_invalid_profile_image_url_warnings_do_not_accumulate_per_value(models_module):
    """`ModelForm.meta` is validated before the workspace permission check, so any signed-in user
    can feed the validator distinct invalid URLs. Clearing one must keep no per-value state."""
    sizes_before = _module_container_sizes(models_module)

    for i in range(20):
        meta = models_module.ModelMeta(profile_image_url=f"data:image/svg+xml;base64,{i}")
        if meta.profile_image_url is not None:
            pytest.fail("the validator no longer clears an SVG data URI")

    grown = [
        name
        for name, size in _module_container_sizes(models_module).items()
        if size > sizes_before.get(name, 0)
    ]
    assert not grown, f"module state grew per rejected URL value: {grown}"
