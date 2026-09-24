"""Regression: a task that finishes while a chat is being stopped must not end the stop early.

open-webui 0.11.4 fix `e35b907f7` (#29844, issue #29816): without Redis, `stop_item_tasks` in
`open_webui/tasks.py` returned on the first `stop_task` that reported failure. A task that
finished on its own while an earlier one was still being cancelled has already cleaned itself
out of `tasks`, so it reports "not found" and every task listed after it kept running. The fix
stops every listed task in turn.

The skipped-task half of the same issue (the live task list) is pinned over HTTP by
integration/security/test_chat_stop_item_tasks.py. This half needs a task to finish inside
another task's cancellation, which only the event loop can arrange.

Discriminates: passes on dev bbfa876af; fails with e35b907f7 reverted (the idle third task is
still running after the stop).
"""

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def tasks_module(owui_module):
    return owui_module("open_webui.tasks")


async def _slow_to_cancel():
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        await asyncio.sleep(0.2)
        raise


@pytest.mark.asyncio
async def test_a_task_finishing_during_the_stop_does_not_end_it_early(tasks_module):
    chat_id = f"chat-{uuid.uuid4()}"
    started = [
        await tasks_module.create_task(redis=None, coroutine=coroutine, id=chat_id)
        for coroutine in (_slow_to_cancel(), asyncio.sleep(0.05), asyncio.sleep(60))
    ]
    idle_task = started[2][1]
    await asyncio.sleep(0)

    try:
        result = await tasks_module.stop_item_tasks(redis=None, item_id=chat_id)
        assert result["status"] is True
        assert idle_task.cancelled(), (
            "a task that finished while the chat was being stopped ended the stop early, so "
            "the tasks listed after it kept running (#29816)"
        )
    finally:
        idle_task.cancel()
