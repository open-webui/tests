"""Regression: stopping a chat must stop every in-flight task, not just the first.

open-webui 0.11.4 fix `e35b907f7` (#29844, issue #29816): `stop_item_tasks` in
`open_webui/tasks.py` returned early on the first `stop_task` result that reported
failure (which a task that had already finished produces), so with more than one
task in flight the rest kept running to their own limit. It also iterated the
live `item_tasks[id]` list, which `stop_task`'s cleanup mutates while the loop is
awaiting, so tasks further down the list were skipped entirely. The fix iterates
a snapshot and stops every task in turn.

Discriminates: passes on dev 344ea5306; on the pre-fix ref the second task is
never stopped when the first reports not-found, and a mid-loop removal skips the
task after it.
"""

import asyncio

import pytest

pytest.importorskip('fastapi')

pytestmark = pytest.mark.regression


@pytest.fixture(scope='module')
def tasks_module(owui_module):
    return owui_module('open_webui.tasks')


def _seed(tasks_module, item_id, task_ids):
    """Register real asyncio tasks under item_id, as register_task would."""
    tasks_module.item_tasks.setdefault(item_id, [])

    async def idle():
        await asyncio.sleep(60)

    for task_id in task_ids:
        loop_task = asyncio.get_event_loop().create_task(idle())
        tasks_module.item_tasks[item_id].append(task_id)
        tasks_module.tasks[task_id] = loop_task
    return [tasks_module.tasks[t] for t in task_ids]


@pytest.mark.asyncio
async def test_every_task_is_stopped_when_one_reports_not_found(tasks_module):
    """The pre-fix early return: a finished first task ends the round."""
    seeded = _seed(tasks_module, 'chat-batch', ['t1', 't2', 't3'])
    seeded[0].cancel()  # t1 already finished by the time the stop round runs
    await asyncio.gather(seeded[0], return_exceptions=True)

    result = await tasks_module.stop_item_tasks(None, 'chat-batch')

    assert result['status'] is True
    assert all(t.cancelled() or t.done() for t in seeded), (
        'stopping a chat left tasks running because a task that had already '
        'finished ended the round early (#29816)'
    )


@pytest.mark.asyncio
async def test_the_task_list_is_iterated_over_a_snapshot(tasks_module):
    """stop_task's cleanup mutates item_tasks[id]; the loop must not follow it."""
    seeded = _seed(tasks_module, 'chat-mutate', ['t1', 't2', 't3'])

    result = await tasks_module.stop_item_tasks(None, 'chat-mutate')

    assert result['status'] is True
    assert all(t.cancelled() or t.done() for t in seeded), (
        'stopping a chat skipped tasks because the list being worked through was '
        'rewritten underneath the loop as each one was cleared (#29816)'
    )


@pytest.mark.asyncio
async def test_an_empty_item_reports_no_tasks(tasks_module):
    result = await tasks_module.stop_item_tasks(None, 'chat-empty')

    assert result['status'] is True
    assert 'No tasks' in result['message']
