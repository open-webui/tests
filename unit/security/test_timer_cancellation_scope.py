"""Regression: marking a chat as read cancelled every user's timers on that chat.

open-webui 0.11.0 fix `e140d8f3c` (#27472): `cancel_timers_for_chat` selected pending timers on
the parent chat without filtering on their owner, so two people holding timers on one chat
cancelled each other's, and the socket read handler let a stranger trigger it. The fix makes
`user_id` a required parameter, filters on `Chat.user_id` and returns early in the socket
handler for a reader who does not own the chat.

The stranger's socket read is pinned end to end by
integration/security/test_timer_cancellation_scope.py. Two owners with timers on one chat cannot
be set up through the API, so the query stays here, run for real against the scratch SQLite
database, next to an `ast` audit that every caller passes the acting user.

Discriminates: passes on dev `bbfa876af`; with `e140d8f3c` reverted the two-owner and both-event
tests fail (the unscoped query cancels the other user's timer).
"""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.regression

CANCEL_TARGET = "cancel_timers_for_chat"


@pytest.fixture(scope="module")
def timers(owui_module):
    owui_module("open_webui.config")  # runs the migrations, so the `chat` table exists
    try:
        return owui_module("open_webui.utils.timers")
    except pytest.skip.Exception:
        pytest.fail("open_webui.utils.timers is gone: retarget these tests at the timer module")


@pytest.fixture
def ids():
    """Unique id prefix so tests sharing the scratch database cannot collide."""
    return uuid4().hex[:12]


def _timer(timers, timer_id, owner_id, parent_chat_id, cancel_on=("chat.read",), status="pending"):
    now = int(time.time())
    due_at = time.time_ns()
    row = timers.Chat(
        id=timer_id,
        user_id=owner_id,
        title=f"Timer: {timer_id}",
        chat={},
        meta={
            "internal": True,
            "type": "timer",
            "parent_chat_id": parent_chat_id,
            "status": status,
            "cancel_on": list(cancel_on),
            "timer_at": due_at,
        },
        created_at=now,
        updated_at=now,
    )
    # 0.11.1 moved the due time onto a column the queries key off; claiming clears it
    if "timer_at" in timers.Chat.__table__.columns:
        row.timer_at = due_at if status == "pending" else None
    return row


async def _seed(timers, rows):
    async with timers.get_async_db() as db:
        for row in rows:
            db.add(row)
        await db.commit()


async def _statuses(timers, timer_ids):
    async with timers.get_async_db() as db:
        return {
            timer_id: (await db.get(timers.Chat, timer_id)).meta.get("status")
            for timer_id in timer_ids
        }


async def _cancelled(timers, timer_ids):
    statuses = await _statuses(timers, timer_ids)
    return {timer_id for timer_id, status in statuses.items() if status == "cancelled"}


async def _cancel_as(timers, parent_chat_id, event, user_id):
    """The pre-fix signature has no `user_id`; calling it without keeps the failure behavioural."""
    arguments = {"parent_chat_id": parent_chat_id, "event": event}
    if "user_id" in inspect.signature(timers.cancel_timers_for_chat).parameters:
        arguments["user_id"] = user_id
    await timers.cancel_timers_for_chat(**arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["chat.read", "chat.user_message"])
async def test_a_cancel_only_touches_the_callers_timers(timers, ids, event):
    """Two users hold a pending timer on one shared chat; only the caller's may go."""
    parent_chat_id = f"{ids}-shared-chat"
    caller_timer, other_timer = f"{ids}-caller", f"{ids}-other"
    await _seed(
        timers,
        [
            _timer(timers, caller_timer, "caller", parent_chat_id, cancel_on=(event,)),
            _timer(timers, other_timer, "other", parent_chat_id, cancel_on=(event,)),
        ],
    )

    await _cancel_as(timers, parent_chat_id, event, "caller")

    assert await _cancelled(timers, [caller_timer, other_timer]) == {caller_timer}, (
        f"the {event} trigger cancelled another user's pending timer on the same chat: their "
        "scheduled prompt silently never fires (#27472)"
    )


def test_every_caller_passes_the_acting_user(timers):
    """Guards the next unscoped sweep, wherever in the backend it is called from."""
    backend = Path(timers.__file__).resolve().parent.parent
    call_sites, unscoped = [], []
    for path in sorted(backend.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if CANCEL_TARGET not in (
                getattr(node.func, "id", None),
                getattr(node.func, "attr", None),
            ):
                continue
            site = f"{path.relative_to(backend)}:{node.lineno}"
            call_sites.append(site)
            if len(node.args) < 3 and not any(kw.arg == "user_id" for kw in node.keywords):
                unscoped.append(site)

    assert call_sites, f"nothing calls {CANCEL_TARGET} any more: retarget this audit"
    assert unscoped == [], (
        f"{CANCEL_TARGET} is called without the acting user at {unscoped}: that call cancels "
        "the pending timers of every user on the chat (#27472)"
    )


# --- Nearby: the scoped query still does its job ---


@pytest.mark.asyncio
async def test_the_callers_own_timer_is_cancelled(timers, ids):
    parent_chat_id = f"{ids}-own-chat"
    own_timer = f"{ids}-own"
    await _seed(timers, [_timer(timers, own_timer, "alice", parent_chat_id)])

    await _cancel_as(timers, parent_chat_id, "chat.read", "alice")

    async with timers.get_async_db() as db:
        meta = (await db.get(timers.Chat, own_timer)).meta
    assert meta["status"] == "cancelled", "scoping the cancel to the caller broke the feature"
    assert meta["timer_cancelled_by"] == "chat.read"
    assert meta["timer_cancelled_at"] > 0


@pytest.mark.asyncio
async def test_only_matching_pending_timers_on_that_chat_are_cancelled(timers, ids):
    read_chat_id, other_chat_id = f"{ids}-read-chat", f"{ids}-other-chat"
    rows = {
        "cancelled": _timer(timers, f"{ids}-read", "alice", read_chat_id),
        "other chat": _timer(timers, f"{ids}-elsewhere", "alice", other_chat_id),
        "not subscribed": _timer(
            timers, f"{ids}-on-message", "alice", read_chat_id, cancel_on=("chat.user_message",)
        ),
        "running": _timer(timers, f"{ids}-running", "alice", read_chat_id, status="running"),
    }
    await _seed(timers, list(rows.values()))

    await _cancel_as(timers, read_chat_id, "chat.read", "alice")

    statuses = await _statuses(timers, [row.id for row in rows.values()])
    assert {label: statuses[row.id] for label, row in rows.items()} == {
        "cancelled": "cancelled",
        "other chat": "pending",
        "not subscribed": "pending",
        "running": "running",
    }


@pytest.mark.asyncio
async def test_reading_a_chat_with_no_timers_is_a_no_op(timers, ids):
    await _cancel_as(timers, f"{ids}-empty-chat", "chat.read", "alice")
