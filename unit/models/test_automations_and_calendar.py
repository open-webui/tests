"""Timer regressions fixed in 0.11.1 that no endpoint can see.

* A timer whose chat completion raised kept the `completed` status written just before the
  handler ran, and the exception escaped the scheduler task (`f5a5a434b`, #27785, issue #27783).
  The timer now records `status: error` with the error text.
* Forking a timer chat copied its `meta` verbatim, and the scheduler claimed timers by `meta`,
  so the fork was a second claim target and the timer fired twice. Timers now hang off the
  `chat.timer_at` column, which a fork does not carry (`16c2a9eda`, #27663, issues
  #27622/#27745).

Both states live on internal timer chats that no endpoint lists, so these tests drive the public
timer functions against the real tables of the scratch SQLite, with the chat completion handler
(the one I/O boundary) stubbed. The API-visible parts of this file moved to
integration/models/test_automations_and_calendar.py.

Discriminates: passes on dev bbfa876af; with the try/except around the completion removed the
handler's error escapes `execute_due_timer`, and with the claim selecting by `meta.timer_at`
again the forked copy is claimed.
"""

from __future__ import annotations

import datetime as dt
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select
from starlette.applications import Starlette

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def db(owui_module):
    """`open_webui.internal.db`, after config has run the migrations."""
    owui_module("open_webui.config")
    return owui_module("open_webui.internal.db")


@pytest.fixture(scope="module")
def chats(db, owui_module):
    return owui_module("open_webui.models.chats")


@pytest.fixture(scope="module")
def users(db, owui_module):
    return owui_module("open_webui.models.users")


@pytest.fixture(scope="module")
def timers(db, owui_module):
    return owui_module("open_webui.utils.timers")


def app_with(handler) -> Starlette:
    """The app the scheduler hands a due timer to."""
    app = Starlette()
    app.state.redis = None
    app.state.CHAT_COMPLETION_HANDLER = handler
    return app


def well_past_due() -> int:
    return time.time_ns() + 10 * 60 * 1_000_000_000


async def set_timer(db, chats, users, timers, due_in_seconds: int = 60) -> str:
    """A fresh user sets a timer from their chat; returns the timer chat's id."""
    user_id = uuid4().hex
    user = await users.Users.insert_new_user(
        id=user_id, name="Timer user", email=f"{user_id}@example.test", role="user"
    )
    parent_id = f"{user_id}-parent"
    await chats.Chats.insert_new_chat(
        id=parent_id,
        user_id=user_id,
        form_data=chats.ChatForm(chat={"title": "Parent", "history": {"messages": {}}}),
    )
    due_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=due_in_seconds)
    status = await timers.create_timer(
        prompt="remind me",
        at=due_at.isoformat(),
        cancel_on=[],
        request=None,
        user_data=user.model_dump(),
        metadata={"model_id": "test-model"},
        parent_chat_id=parent_id,
        parent_message_id=None,
    )
    assert not status.startswith("Error"), status

    async with db.get_async_db_context() as session:
        rows = await session.execute(
            select(chats.Chat.id).where(chats.Chat.user_id == user_id, chats.Chat.id != parent_id)
        )
    (timer_id,) = rows.scalars().all()
    return timer_id


async def run_due_timer(timers, timer_id: str, handler) -> None:
    claimed = dict(await timers.claim_due_timers(now_ns=well_past_due(), limit=100))
    await timers.execute_due_timer(
        app=app_with(handler), timer_id=timer_id, claim_id=claimed[timer_id]
    )


# ---------------------------------------------------------------- narrow


@pytest.mark.asyncio
async def test_a_failed_completion_marks_the_timer_as_errored(db, chats, users, timers):
    timer_id = await set_timer(db, chats, users, timers)
    handler = AsyncMock(side_effect=RuntimeError("model not found"))

    await run_due_timer(timers, timer_id, handler)

    handler.assert_awaited_once()
    meta = (await chats.Chats.get_chat_by_id(timer_id)).meta
    assert meta["status"] == "error", "a failed completion was still reported as delivered"
    assert "model not found" in meta["timer_error"]


@pytest.mark.asyncio
async def test_a_forked_timer_chat_is_not_claimed(db, chats, users, timers):
    """The fork route re-inserts the source meta, which used to define a claimable timer."""
    timer_id = await set_timer(db, chats, users, timers)
    timer = await chats.Chats.get_chat_by_id(timer_id)
    fork_id = f"{timer_id}-fork"
    await chats.Chats.insert_new_chat(
        id=fork_id,
        user_id=timer.user_id,
        form_data=chats.ChatForm(chat={"title": "Timer (fork)", "history": {"messages": {}}}),
        internal_meta={**timer.meta, "forked_from": timer_id},
    )

    claimed = dict(await timers.claim_due_timers(now_ns=well_past_due(), limit=100))

    assert timer_id in claimed, "the real timer was not claimed"
    assert fork_id not in claimed, "the forked copy was claimed as a second timer"


# ---------------------------------------------------------------- nearby


@pytest.mark.asyncio
async def test_a_successful_completion_stays_completed(db, chats, users, timers):
    timer_id = await set_timer(db, chats, users, timers)

    await run_due_timer(timers, timer_id, AsyncMock(return_value={}))

    meta = (await chats.Chats.get_chat_by_id(timer_id)).meta
    assert meta["status"] == "completed"
    assert "timer_error" not in meta


@pytest.mark.asyncio
async def test_a_timer_not_yet_due_is_left_alone(db, chats, users, timers):
    timer_id = await set_timer(db, chats, users, timers, due_in_seconds=3600)

    claimed = dict(await timers.claim_due_timers(now_ns=time.time_ns(), limit=100))

    assert timer_id not in claimed
