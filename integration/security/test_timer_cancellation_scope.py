"""Regression: marking someone else's chat as read cancelled their timers.

open-webui 0.11.0 `e140d8f3c` (#27472): `cancel_timers_for_chat` selected pending timers on the
parent chat without filtering on their owner, and the `events:chat` `last_read_at` socket
handler ran it even when the reader did not own the chat. Anyone who knew a chat id could open
a socket and silently cancel that owner's scheduled prompt. The fix filters on the owner and
returns early for a reader who does not own the chat.

Here the model sets a real timer through the `timer` tool (behind ENABLE_SUBAGENTS), another
account sends the read event for that chat over its own socket, and the test waits for the
timer's prompt to reach the model.

Twin of unit/security/test_timer_cancellation_scope.py.

Discriminates: passes on dev `bbfa876af`; with `e140d8f3c` reverted (the query's owner filter
and the handler's early return) the stranger's read cancels the timer and it never fires.
"""

from __future__ import annotations

import json
import time

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "The oven timer went off."


@pytest.fixture
def timers_enabled(preserve, admin):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def _set_timer(owner, upstream) -> str:
    """The owner's model sets a timer that a read of the chat cancels; returns the chat id."""
    arguments = {"prompt": TIMER_PROMPT, "at": "3s", "cancel_on": ["chat.read"]}
    upstream.queue(reply.tool_call("timer", arguments), reply.text("Timer set."))
    with owner.client() as client:
        turn, message = ask(client, "remind me in three seconds")
    result = next(item for item in message["output"] if item["type"] == "function_call_output")
    assert json.loads(result["output"][0]["text"])["status"] == "set", result
    return turn.chat_id


def _read_chat_as(reader, chat_id: str) -> None:
    with connected(reader) as socket:
        socket.call("events:chat", {"chat_id": chat_id, "data": {"type": "last_read_at"}})


def _timer_fired(upstream, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        for request in upstream.chat_requests():
            last = request["messages"][-1]
            if last["role"] == "user" and TIMER_PROMPT in str(last["content"]):
                return True
        time.sleep(0.2)
    return False


def test_a_stranger_reading_the_chat_leaves_the_owners_timer_running(
    timers_enabled, make_user, upstream
):
    owner, stranger = make_user(), make_user()
    chat_id = _set_timer(owner, upstream)

    _read_chat_as(stranger, chat_id)

    assert _timer_fired(upstream, within=20), (
        "another account marking the chat as read cancelled the owner's timer, so its "
        "scheduled prompt never fired (#27472)"
    )


def test_the_owner_reading_the_chat_still_cancels_the_timer(timers_enabled, make_user, upstream):
    owner = make_user()
    chat_id = _set_timer(owner, upstream)

    _read_chat_as(owner, chat_id)

    # due after three seconds and polled every second, so six seconds is ample
    assert not _timer_fired(upstream, within=6), "the owner's own read no longer cancels"
