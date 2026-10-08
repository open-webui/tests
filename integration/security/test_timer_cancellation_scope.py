"""Regression: marking someone else's chat as read cancelled their timers.

open-webui 0.11.0 `e140d8f3c` (#27472): `cancel_timers_for_chat` selected pending timers on the
parent chat without filtering on their owner, and the `events:chat` `last_read_at` socket
handler ran it even when the reader did not own the chat. Anyone who knew a chat id could open
a socket and silently cancel that owner's scheduled prompt. The fix filters on the owner and
returns early for a reader who does not own the chat.

Here the model sets a real timer through the `timer` tool (behind ENABLE_SUBAGENTS), another
account sends the read event for that chat over its own socket, and the test waits for the
timer's prompt to reach the model. The owner filter shows on its own when someone the owner
shared the chat with to continue sends a message there: only that person's own timers may go.
(Before 6cfd6987e this was an admin continuing the chat; an admin may now only read it.)

Twin of unit/security/test_timer_cancellation_scope.py, which keeps the audit that every caller
passes the acting user.

`test_a_collaborator_message_in_the_chat_leaves_the_owners_timer_running`,
`test_a_read_leaves_timers_it_does_not_match` and
`test_a_stranger_reading_the_chat_leaves_the_owners_timer_running` are red on dev 62f70a844: since
de73bb830 a chat request whose reply message is already stored in the chat, the way automations,
sub-agents and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev `ef67cc3fa`; with `e140d8f3c` reverted (the query's owner filter
and the handler's early return) the stranger's read cancels the timer and it never fires, and
with the owner filter alone dropped the collaborator's message cancels the owner's timer (shown
on dev 62f70a844 with the #32066 refusal removed). Dropping the parent chat or the `cancel_on`
match from the query fails the nearby tests.
"""

from __future__ import annotations

import json
import time

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask
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


def _set_timer(owner, upstream, *, cancel_on=("chat.read",), at="3s", **turn) -> ChatTurn:
    """The owner's model sets a timer; `turn` continues an existing chat."""
    arguments = {"prompt": TIMER_PROMPT, "at": at, "cancel_on": list(cancel_on)}
    upstream.queue(reply.tool_call("timer", arguments), reply.text("Timer set."))
    with owner.client() as client:
        chat_turn, message = ask(client, "remind me in a few seconds", **turn)
    result = next(item for item in message["output"] if item["type"] == "function_call_output")
    assert json.loads(result["output"][0]["text"])["status"] == "set", result
    return chat_turn


def _read_chat_as(reader, chat_id: str) -> None:
    with connected(reader) as socket:
        socket.call("events:chat", {"chat_id": chat_id, "data": {"type": "last_read_at"}})


def _fired_count(upstream) -> int:
    """How many timer prompts have reached the model."""
    last_messages = [request["messages"][-1] for request in upstream.chat_requests()]
    return sum(
        1
        for message in last_messages
        if message["role"] == "user" and TIMER_PROMPT in str(message["content"])
    )


def _timer_fired(upstream, within: float, timers: int = 1) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if _fired_count(upstream) >= timers:
            return True
        time.sleep(0.2)
    return False


def test_a_stranger_reading_the_chat_leaves_the_owners_timer_running(
    timers_enabled, make_user, upstream
):
    owner, stranger = make_user(), make_user()
    chat_id = _set_timer(owner, upstream).chat_id

    _read_chat_as(stranger, chat_id)

    assert _timer_fired(upstream, within=20), (
        "another account marking the chat as read cancelled the owner's timer, so its "
        "scheduled prompt never fired (#27472)"
    )


def test_the_owner_reading_the_chat_still_cancels_the_timer(timers_enabled, make_user, upstream):
    owner = make_user()
    chat_id = _set_timer(owner, upstream).chat_id

    _read_chat_as(owner, chat_id)

    # due after three seconds and polled every second, so six seconds is ample
    assert not _timer_fired(upstream, within=6), "the owner's own read no longer cancels"


def _continue_chat(actor, upstream, turn: ChatTurn) -> None:
    upstream.queue(reply.text("Noted."))
    with actor.client() as client:
        ask(client, "one more thing", chat_id=turn.chat_id, parent_id=turn.assistant_message_id)


def _share_to_continue(owner, chat_id: str, collaborator) -> None:
    """Share the way the share dialog does with Allow replies: the link, then the grant."""
    grant = {"principal_type": "user", "principal_id": collaborator.id, "permission": "read"}
    with owner.client() as client:
        link = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": "continue"})
        assert link.status_code == 200, link.text
        shared = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )
        assert shared.status_code == 200, shared.text


def test_a_collaborator_message_in_the_chat_leaves_the_owners_timer_running(
    timers_enabled, make_user, upstream
):
    """Narrow: someone continuing the owner's shared chat cancels only their own timers."""
    owner, collaborator = make_user(), make_user()
    turn = _set_timer(owner, upstream, cancel_on=["chat.user_message"], at="6s")
    _share_to_continue(owner, turn.chat_id, collaborator)

    _continue_chat(collaborator, upstream, turn)

    assert _timer_fired(upstream, within=25), (
        "the collaborator's message in the owner's chat cancelled the owner's timer, so its "
        "scheduled prompt never fired (#27472)"
    )


def test_the_owners_next_message_cancels_their_timer(timers_enabled, make_user, upstream):
    owner = make_user()
    turn = _set_timer(owner, upstream, cancel_on=["chat.user_message"], at="4s")

    _continue_chat(owner, upstream, turn)

    assert not _timer_fired(upstream, within=8), "the owner's own message no longer cancels"


def test_a_read_leaves_timers_it_does_not_match(timers_enabled, make_user, upstream):
    """Nearby: the read chat's timer waiting for a message, and another chat's timer that a
    read would cancel, both survive it."""
    owner = make_user()
    read_chat = _set_timer(owner, upstream, cancel_on=["chat.user_message"], at="6s")
    _set_timer(owner, upstream, at="6s")

    _read_chat_as(owner, read_chat.chat_id)

    assert _timer_fired(upstream, within=25, timers=2), (
        f"only {_fired_count(upstream)} of two timers fired"
    )
