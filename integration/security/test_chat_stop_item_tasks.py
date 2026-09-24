"""Regression: stopping a chat stops every reply in flight, not just the first ones.

open-webui 0.11.4 fix for #29816 (PR #29844, `0313ea023` and `e35b907f7`): without Redis,
`stop_item_tasks` walked the live `item_tasks[chat_id]` list while each stopped task's cleanup
removed itself from that same list, so with three replies in flight the third was skipped and
kept streaming to its end. It also returned early on the first task that had already finished.
The fix walks a copy and stops every task in turn.

Twin of unit/security/test_chat_stop_item_tasks.py.

Discriminates: passes on dev bbfa876af; with both commits reverted (the live list and the early
return) the third reply's task is still running after the stop. Reverting either commit alone
stays green: each copy of the list is enough on its own.
"""

from __future__ import annotations

import time

import pytest

from harness.inflight import start_slow_reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _running_tasks(client, chat_id: str) -> list[str]:
    return client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]


def test_stopping_a_chat_stops_all_three_replies_in_flight(make_user, upstream):
    with make_user().client() as client:
        first = start_slow_reply(client, upstream, chunk_delay=0.2)
        for _ in range(2):
            start_slow_reply(client, upstream, chunk_delay=0.2, chat_id=first.chat_id)
        assert len(_running_tasks(client, first.chat_id)) == 3

        stopped = client.post(f"/api/tasks/chat/{first.chat_id}/stop")
        assert stopped.status_code == 200, stopped.text

        # The replies stream for four seconds, so a skipped one is still running at two.
        deadline = time.monotonic() + 2
        while _running_tasks(client, first.chat_id) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _running_tasks(client, first.chat_id) == [], (
            "stopping the chat left a reply streaming: the stop skipped a task (#29816)"
        )


def test_stopping_a_chat_with_nothing_in_flight_is_harmless(make_user, upstream):
    with make_user().client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": "idle"}}).json()

        stopped = client.post(f"/api/tasks/chat/{created['id']}/stop")

    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] is True
    assert "No tasks" in stopped.json()["message"]


def test_a_stranger_cannot_stop_someone_elses_chat(make_user, upstream):
    owner, stranger = make_user(), make_user()
    with owner.client() as owner_client, stranger.client() as stranger_client:
        turn = start_slow_reply(owner_client, upstream)

        refused = stranger_client.post(f"/api/tasks/chat/{turn.chat_id}/stop")

        assert refused.status_code == 404, refused.text
        assert _running_tasks(owner_client, turn.chat_id)
