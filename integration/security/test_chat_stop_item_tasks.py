"""Regression: stopping a chat stops every reply in flight, not just the first ones.

open-webui 0.11.4 fix for #29816 (PR #29844, `0313ea023` and `e35b907f7`): without Redis,
`stop_item_tasks` walked the live `item_tasks[chat_id]` list while each stopped task's cleanup
removed itself from that same list, so with three replies in flight the third was skipped and
kept streaming to its end. It also returned early on the first task it could no longer find: a
reply that finished on its own while an earlier one was still being stopped ended the stop, and
every reply listed after it kept running. The fix walks a copy and stops every task in turn.

For the second half the test holds the replies at chosen points: an admin's tool keeps the
first reply busy and, once stopped, waits for the test before it lets go; a second tool holds
the second reply until the test releases it inside that window, so it finishes on its own while
the first is still being stopped.

Discriminates: passes on dev ef67cc3fa; with both commits reverted (the live list and the early
return) the third reply's task is still running after the stop in the three replies test, and
with the early return of `e35b907f7` restored alone it is still running in the finishing reply
test. Reverting the live list alone stays green: each copy of the list is enough on its own.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.inflight import start_slow_reply
from harness.python_tools import python_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


HOLDING_TOOLS = """
import asyncio
import urllib.request


def _call(url):
    urllib.request.urlopen(url, timeout=60).read()


class Tools:
    async def hold(self, ready_url: str, stopping_url: str) -> str:
        \"\"\"Keep the reply busy until it is stopped, then wait for the test to let go.\"\"\"
        await asyncio.to_thread(_call, ready_url)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.to_thread(_call, stopping_url)
            raise
        return "never stopped"

    async def wait_for(self, url: str) -> str:
        \"\"\"Hold the reply until the test releases it.\"\"\"
        await asyncio.to_thread(_call, url)
        return "released"
"""


def _running_tasks(client, chat_id: str) -> list[str]:
    return client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]


def _wait_until(condition, what: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting until {what}")
        time.sleep(0.05)


def _held_until(release: threading.Event):
    def answer(_request):
        release.wait(timeout=30)
        return 200, {"Content-Type": "text/plain"}, b"ok"

    return answer


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


def test_a_reply_finishing_during_the_stop_does_not_end_it_early(
    instance, admin, make_user, upstream, listener
):
    if instance.redis_url:
        pytest.skip("with Redis the stop sends every cancel without waiting on any, so no window")
    first_may_stop, second_may_finish = threading.Event(), threading.Event()
    listener.route("GET", "/ready", (200, {"Content-Type": "text/plain"}, b"ok"))
    listener.route("GET", "/stopping", _held_until(first_may_stop))
    listener.route("GET", "/second", _held_until(second_may_finish))
    hold = {
        "ready_url": f"{listener.base_url}/ready",
        "stopping_url": f"{listener.base_url}/stopping",
    }
    upstream.queue(
        reply.tool_call("hold", hold, match=reply.answering("hold the line")),
        reply.tool_call(
            "wait_for", {"url": f"{listener.base_url}/second"}, match=reply.answering("wait for me")
        ),
        reply.text("the second reply finished", match=reply.answering("wait for me")),
    )
    person = make_user()
    with (
        python_tool(admin, HOLDING_TOOLS) as tool_id,
        person.client() as client,
        person.client() as stopping_client,
        ThreadPoolExecutor(max_workers=1) as background,
    ):
        first = send_message(client, "hold the line", tool_ids=[tool_id])
        try:
            _wait_until(lambda: listener.requests_to("/ready"), "the first reply's tool runs")
            second = send_message(client, "wait for me", chat_id=first.chat_id, tool_ids=[tool_id])
            _wait_until(lambda: listener.requests_to("/second"), "the second reply's tool runs")
            start_slow_reply(client, upstream, chunk_delay=0.5, chat_id=first.chat_id)
            assert len(_running_tasks(client, first.chat_id)) == 3

            stopping = background.submit(
                stopping_client.post, f"/api/tasks/chat/{first.chat_id}/stop"
            )
            _wait_until(lambda: listener.requests_to("/stopping"), "the first reply is stopping")
            second_may_finish.set()
            assert wait_for_reply(client, second)["content"] == "the second reply finished"
            _wait_until(
                lambda: len(_running_tasks(client, first.chat_id)) == 2,
                "the second reply's task is gone",
            )
            first_may_stop.set()
            assert stopping.result(timeout=30).status_code == 200

            # The third reply streams for ten seconds, so a skipped one is still running at two.
            deadline = time.monotonic() + 2
            while _running_tasks(client, first.chat_id) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert _running_tasks(client, first.chat_id) == [], (
                "a reply that finished while the chat was being stopped ended the stop early, "
                "so the reply listed after it kept streaming (#29816)"
            )
        finally:
            first_may_stop.set()
            second_may_finish.set()
            client.post(f"/api/tasks/chat/{first.chat_id}/stop")
