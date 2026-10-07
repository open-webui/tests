"""Replies still streaming while a test does something else to their chat."""

from __future__ import annotations

import time

import httpx

from harness import upstream as reply
from harness.chat import ChatTurn, send_message
from harness.upstream import MockUpstream

SLOW_PIECES = [f"part-{index} " for index in range(20)]
LAST_PIECE = "part-19"


def start_slow_reply(
    client: httpx.Client, upstream: MockUpstream, chunk_delay: float = 0.1, **options
) -> ChatTurn:
    """Send a message whose reply streams for `len(SLOW_PIECES) * chunk_delay` seconds.

    Returns once the provider has the request, so the reply's task is running.
    """
    already_sent = len(upstream.chat_requests())
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=chunk_delay))
    turn = send_message(client, "tell me a long story", **options)
    deadline = time.monotonic() + 15
    while len(upstream.chat_requests()) == already_sent:
        if time.monotonic() > deadline:
            raise AssertionError("the provider never received the slow reply's request")
        time.sleep(0.05)
    return turn


def wait_until_no_reply_runs(client: httpx.Client, chat_id: str, timeout: float = 30.0) -> None:
    """Return once the server runs no reply for the chat any more."""
    deadline = time.monotonic() + timeout
    while client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]:
        if time.monotonic() > deadline:
            raise AssertionError(f"a reply in chat {chat_id} was still running after {timeout}s")
        time.sleep(0.1)
