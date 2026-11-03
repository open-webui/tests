"""Chats that are never saved are not offered the task-list tools, and a reply reloads mid-stream.

0.11.0 `d484a2a`, `d2936c8`, `b45c020` and `71c4da8` (issue #27432): call sites open-coded
`startswith(('local:', 'channel:'))` and never learned the `temporary:` prefix the web client
now gives temporary chats. `get_builtin_tools` therefore offered `create_tasks` and
`update_task` to a temporary chat, whose task state has no chats row to live on.
`utils/chat_id.is_saved_chat_id` is now the single answer, and only a saved chat gets them; a
model answering in a channel does not either.

0.11.0 `aadab2f`: `create_task` minted its own task id, while the reply streamed its progress
under the id `chat_completion` had stamped into the turn. A chat opened while its reply was
still streaming looks its running tasks up by the registered id, so it found no progress and
showed an empty reply until the end. `create_task` now registers the caller's id.

Twin of unit/chat/test_socket_and_redis_runtime.py.

Discriminates: passes on dev bbfa876af and ef67cc3fa; with the task-list gate keyed on the `local:`
and `channel:` prefixes again, the `temporary:` chat is offered both tools; with `create_task`
ignoring the id it is handed, or `chat_completion` not handing it over, the reloaded chat shows no
progress while the reply streams.
"""

from __future__ import annotations

import time

import pytest

from harness import channel_chat
from harness.chat import send_message, wait_for_reply
from harness.inflight import start_slow_reply
from harness.socket_client import connected
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TASK_LIST_TOOLS = {"create_tasks", "update_task"}


def _offered_tools(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _first_provider_request(upstream, timeout: float = 30.0) -> dict:
    """A chat that is never saved has no stored reply to wait for, so wait for the provider."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sent := upstream.chat_requests():
            return sent[0]
        time.sleep(0.1)
    raise AssertionError("the chat never reached the provider")


@pytest.mark.parametrize("prefix", ["temporary:", "local:"])
def test_a_temporary_chat_is_not_offered_the_task_list_tools(make_user, upstream, prefix):
    person = make_user()
    with connected(person) as tab, person.client() as client:
        send_message(
            client, "plan my week", chat_id=tab.temporary_chat_id(prefix), session_id=tab.sid
        )
        offered = _offered_tools(_first_provider_request(upstream))

    assert offered, "no builtin tools were offered at all; the check below would prove nothing"
    assert not offered & TASK_LIST_TOOLS, (
        f"a {prefix} chat was offered {sorted(offered & TASK_LIST_TOOLS)}, which have no chats "
        "row to store their state on (#27432)"
    )


def test_a_saved_chat_is_offered_the_task_list_tools(make_user, upstream):
    with make_user().client() as client:
        wait_for_reply(client, send_message(client, "plan my week"))

    assert TASK_LIST_TOOLS <= _offered_tools(upstream.chat_requests()[0])


def test_a_channel_turn_is_not_offered_the_task_list_tools(admin, upstream, preserve):
    preserve("admin_config")
    with admin.client() as client:
        channel_chat.enable_channels(client)
        channel_chat.mention(client, MOCK_MODEL_ID, "plan my week")

    offered = _offered_tools(_first_provider_request(upstream))
    assert offered, "no builtin tools were offered at all; the check below would prove nothing"
    assert not offered & TASK_LIST_TOOLS


def test_a_chat_opened_while_its_reply_streams_shows_the_reply_so_far(make_user, upstream):
    with make_user().client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.2)
        seen_mid_stream = ""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()
            message = stored["chat"]["history"]["messages"][turn.assistant_message_id]
            if message.get("done"):
                break
            seen_mid_stream = message.get("content") or seen_mid_stream
            time.sleep(0.2)
        wait_for_reply(client, turn)

    assert seen_mid_stream.startswith("part-0"), (
        "a chat opened while its reply streamed showed none of the reply so far"
    )
