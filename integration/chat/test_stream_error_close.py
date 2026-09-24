"""Regression: a failed reply was stored unfinished and replayed to the provider as an empty turn.

dbb17a572: when the provider answered with an error, the chat entrypoint stored the error on the
assistant message without `done`, so a reload showed a reply that never finished, and it
also cancelled the task queue. The next turn then rebuilt the conversation from the database
and sent the errored reply along as an assistant message with empty content. The fix stores the
error with `done: True` and leaves an errored reply without content out of the replayed history.

Twin of unit/chat/test_stream_error_close.py.

Discriminates: passes on bbfa876af; with dbb17a572's backend half reverted, the stored error has
no `done` and the second turn's provider request carries `{"role": "assistant", "content": ""}`.
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def wait_for_stored_error(client: httpx.Client, turn: ChatTurn, timeout: float = 30.0) -> dict:
    """The assistant message once its error is stored, without waiting on `done`."""
    deadline = time.monotonic() + timeout
    message: dict = {}
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        message = chat["history"]["messages"].get(turn.assistant_message_id, {})
        if message.get("error"):
            return message
        time.sleep(0.1)
    raise AssertionError(f"no error was stored on the reply; last stored state: {message}")


def replayed_roles_and_contents(request: dict) -> list[tuple[str, str]]:
    return [(entry["role"], entry.get("content")) for entry in request["messages"]]


@pytest.fixture
def client(make_user):
    with make_user().client() as client:
        yield client


def test_a_failed_reply_is_stored_finished_with_its_error(client, upstream):
    upstream.queue(reply.error(500, "the provider is down"))
    failed = wait_for_stored_error(client, send_message(client, "hello?"))

    assert "the provider is down" in failed["error"]["content"]
    assert failed.get("done") is True, (
        f"the failed reply was stored without done, so a reload shows it still running: {failed}"
    )


def test_the_next_turn_does_not_replay_the_empty_failed_reply(client, upstream):
    upstream.queue(reply.error(500, "the provider is down"), reply.text("back again"))
    failed_turn = send_message(client, "hello?")
    wait_for_stored_error(client, failed_turn)

    _, answer = ask(
        client,
        "hello again",
        chat_id=failed_turn.chat_id,
        parent_id=failed_turn.assistant_message_id,
    )

    assert answer["content"] == "back again"
    assert replayed_roles_and_contents(upstream.chat_requests()[-1]) == [
        ("user", "hello?"),
        ("user", "hello again"),
    ], "the failed reply was replayed to the provider as an empty assistant turn"


def test_an_answered_turn_is_still_replayed(client, upstream):
    upstream.queue(reply.text("first answer"), reply.text("second answer"))
    first_turn, _ = ask(client, "first question")

    ask(
        client,
        "second question",
        chat_id=first_turn.chat_id,
        parent_id=first_turn.assistant_message_id,
    )

    assert replayed_roles_and_contents(upstream.chat_requests()[-1]) == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
    ]
