"""Regression: a failed reply was stored unfinished and replayed to the provider as an empty turn.

dbb17a572: when the provider answered with an error, the chat entrypoint stored the error on the
assistant message without `done`, so a reload showed a reply that never finished, and it
also cancelled the task queue. The next turn then rebuilt the conversation from the database
and sent the errored reply along as an assistant message with empty content. The fix stores the
error with `done: True` and leaves an errored reply without content out of the replayed history.

Twin of unit/chat/test_stream_error_close.py.

Discriminates: passes on bbfa876af; with dbb17a572's backend half reverted, the stored error has
no `done` and the second turn's provider request carries `{"role": "assistant", "content": ""}`.

open-webui/open-webui#34814: Continue Response on a reply whose provider dropped mid-stream kept the
old error on the stored message, so the recovered answer showed the failure under it after every
reload. Continuing now clears it, and a continuation that fails again stores its own error.

Discriminates (#34814): the continue test fails on dev c1f3b6fd1 (the stored message keeps the drop
error next to the continued answer) and passes with the fix.
"""

from __future__ import annotations

import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, ask, send_message

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


def wait_for_stored(client: httpx.Client, turn: ChatTurn, settled, timeout: float = 30.0) -> dict:
    """The assistant message once `settled` accepts it; a continued reply is stored done already."""
    deadline = time.monotonic() + timeout
    message: dict = {}
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        message = chat["history"]["messages"].get(turn.assistant_message_id, {})
        if settled(message):
            return message
        time.sleep(0.1)
    raise AssertionError(f"the reply never settled; last stored state: {message}")


def continue_reply(client: httpx.Client, turn: ChatTurn, prompt: str, partial: str) -> None:
    """Press Continue Response on the reply, as the web client sends it."""
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    user_message = chat["history"]["messages"][turn.user_message_id]
    accepted = client.post(
        "/api/chat/completions",
        json={
            "model": reply.MOCK_MODEL_ID,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": partial},
            ],
            "stream": True,
            "chat_id": turn.chat_id,
            "id": turn.assistant_message_id,
            "assistant_message_id": turn.assistant_message_id,
            "parent_id": user_message["parentId"],
            "user_message": user_message,
            "session_id": "harness-continue",
            "background_tasks": NO_BACKGROUND_TASKS,
        },
    )
    assert accepted.status_code == 200, accepted.text


def replayed_roles_and_contents(request: dict) -> list[tuple[str, str]]:
    return [(entry["role"], entry.get("content")) for entry in reply.conversation(request)]


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


def test_continuing_a_dropped_reply_clears_its_error(client, upstream):
    prompt = "tell a tale"
    upstream.queue(
        reply.text(["Once ", "upon ", "a time"], hang_up_after=1, match=reply.answering(prompt)),
        reply.text([" there ", "was ", "a fox."]),
    )
    dropped_turn = send_message(client, prompt)
    dropped = wait_for_stored_error(client, dropped_turn)

    continue_reply(client, dropped_turn, prompt, dropped["content"])
    continued = wait_for_stored(
        client, dropped_turn, lambda message: message.get("content", "").endswith("a fox.")
    )

    assert not continued.get("error"), (
        f"the continued answer still shows the failed attempt's error: {continued['error']}"
    )


def test_a_continuation_that_fails_again_stores_its_own_error(client, upstream):
    prompt = "tell another tale"
    upstream.queue(
        reply.text(["Once ", "upon ", "a time"], hang_up_after=1, match=reply.answering(prompt)),
        reply.error(500, "the provider is down again"),
    )
    dropped_turn = send_message(client, prompt)
    dropped = wait_for_stored_error(client, dropped_turn)

    continue_reply(client, dropped_turn, prompt, dropped["content"])
    failed_again = wait_for_stored(
        client,
        dropped_turn,
        lambda message: "down again" in (message.get("error") or {}).get("content", ""),
    )

    assert failed_again.get("done") is True
