"""Regression: a reply stopped or broken before any text kept its chat unusable.

Fix 743a46bdc (PR #31892, issue #25083) skips every empty assistant message when the chat is
replayed to the model. Before, only an empty reply that carried an error was skipped, so a reply
stopped before its first token, or one that ended without text or error, went back to the provider
as an empty assistant message on every later turn, and providers that refuse those refused the
chat for good.

Discriminates: passes on dev b859124f9, fails with 743a46bdc reverted (the empty reply is replayed
as `("assistant", "")` between the two questions).
"""

from __future__ import annotations

import time

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message
from harness.chat_history import seed_chat
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def replayed(upstream) -> list[tuple[str, str]]:
    return [
        (entry["role"], entry["content"]) for entry in conversation(upstream.chat_requests()[-1])
    ]


def wait_for_provider_request(upstream, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not upstream.chat_requests():
        assert time.monotonic() < deadline, "the provider never received the first question"
        time.sleep(0.05)


def wait_until_stopped(client, chat_id: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]:
        assert time.monotonic() < deadline, "the stopped reply's task never ended"
        time.sleep(0.1)


def test_a_reply_stopped_before_any_text_is_not_replayed(make_user, upstream):
    upstream.queue(reply.text("too late", delay=10, match=reply.answering("first question")))
    with make_user().client() as client:
        turn = send_message(client, "first question")
        wait_for_provider_request(upstream)
        stopped = client.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        wait_until_stopped(client, turn.chat_id)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]
        upstream.queue(reply.text("here you go", match=reply.answering("second question")))
        _, answered = ask(
            client,
            "second question",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )

    assert stopped.status_code == 200, stopped.text
    assert not stored["messages"][turn.assistant_message_id].get("content")
    assert replayed(upstream) == [("user", "first question"), ("user", "second question")]
    assert answered["content"] == "here you go"


def test_an_empty_reply_without_an_error_is_not_replayed(make_user, upstream):
    with make_user().client() as client:
        chat_id, empty_id = seed_chat(
            client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": ""},
            ],
        )
        ask(client, "second question", chat_id=chat_id, parent_id=empty_id)

    assert replayed(upstream) == [("user", "first question"), ("user", "second question")]


def test_an_empty_reply_with_an_empty_output_list_is_not_replayed(make_user, upstream):
    with make_user().client() as client:
        chat_id, empty_id = seed_chat(
            client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "", "output": []},
            ],
        )
        ask(client, "second question", chat_id=chat_id, parent_id=empty_id)

    assert replayed(upstream) == [("user", "first question"), ("user", "second question")]


def test_a_reply_with_text_is_still_replayed(make_user, upstream):
    with make_user().client() as client:
        chat_id, answered_id = seed_chat(
            client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        )
        ask(client, "second question", chat_id=chat_id, parent_id=answered_id)

    assert replayed(upstream) == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
    ]
