"""A chat sent under a temporary id is answered over the socket and never stored.

The web client marks a temporary chat by sending its completion with a `temporary:` chat id
made from its socket session id. The server streams the reply to that session only and creates
no chat, so the account's chat list and search stay empty while the same message under no id
makes a saved chat.

With follow-up generation on, the temporary chat also gets its follow-up suggestions over the
socket. Two regressions ride on that walk. Commit 53a8f131f: the follow-ups of a temporary chat
were asked for with the messages the client sent, which end at the question, so the task model
never saw the reply just written. PR #32323 (1c5f101d0): a task model that wrapped its JSON in
extra text got no suggestions shown, because only code fences were stripped before parsing.

Twin of e2e/chat/test_temporary_chat.py.

Discriminates: passes on dev 176d31d1d, fails with `is_saved_chat_id` treating a `temporary:` id
as saved (the temporary chat is stored). The follow-up checks pass on dev 5032e834e; with the
follow-up request built from the client's messages again (53a8f131f undone) it lacks the reply,
and with 1c5f101d0 reverted the follow-ups arrive empty.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PROMPT = "Which lake is the deepest in Carinthia?"
ANSWER = "That is the Millstatter See."
FOLLOW_UPS = ["How deep is the Millstatter See?", "Can you swim in it?"]
# the JSON key the default follow-up prompt asks for
FOLLOW_UP_TASK = '"follow_ups"'
TASKS_CONFIG = ("/api/v1/tasks/config", "/api/v1/tasks/config/update")


@pytest.fixture
def follow_ups_on(admin, preserve):
    preserve(TASKS_CONFIG)
    with admin.client() as client:
        current = client.get(TASKS_CONFIG[0]).json()
        saved = client.post(TASKS_CONFIG[1], json={**current, "ENABLE_FOLLOW_UP_GENERATION": True})
    assert saved.status_code == 200, saved.text


def chats_about_carinthia(client) -> list[dict]:
    found = client.get("/api/v1/chats/search", params={"text": "Carinthia"})
    found.raise_for_status()
    return found.json()


def follow_up_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(FOLLOW_UP_TASK)(body)]


@pytest.mark.regression
def test_a_temporary_chat_is_answered_over_the_socket_and_not_stored(
    make_user, upstream, follow_ups_on
):
    account = make_user()
    wrapped = f"Here are some ideas:\n{json.dumps({'follow_ups': FOLLOW_UPS})}\nHope that helps!"
    # queued first, so the follow-up task, which quotes the question, cannot take the chat's reply
    upstream.queue(
        reply.text(wrapped, match=reply.answering(FOLLOW_UP_TASK)),
        reply.text(ANSWER, match=reply.answering(PROMPT)),
    )
    with connected(account) as socket, account.client() as client:
        chat_id = socket.temporary_chat_id()

        send_message(
            client,
            PROMPT,
            chat_id=chat_id,
            session_id=socket.sid,
            background_tasks={"follow_up_generation": True},
        )

        streamed = socket.wait_for(chat_id, "chat:completion", done=True)
        assert ANSWER in str(streamed["data"]["output"])
        suggested = socket.wait_for(chat_id, "chat:message:follow_ups")
        assert chats_about_carinthia(client) == []
        assert client.get("/api/v1/chats/").json() == []
        assert client.get(f"/api/v1/chats/{chat_id}").status_code != 200

    (asked,) = follow_up_requests(upstream)
    assert ANSWER in json.dumps(asked), (
        "the follow-ups of a temporary chat were asked for without the reply just written"
    )
    assert suggested["data"]["follow_ups"] == FOLLOW_UPS, (
        "follow-ups the task model wrapped in extra text were dropped (#32323)"
    )


def test_the_same_message_without_a_temporary_id_is_stored(make_user, upstream):
    account = make_user()
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with account.client() as client:
        turn = send_message(client, PROMPT)
        wait_for_reply(client, turn)

        assert [chat["id"] for chat in chats_about_carinthia(client)] == [turn.chat_id]
