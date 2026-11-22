"""Regression: a message sent in a long chat waited for three saves of the whole chat.

Every message sent to a saved chat stored the whole conversation three times before its reply
started: the new turn, then the reply's task (who may stop it), then the reply's start time. In a
chat of 130 MB each save took about 4.5 s on Postgres, so the send was accepted only after about
11 s and the reply finished after about 22 s; SQLite took 2 s and 5 s
(open-webui/open-webui#34672). The fix stores the task and the start with the new turn.

The narrow test counts the saves of the stored 1 MB chat (`harness/fat_chats.py`) with a trigger
on that one chat's row, up to the moment the provider has the reply's request while the reply is
held. Counting is exact where timing is not: on SQLite one save costs less than the reads around
it, so two saves and one take about as long. The nearby tests pin what those saves held, on both
refs: the reply is stored with the task the send got and a start before its end, a message sent
while the reply runs is answered after it with its own task and start, and in a chat shared to
continue each person's running reply is listed and stopped for that person only.

Discriminates: passes on the fix of #34672 (one save before the reply starts); fails on dev
d74882eee (three saves), on SQLite and Postgres. The nearby tests pass on both.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.backends import on_postgres, read_rows, write_rows
from harness.chat import ask, send_message, wait_for_reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def test_a_send_saves_the_long_chat_once_before_its_reply_starts(
    instance, make_user, fat_chat, upstream
):
    owner = make_user()
    fat = fat_chat(owner, "fat-1mb")
    held = threading.Event()
    prompt = f"one more question {uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text(f"answer to {prompt}", hold=held, match=reply.answering(prompt)))
    try:
        with _counting_saves(instance, fat.chat_id) as saves, owner.client() as client:
            turn = send_message(
                client, prompt, chat_id=fat.chat_id, parent_id=fat.manifest["last_reply"]
            )
            _wait_until_asked(upstream, prompt)
            before_the_reply = saves()
            held.set()
            finished = wait_for_reply(client, turn)
            after_the_reply = saves()
    finally:
        held.set()

    assert before_the_reply == 1, (
        f"the chat was saved {before_the_reply} times before its reply started; each save of a "
        f"long chat costs seconds, and the new messages need only one"
    )
    assert after_the_reply > before_the_reply, "the finished reply was never saved"
    assert finished["content"] == f"answer to {prompt}", f"the reply reads {finished['content']!r}"


def test_a_reply_is_stored_with_the_task_that_writes_it_and_its_start(
    make_user, fat_chat, upstream
):
    owner = make_user()
    fat = fat_chat(owner, "fat-1mb")
    held = threading.Event()
    prompt = f"a question for the long chat {uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text(f"answer to {prompt}", hold=held, match=reply.answering(prompt)))
    try:
        with owner.client() as client:
            turn = send_message(
                client, prompt, chat_id=fat.chat_id, parent_id=fat.manifest["last_reply"]
            )
            running = _running(client, fat.chat_id)
            while_running = _stored_reply(client, turn)
            held.set()
            finished = wait_for_reply(client, turn)
    finally:
        held.set()

    assert len(running) == 1, f"the running reply was listed as {running}"
    assert (while_running.get("meta") or {}).get("task_id") == running[0], (
        f"the running reply was stored with the task {while_running.get('meta')}, not {running[0]}"
    )
    assert finished["content"] == f"answer to {prompt}", (
        f"the stored reply reads {finished['content']!r}"
    )
    meta = finished.get("meta") or {}
    assert meta.get("task_id") == running[0], f"the finished reply lost its task: {meta}"
    started, ended = meta.get("response_started_at"), meta.get("response_ended_at")
    assert started and ended and started <= ended, (
        f"the reply's start and end were stored as {meta}"
    )


def test_a_message_sent_while_the_reply_runs_is_answered_after_it(make_user, fat_chat, upstream):
    owner = make_user()
    fat = fat_chat(owner, "fat-1mb")
    held = threading.Event()
    first, second = (f"{word} question {uuid.uuid4().hex[:8]}" for word in ("first", "second"))
    upstream.queue(reply.text(f"answer to {first}", hold=held, match=reply.answering(first)))
    upstream.queue(reply.text(f"answer to {second}", match=reply.answering(second)))
    try:
        with owner.client() as client:
            running = send_message(
                client, first, chat_id=fat.chat_id, parent_id=fat.manifest["last_reply"]
            )
            _wait_until_asked(upstream, first)
            # the web client asks to queue a message sent while a reply runs (bc754ef4e)
            send_message(
                client,
                second,
                chat_id=fat.chat_id,
                parent_id=running.assistant_message_id,
                queue=True,
            )
            held.set()
            first_reply = wait_for_reply(client, running)
            answer = _answer_after(client, fat.chat_id, running.assistant_message_id)
    finally:
        held.set()

    assert first_reply["content"] == f"answer to {first}", f"the first reply reads {first_reply}"
    assert answer.get("content") == f"answer to {second}", (
        f"the message sent while the reply ran was answered with {answer}"
    )
    asked = next(filter(reply.answering(second), upstream.chat_requests()))
    assert f"answer to {first}" in str(asked["messages"]), (
        "the message sent while the reply ran was answered without that reply in its context"
    )
    meta = answer.get("meta") or {}
    assert meta.get("task_id") and meta.get("response_started_at"), (
        f"the answer to the held message was stored without its task or start: {meta}"
    )


def test_in_a_shared_chat_each_person_sees_and_stops_only_their_own_reply(make_user, upstream):
    owner, member = make_user(), make_user()
    question = f"what is the capital of Norway? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Oslo.", match=reply.answering(question)))
    with owner.client() as client:
        first, _ = ask(client, question)
        _share(client, first.chat_id, member)

    held = threading.Event()
    follow_up = f"and the capital of Sweden? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Stockholm.", hold=held, match=reply.answering(follow_up)))
    try:
        with member.client() as member_client, owner.client() as owner_client:
            turn = send_message(
                member_client,
                follow_up,
                chat_id=first.chat_id,
                parent_id=first.assistant_message_id,
            )
            seen_by_member = _running(member_client, first.chat_id)
            seen_by_owner = _running(owner_client, first.chat_id)
            stopped_by_owner = owner_client.post(f"/api/tasks/chat/{first.chat_id}/stop")
            still_running = _running(member_client, first.chat_id)
            stopped_by_member = member_client.post(f"/api/tasks/chat/{first.chat_id}/stop")
            stored = wait_for_reply(member_client, turn)
    finally:
        held.set()

    assert len(seen_by_member) == 1, f"the member's running reply was listed as {seen_by_member}"
    assert seen_by_owner == [], f"the owner was shown the member's running reply: {seen_by_owner}"
    assert stopped_by_owner.status_code == 200, stopped_by_owner.text
    assert still_running == seen_by_member, (
        f"the owner's stop ended the member's reply: still running {still_running}"
    )
    assert stopped_by_member.status_code == 200, stopped_by_member.text
    assert stored["done"] is True and stored["content"] != "Stockholm.", (
        f"the member's stop did not end their reply: {stored}"
    )


def _share(client, chat_id: str, member) -> None:
    """Share the chat with `member` to continue, as the share dialog does."""
    shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
    assert shared.status_code == 200, shared.text
    saved = client.post(
        f"/api/v1/chats/shared/{chat_id}/access/update",
        json={"access_grants": [grant("user", member.id, "read")], "share_mode": "continue"},
    )
    assert saved.status_code == 200, saved.text


def _running(client, chat_id: str) -> list[str]:
    listed = client.get(f"/api/tasks/chat/{chat_id}")
    assert listed.status_code == 200, listed.text
    return listed.json()["task_ids"]


def _stored_reply(client, turn) -> dict:
    stored = client.get(f"/api/v1/chats/{turn.chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"]["history"]["messages"][turn.assistant_message_id]


@contextmanager
def _counting_saves(instance, chat_id: str) -> Iterator[Callable[[], int]]:
    """Count the saves of the chat's whole stored conversation, each seconds in a long chat."""
    probe = f"chat_save_probe_{uuid.uuid4().hex[:8]}"
    if on_postgres(instance):
        setup = [
            f"CREATE TABLE {probe} (chat_id TEXT)",
            f"CREATE FUNCTION {probe}() RETURNS trigger AS $$ BEGIN "
            f"INSERT INTO {probe} VALUES (NEW.id); RETURN NEW; END $$ LANGUAGE plpgsql",
            f"CREATE TRIGGER {probe} AFTER UPDATE OF chat ON chat FOR EACH ROW "
            f"WHEN (NEW.id = '{chat_id}') EXECUTE FUNCTION {probe}()",
        ]
        teardown = [
            f"DROP TRIGGER {probe} ON chat",
            f"DROP FUNCTION {probe}()",
            f"DROP TABLE {probe}",
        ]
    else:
        setup = [
            f"CREATE TABLE {probe} (chat_id TEXT)",
            f"CREATE TRIGGER {probe} AFTER UPDATE OF chat ON chat WHEN NEW.id = '{chat_id}' "
            f"BEGIN INSERT INTO {probe} VALUES (NEW.id); END",
        ]
        teardown = [f"DROP TRIGGER {probe}", f"DROP TABLE {probe}"]
    for statement in setup:
        write_rows(instance, statement, [{}])
    try:
        yield lambda: read_rows(instance, f"SELECT count(*) AS saves FROM {probe}")[0]["saves"]
    finally:
        for statement in teardown:
            write_rows(instance, statement, [{}])


def _wait_until_asked(upstream, prompt: str, timeout: float = 60.0) -> None:
    """Return once the provider has the reply's request, so the reply is about to start."""
    deadline = time.monotonic() + timeout
    while not any(map(reply.answering(prompt), upstream.chat_requests())):
        if time.monotonic() > deadline:
            raise AssertionError(f"the provider never received the request for {prompt!r}")
        time.sleep(0.05)


def _answer_after(client, chat_id: str, reply_id: str, timeout: float = 60.0) -> dict:
    """The finished answer to the message sent after `reply_id`, under the ids the server picks."""
    deadline = time.monotonic() + timeout
    answer: dict = {}
    while time.monotonic() < deadline:
        stored = client.get(f"/api/v1/chats/{chat_id}")
        assert stored.status_code == 200, stored.text
        messages = stored.json()["chat"]["history"]["messages"]
        asked = [messages[child] for child in messages[reply_id].get("childrenIds") or []]
        answers = [
            messages[child] for message in asked for child in message.get("childrenIds") or []
        ]
        answer = answers[-1] if answers else {}
        if answer.get("done"):
            return answer
        time.sleep(0.2)
    raise AssertionError(
        f"the message sent after {reply_id} was never answered; last seen {answer}"
    )
