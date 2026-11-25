"""Journey: a message sent to one's own chat while its reply works steers that reply.

Since 0.13.4 a message a person sends to their own saved chat while a reply is still running
waits on the server, so it is still there when they reload the page or open the chat elsewhere.
The running reply wraps up at its next step, after the tool call it is making, and does not go on
with its plan. The waiting messages are then answered together, as one question, with what the
reply did so far: its tool call and the result reach the model with them. A queued message the
person deletes before that step is never answered, and the reply then goes on as planned.
The browser side (the queue row after a reload, Send now) is pinned in
e2e/chat/test_send_queue.py.

Discriminates: passes on dev f80a4f782. In backend copies: the reply going on after its tool call
despite the queue turns `test_a_message_sent_while_the_reply_works_steers_it_at_its_next_step`
red (the plan's next step is stored as the reply); the server answering the message as a normal
turn instead of holding it turns both tests red (nothing waits in the chat).
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.tool_calls import without_hidden_calls
from harness.upstream import conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PLAN_GOES_ON = "Next I will check the calendar as planned."


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def requests_for(upstream, prompt: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]


def wait_until_asked(upstream, prompt: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not requests_for(upstream, prompt):
        if time.monotonic() > deadline:
            raise AssertionError(f"the provider never received the request for {prompt!r}")
        time.sleep(0.05)


def messages_of(client, chat_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"]["history"]["messages"]


def waiting_under(messages: dict, reply_id: str) -> list[dict]:
    """The person's messages waiting under the reply, in the order they were sent."""
    children = [messages[child] for child in messages[reply_id].get("childrenIds") or []]
    return [
        message
        for message in children
        if message["role"] == "user" and (message.get("meta") or {}).get("status") == "pending"
    ]


def queue_while_running(client, chat_id: str, reply_id: str, text: str) -> str:
    """Send `text` as the web client does while the reply `reply_id` still runs."""
    turn = send_message(client, text, chat_id=chat_id, parent_id=reply_id, queue=True)
    return turn.user_message_id


def answer_after(client, chat_id: str, reply_id: str, timeout: float = 60.0) -> tuple[dict, dict]:
    """The question asked after `reply_id` and its finished answer."""
    deadline = time.monotonic() + timeout
    state: dict = {}
    while time.monotonic() < deadline:
        messages = messages_of(client, chat_id)
        asked = [messages[child] for child in messages[reply_id].get("childrenIds") or []]
        answered = [
            (question, messages[child])
            for question in asked
            if question["role"] == "user"
            for child in question.get("childrenIds") or []
        ]
        state = {"asked": asked, "answered": answered}
        if answered and answered[-1][1].get("done"):
            return answered[-1]
        time.sleep(0.1)
    raise AssertionError(f"the queued messages were never answered; last stored state: {state}")


def test_a_message_sent_while_the_reply_works_steers_it_at_its_next_step(make_user, upstream):
    person = make_user()
    work = unique("check the time, then the calendar, then report")
    first, second = unique("Use the Berlin time zone."), unique("And keep it short.")
    answer = unique("It is noon in Berlin.")
    released = threading.Event()
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(work), hold=released),
        # the reply's next step if nobody had written; it must never be asked for
        reply.text(PLAN_GOES_ON, match=reply.answering(work)),
        reply.text(answer, match=reply.answering(second)),
    )
    try:
        with person.client() as client:
            running = send_message(client, work)
            wait_until_asked(upstream, work)
            first_id = queue_while_running(
                client, running.chat_id, running.assistant_message_id, first
            )
            second_id = queue_while_running(
                client, running.chat_id, running.assistant_message_id, second
            )

        # a page loaded afresh reads the chat from the server
        with person.client() as reloaded:
            messages = messages_of(reloaded, running.chat_id)
        waiting = waiting_under(messages, running.assistant_message_id)
        assert [(m["id"], m["content"]) for m in waiting] == [
            (first_id, first),
            (second_id, second),
        ], f"the messages sent while the reply ran are not kept in the chat: {waiting}"
        assert messages[running.assistant_message_id].get("done") is not True
        assert not requests_for(upstream, first) and not requests_for(upstream, second), (
            "a message sent while the reply ran was answered before the reply wrapped up"
        )

        released.set()
        with person.client() as client:
            stopped = wait_for_reply(client, running)
            question, answered = answer_after(client, running.chat_id, running.assistant_message_id)
            messages = messages_of(client, running.chat_id)
    finally:
        released.set()

    done_so_far = without_hidden_calls(stopped["output"])
    kinds = [item["type"] for item in done_so_far]
    assert "function_call" in kinds and "function_call_output" in kinds, (
        f"the reply lost the tool call it made before it wrapped up: {kinds}"
    )
    assert PLAN_GOES_ON not in str(stopped.get("content")), (
        f"the reply went on with its plan despite the queue: {stopped.get('content')!r}"
    )
    went_on = [
        body
        for body in requests_for(upstream, work)
        if any(entry.get("role") == "tool" for entry in conversation(body))
    ]
    assert went_on == [], "the reply asked the model for its next step instead of wrapping up"

    assert question["content"] == f"{first}\n\n{second}", (
        f"the queued messages were not answered as one question, in order: {question['content']!r}"
    )
    assert (question.get("meta") or {}).get("status") != "pending"
    assert answered["content"] == answer, f"the queued question's answer: {answered['content']!r}"
    assert waiting_under(messages, running.assistant_message_id) == [], (
        "the answered messages still wait in the queue"
    )

    [request] = requests_for(upstream, second)
    sent = conversation(request)
    assert sent[-1]["role"] == "user" and sent[-1]["content"] == question["content"], sent[-1]
    calls = [call["function"]["name"] for entry in sent for call in entry.get("tool_calls") or []]
    results = [entry for entry in sent if entry.get("role") == "tool"]
    assert "get_current_timestamp" in calls and results, (
        f"the queued question was answered without what the reply had done so far: {sent}"
    )
    assert any(work in str(entry.get("content")) for entry in sent if entry["role"] == "user")


def test_a_queued_message_deleted_in_time_is_never_answered_and_the_reply_goes_on(
    make_user, upstream
):
    person = make_user()
    work = unique("check the time, then the calendar, then report")
    taken_back = unique("Actually use the Vienna time zone.")
    released = threading.Event()
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(work), hold=released),
        reply.text(PLAN_GOES_ON, match=reply.answering(work)),
        reply.text("Answered the deleted message.", match=reply.answering(taken_back)),
    )
    try:
        with person.client() as client:
            running = send_message(client, work)
            wait_until_asked(upstream, work)
            queued_id = queue_while_running(
                client, running.chat_id, running.assistant_message_id, taken_back
            )
            assert [
                m["id"]
                for m in waiting_under(
                    messages_of(client, running.chat_id), running.assistant_message_id
                )
            ] == [queued_id], "the message sent while the reply ran was not queued"
            # the queue row's Delete button
            deleted = client.delete(f"/api/v1/chats/{running.chat_id}/messages/{queued_id}")
            assert deleted.status_code == 200, deleted.text
            released.set()
            finished = wait_for_reply(client, running)
            time.sleep(1.0)  # bounded: nothing may be answered after the reply ends
            messages = messages_of(client, running.chat_id)
    finally:
        released.set()

    assert finished["content"] == PLAN_GOES_ON, (
        f"the reply did not go on once the queued message was deleted: {finished['content']!r}"
    )
    assert queued_id not in messages, "the deleted message is still in the chat"
    assert messages[running.assistant_message_id].get("childrenIds") in ([], None), (
        "something was answered after the reply although the queued message was deleted"
    )
    assert not requests_for(upstream, taken_back), "the deleted message was sent to the model"
