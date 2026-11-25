"""Journey: an agent's message to another chat reaches only the user's own chats, and arrives whole.

An agent sends a message to another of the person's chats with the `send_message` builtin tool,
which comes with sub-agents. The other chat wakes up and answers it, and a message shows which chat
it came from. A chat the person cannot reach is refused alike for every reason (another account's
chat, a deleted one, one that never existed, the sending chat itself), so the refusal tells the
agent nothing about whose chat it named, and nothing arrives there. Several messages sent while
the other chat is working wait for it and reach its model together, in the order sent. A message
whose answer fails leaves the chat usable and is not answered a second time. Deleting a chat stops
its running sub-agents and removes their threads. While sub-agents are off for the instance or the
account lacks the Sub-agents permission, the model is offered no `send_message`, and a call made
anyway reaches nothing.

Discriminates: passes on dev 9b626e4e5, two runs in a row. In backend copies of it the refusal
test turns red with the owner check on the target chat removed (another account's chat gets the
message) and the delete test with the delete of a chat no longer reaching its sub-agents (the
sub-agent runs on).
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply
from harness.tool_calls import without_hidden_calls

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
AGENT_TOOLS = {"send_message", "delegate_task"}


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


@pytest.fixture
def sub_agents(admin, preserve):
    """`sub_agents(True)` switches sub-agents on or off for the instance."""
    preserve(SUBAGENTS)

    def switch(enabled: bool) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": enabled})
        assert saved.status_code == 200, saved.text

    return switch


def send_to(chat_id: str, text: str, prompt: str, call_id: str = "call_1") -> reply.Reply:
    return reply.tool_call(
        "send_message",
        {"chat_id": chat_id, "message": text},
        call_id,
        match=reply.answering(prompt),
    )


def tool_results(message: dict) -> list[str]:
    return [
        item["output"][0]["text"]
        for item in without_hidden_calls(message["output"])
        if item["type"] == "function_call_output"
    ]


def history_of(client, chat_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"]["history"]


def wait_for(condition, what: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = condition()
        if found:
            return found
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def received(history: dict) -> list[dict]:
    """The messages another chat sent here, oldest first."""
    rows = [
        message
        for message in history["messages"].values()
        if (message.get("meta") or {}).get("type") == "message"
    ]
    return sorted(rows, key=lambda message: message["timestamp"])


def user_texts_mentioning(upstream, marker: str) -> list[str]:
    """Every user message the provider was sent that carries `marker`."""
    return [
        str(entry.get("content"))
        for body in upstream.chat_requests()
        for entry in body.get("messages", [])
        if entry.get("role") == "user" and marker in str(entry.get("content"))
    ]


def open_chat(client, upstream, text: str) -> tuple[str, str]:
    """A chat with one finished turn, as `(chat_id, reply_id)`."""
    upstream.queue(reply.text(f"Noted {text}.", match=reply.answering(text)))
    turn, stored = ask(client, text)
    assert stored["content"] == f"Noted {text}.", stored
    return turn.chat_id, turn.assistant_message_id


def test_a_message_reaches_only_the_users_own_chats_and_a_refusal_names_no_reason(
    sub_agents, make_user, upstream
):
    sub_agents(True)
    owner, stranger = make_user(), make_user()
    with stranger.client() as client:
        foreign_id, _ = open_chat(client, upstream, unique("the stranger's own secret"))
        foreign_before = history_of(client, foreign_id)

    with owner.client() as client:
        deleted_id, _ = open_chat(client, upstream, unique("soon gone"))
        assert client.delete(f"/api/v1/chats/{deleted_id}").status_code == 200
        own_id, _ = open_chat(client, upstream, unique("the owner's other chat"))

        self_id, self_reply = open_chat(client, upstream, unique("the sending chat"))
        targets = {
            "another account's chat": foreign_id,
            "a deleted chat": deleted_id,
            "a chat that never existed": str(uuid.uuid4()),
            "the sending chat itself": self_id,
        }
        results = {}
        for label, target in targets.items():
            prompt, text = unique("pass this on"), unique("note for the target")
            upstream.queue(
                send_to(target, text, prompt),
                reply.text("Tried.", match=reply.answering(prompt)),
            )
            if target == self_id:
                turn = send_message(client, prompt, chat_id=self_id, parent_id=self_reply)
                stored = wait_for_reply(client, turn)
            else:
                _, stored = ask(client, prompt)
            [results[label]] = tool_results(stored)
            assert user_texts_mentioning(upstream, text) == [], (
                f"a message to {label} reached a model: {user_texts_mentioning(upstream, text)}"
            )

        prompt, text = unique("pass this on"), unique("note for the owner's chat")
        upstream.queue(
            send_to(own_id, text, prompt),
            reply.text("Sent on.", match=reply.answering(prompt)),
            reply.text("Got the note.", match=reply.answering(text)),
        )
        sending, stored = ask(client, prompt)
        wait_for(
            lambda: [
                m
                for m in history_of(client, own_id)["messages"].values()
                if m.get("content") == "Got the note." and m.get("done")
            ],
            "the owner's other chat to answer the message",
        )
        [row] = received(history_of(client, own_id))

    assert all(result.startswith("Error:") for result in results.values()), results
    assert len(set(results.values())) == 1, (
        f"the refusal differs by reason, so it tells whose chat was named: {results}"
    )
    assert foreign_before == history_of_as(stranger, foreign_id), (
        "the other account's chat changed after the refused message"
    )
    assert not tool_results(stored)[0].startswith("Error:"), tool_results(stored)
    assert row["meta"]["internal"] is True and text in row["content"]
    [delivered] = user_texts_mentioning(upstream, text)
    assert sending.chat_id in delivered, (
        f"the message reached the other chat without naming the chat it came from: {delivered}"
    )


def history_of_as(actor, chat_id: str) -> dict:
    with actor.client() as client:
        return history_of(client, chat_id)


def test_messages_sent_while_the_chat_works_reach_it_together_and_in_order(
    sub_agents, make_user, upstream
):
    sub_agents(True)
    busy_prompt, prompt = unique("think this through"), unique("send three notes")
    texts = [unique("first note"), unique("second note"), unique("third note")]
    release = threading.Event()
    upstream.queue(reply.text("Thought through.", hold=release, match=reply.answering(busy_prompt)))
    try:
        with make_user().client() as client:
            busy = send_message(client, busy_prompt)
            # one call per round, so the three are sent one after the other
            upstream.queue(
                *[send_to(busy.chat_id, text, prompt, f"call_{n}") for n, text in enumerate(texts)],
                reply.text("Three sent.", match=reply.answering(prompt)),
                reply.text("Read all three.", match=reply.answering(texts[-1])),
            )
            wait_for(
                lambda: any(reply.answering(busy_prompt)(b) for b in upstream.chat_requests()),
                "the busy chat's reply to start",
            )
            _, stored = ask(client, prompt)
            assert tool_results(stored) == ["Sent."] * 3, tool_results(stored)
            assert user_texts_mentioning(upstream, texts[0]) == [], (
                "a message reached the chat's model while it was still writing its reply"
            )
            release.set()
            wait_for_reply(client, busy)
            wait_for(
                lambda: [
                    m
                    for m in history_of(client, busy.chat_id)["messages"].values()
                    if m.get("content") == "Read all three." and m.get("done")
                ],
                "the busy chat to answer the messages",
            )
            history = history_of(client, busy.chat_id)
    finally:
        release.set()

    answered = user_texts_mentioning(upstream, texts[0])
    assert len(answered) == 1, f"the messages were answered more than once: {answered}"
    positions = [answered[0].find(text) for text in texts]
    assert -1 not in positions, f"a message was lost on the way to the model: {answered[0]}"
    assert positions == sorted(positions), f"the messages arrived out of order: {answered[0]}"
    shown, cursor = [], history["currentId"]
    while cursor:
        shown.append(history["messages"][cursor]["content"])
        cursor = history["messages"][cursor]["parentId"]
    shown_text = "\n".join(reversed(shown))
    assert [shown_text.find(text) for text in texts] == sorted(
        shown_text.find(text) for text in texts
    ), "the chat shows the messages out of order"
    assert all(text in shown_text for text in texts), "the chat does not show every message"


def test_a_message_whose_answer_fails_leaves_the_chat_usable_and_is_not_answered_again(
    sub_agents, make_user, upstream
):
    sub_agents(True)
    with make_user().client() as client:
        target_id, _ = open_chat(client, upstream, unique("standing by"))
        prompt, text = unique("pass this on"), unique("note that fails")
        upstream.queue(
            send_to(target_id, text, prompt),
            reply.text("Sent on.", match=reply.answering(prompt)),
            reply.error(500, "the model is down", match=reply.answering(text)),
        )
        _, stored = ask(client, prompt)
        assert tool_results(stored) == ["Sent."], tool_results(stored)

        failed = wait_for(
            lambda: [
                m
                for m in history_of(client, target_id)["messages"].values()
                if m["role"] == "assistant" and m.get("done") and m.get("error")
            ],
            "the failed answer to be stored",
        )[0]
        assert "the model is down" in json.dumps(failed["error"]), failed["error"]

        later = unique("are you back?")
        upstream.queue(reply.text("Back again.", match=reply.answering(later)))
        history = history_of(client, target_id)
        again = send_message(client, later, chat_id=target_id, parent_id=history["currentId"])
        answered = wait_for_reply(client, again)
        history = history_of(client, target_id)

    assert answered["content"] == "Back again."
    assert len(received(history)) == 1, received(history)
    answers = [b for b in upstream.chat_requests() if reply.answering(text)(b)]
    assert len(answers) == 1, f"the failed message was answered {len(answers)} times"


def test_deleting_a_chat_stops_its_running_sub_agent_and_removes_its_thread(
    sub_agents, make_user, upstream
):
    # on Postgres with Redis the delete left the chat in place (issue #35034)
    sub_agents(True)
    prompt, task = unique("hand this over"), unique("count the crates")
    release = threading.Event()
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text("never reported", hold=release, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
    )
    try:
        with make_user().client() as client:
            turn, stored = ask(client, prompt)
            [handle] = [json.loads(text) for text in tool_results(stored)]
            sub_id = handle["subagent_chat_id"]
            wait_for(
                lambda: any(reply.answering(task)(b) for b in upstream.chat_requests()),
                "the sub-agent to start",
            )
            threads = client.get(f"/api/v1/chats/{turn.chat_id}/children").json()
            assert [thread["id"] for thread in threads] == [sub_id], threads

            assert client.delete(f"/api/v1/chats/{turn.chat_id}").status_code == 200
            # still held, so only the delete can end the sub-agent's task
            wait_for(
                lambda: client.get(f"/api/tasks/chat/{sub_id}").json()["task_ids"] == [],
                "the deleted chat's sub-agent to stop",
                timeout=20.0,
            )
            thread = client.get(f"/api/v1/chats/{sub_id}")
            listed = [chat["id"] for chat in client.get("/api/v1/chats/").json()]
    finally:
        release.set()

    # a chat that is gone reads as not found, with 401 or 404 depending on the route
    assert thread.status_code in (401, 404), f"the thread is still there: {thread.text}"
    assert "find" in thread.text, thread.text
    assert listed == [], f"chats are left after deleting the only one: {listed}"
    assert not any("[ASYNC SUBAGENT COMPLETE" in t for t in user_texts_mentioning(upstream, task))


@pytest.fixture
def without_permission(sub_agents, admin, preserve):
    """Sub-agents on for the instance, the Sub-agents permission off in the defaults."""
    sub_agents(True)
    preserve("permissions")
    with admin.client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        features = {**current["features"], "subagents": False}
        saved = client.post(DEFAULT_PERMISSIONS, json={**current, "features": features})
    assert saved.status_code == 200, saved.text


@pytest.mark.parametrize("withheld_by", ["the instance switch", "the account's permission"])
def test_without_sub_agents_a_message_call_reaches_nothing_and_starts_no_thread(
    withheld_by, sub_agents, request, make_user, upstream
):
    if withheld_by == "the instance switch":
        sub_agents(False)
    else:
        request.getfixturevalue("without_permission")
    with make_user().client() as client:
        target_id, _ = open_chat(client, upstream, unique("standing by"))
        before = history_of(client, target_id)
        prompt, text, task = unique("pass this on"), unique("stray note"), unique("stray task")
        calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "send_message",
                    "arguments": json.dumps({"chat_id": target_id, "message": text}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "delegate_task", "arguments": json.dumps({"task": task})},
            },
        ]
        upstream.queue(
            reply.Reply(tool_calls=calls, match=reply.answering(prompt)),
            reply.text("Tried.", match=reply.answering(prompt)),
        )
        turn, stored = ask(client, prompt)
        offered = {
            tool["function"]["name"]
            for tool in next(b for b in upstream.chat_requests() if reply.answering(prompt)(b))[
                "tools"
            ]
            or []
        }
        threads = client.get(f"/api/v1/chats/{turn.chat_id}/children")
        after = history_of(client, target_id)

    assert offered, "the model was offered no tools at all, so this shows nothing"
    assert not AGENT_TOOLS & offered, f"with sub-agents withheld by {withheld_by}: {offered}"
    assert stored["content"] == "Tried."
    assert after == before, f"the message reached the other chat although {withheld_by} is off"
    assert user_texts_mentioning(upstream, text) == []
    assert not any(reply.answering(task)(b) for b in upstream.chat_requests())
    assert threads.status_code == 200 and threads.json() == [], threads.text
