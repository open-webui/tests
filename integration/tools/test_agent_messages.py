"""Journey: agents message each other's chats with the send_message tool.

An agent sends a message to another of the person's chats, to one of its sub-agents or, from a
sub-agent, to the lead agent. A chat that is idle wakes up and answers it on its own; a chat
that is working reads it right after its current tool call. The message is stored in the
receiving chat as a message the person did not write, naming the chat it came from by id and
title, and the answer an agent sends back lands in the chat that asked, after its latest reply.
A chat of another account cannot be reached.

Discriminates: passes on dev 9b626e4e5. In backend copies: the message stored without the
sender's chat id and title turns the four delivery tests red; the reply going on after its tool
call without the queued message turns
`test_a_working_chat_reads_a_message_right_after_its_current_tool_call` red; the owner check
dropped turns `test_a_chat_of_another_account_cannot_be_messaged` red.
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


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True})
    assert saved.status_code == 200, saved.text


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def message_to(chat_id: str, text: str, prompt: str, **options) -> reply.Reply:
    return reply.tool_call(
        "send_message",
        {"chat_id": chat_id, "message": text},
        match=reply.answering(prompt),
        **options,
    )


def report_of(task: str):
    """A `match` for the request that carries the finished sub-agent's report to the lead."""

    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def history_of(client, chat_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"]["history"]


def wait_for_message(client, chat_id: str, condition, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for message in history_of(client, chat_id)["messages"].values():
            if condition(message):
                return message
        time.sleep(0.1)
    raise AssertionError(f"no stored message of chat {chat_id} ever satisfied the condition")


def wait_for_request(upstream, condition, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for body in upstream.chat_requests():
            if condition(body):
                return body
        time.sleep(0.1)
    raise AssertionError("the model never got the request")


def finished_reply(text: str):
    return lambda message: (
        message["role"] == "assistant" and message.get("done") and message["content"] == text
    )


def received(text: str):
    return lambda message: message["role"] == "user" and text in str(message["content"])


def tool_outputs(message: dict) -> list[str]:
    return [
        item["output"][0]["text"]
        for item in without_hidden_calls(message["output"])
        if item["type"] == "function_call_output"
    ]


def rename(client, chat_id: str, title: str) -> None:
    renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert renamed.status_code == 200, renamed.text


def assert_shows_sender(message: dict, text: str, sender_id: str, sender_title: str) -> None:
    content = str(message["content"])
    assert text in content, f"the received message lost its text: {content!r}"
    assert sender_id in content and sender_title in content, (
        f"the received message does not say it came from chat {sender_title!r}: {content!r}"
    )
    assert (message.get("meta") or {}).get("internal") is True, (
        f"the message shows as written by the person: {message.get('meta')}"
    )


def shown_branch(history: dict) -> list[dict]:
    shown, cursor = [], history["currentId"]
    while cursor:
        shown.append(history["messages"][cursor])
        cursor = history["messages"][cursor]["parentId"]
    return list(reversed(shown))


def test_a_message_wakes_an_idle_chat_and_its_answer_comes_back_to_the_asking_chat(
    subagents_on, make_user, upstream
):
    asker_title, helper_title = unique("Demo planning"), unique("Demo calendar")
    ask_prompt, open_prompt = unique("ask the calendar chat"), unique("keep the calendar")
    question, answer = unique("When is the review?"), unique("The review is on Friday.")
    helper_turn_held = threading.Event()
    with make_user().client() as client:
        upstream.queue(reply.text("Calendar ready.", match=reply.answering(open_prompt)))
        helper, _ = ask(client, open_prompt)
        rename(client, helper.chat_id, helper_title)
        upstream.queue(reply.text("Planning ready.", match=reply.answering(asker_title)))
        asker, _ = ask(client, asker_title)
        rename(client, asker.chat_id, asker_title)

        upstream.queue(
            message_to(helper.chat_id, question, ask_prompt),
            reply.text("Asked the calendar.", match=reply.answering(ask_prompt)),
            # held so the asking chat is idle again when the answer comes back
            message_to(asker.chat_id, answer, question, hold=helper_turn_held),
            reply.text("Answered.", match=reply.answering(question)),
            reply.text("Friday it is.", match=reply.answering(answer)),
        )
        turn = send_message(
            client, ask_prompt, chat_id=asker.chat_id, parent_id=asker.assistant_message_id
        )
        asked = wait_for_reply(client, turn)
        assert [o for o in tool_outputs(asked) if o.startswith("Error")] == []
        assert asked["content"] == "Asked the calendar."

        question_row = wait_for_message(client, helper.chat_id, received(question))
        assert_shows_sender(question_row, question, asker.chat_id, asker_title)
        assert question_row["parentId"] == helper.assistant_message_id
        woken = wait_for_request(upstream, reply.answering(question))
        assert asker.chat_id in reply.conversation(woken)[-1]["content"]
        helper_turn_held.set()

        wait_for_message(client, helper.chat_id, finished_reply("Answered."))
        answer_row = wait_for_message(client, asker.chat_id, received(answer))
        assert_shows_sender(answer_row, answer, helper.chat_id, helper_title)
        assert answer_row["parentId"] == turn.assistant_message_id, (
            "the answer did not come back after the asking chat's latest reply"
        )
        wait_for_message(client, asker.chat_id, finished_reply("Friday it is."))
        shown = shown_branch(history_of(client, asker.chat_id))

    assert [m["content"] for m in shown[-2:]] == [answer_row["content"], "Friday it is."], (
        "the asking chat does not show the answer and its reply as its latest messages"
    )


def test_a_working_chat_reads_a_message_right_after_its_current_tool_call(
    subagents_on, make_user, upstream
):
    sender_title, work_prompt = unique("Demo sender"), unique("check the time and report")
    send_prompt, note = unique("tell the worker"), unique("Use the Berlin time zone.")
    worker_held = threading.Event()
    with make_user().client() as client:
        upstream.queue(reply.text("Sender ready.", match=reply.answering(sender_title)))
        sender, _ = ask(client, sender_title)
        rename(client, sender.chat_id, sender_title)

        upstream.queue(
            reply.tool_call(
                "get_current_timestamp", {}, match=reply.answering(work_prompt), hold=worker_held
            ),
            reply.text("Read the note.", match=reply.answering(note)),
            reply.text("Finished without the note.", match=reply.answering(work_prompt)),
        )
        worker = send_message(client, work_prompt)
        wait_for_request(upstream, reply.answering(work_prompt))
        # the worker's chat id is known only now, so the sender's call is queued with it here
        upstream.queue(
            message_to(worker.chat_id, note, send_prompt),
            reply.text("Told the worker.", match=reply.answering(send_prompt)),
        )
        turn = send_message(
            client, send_prompt, chat_id=sender.chat_id, parent_id=sender.assistant_message_id
        )
        sent = wait_for_reply(client, turn)
        assert [o for o in tool_outputs(sent) if o.startswith("Error")] == []
        worker_held.set()

        wait_for_message(client, worker.chat_id, finished_reply("Read the note."))
        note_row = wait_for_message(client, worker.chat_id, received(note))
        history = history_of(client, worker.chat_id)

    assert_shows_sender(note_row, note, sender.chat_id, sender_title)
    after_tool = [
        body
        for body in upstream.chat_requests()
        if any(entry.get("role") == "tool" for entry in reply.conversation(body))
        and (reply.answering(work_prompt)(body) or reply.answering(note)(body))
    ]
    assert after_tool, "the worker never continued after its tool call"
    assert all(reply.answering(note)(body) for body in after_tool), (
        "the worker went on after its tool call without the message it was sent"
    )
    contents = [m["content"] for m in shown_branch(history)]
    assert "Finished without the note." not in contents
    assert contents[-1] == "Read the note."


def dispatch_handle(message: dict) -> dict:
    [handle] = [json.loads(text) for text in tool_outputs(message) if text.startswith("{")]
    return handle


def test_the_lead_messages_its_working_subagent_and_the_report_carries_the_answer(
    subagents_on, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("count the open invoices")
    forward, note = unique("tell the helper"), unique("Only count invoices from 2024.")
    sub_answer = unique("There are 7 open invoices from 2024.")
    sub_held = threading.Event()
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(task), hold=sub_held),
        reply.text(sub_answer, match=reply.answering(note)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("Seven invoices.", match=report_of(task)),
    )
    with make_user().client() as client:
        lead, stored = ask(client, prompt)
        sub_id = dispatch_handle(stored)["subagent_chat_id"]
        upstream.queue(
            message_to(sub_id, note, forward),
            reply.text("Passed on.", match=reply.answering(forward)),
        )
        turn = send_message(
            client, forward, chat_id=lead.chat_id, parent_id=lead.assistant_message_id
        )
        sent = wait_for_reply(client, turn)
        assert [o for o in tool_outputs(sent) if o.startswith("Error")] == []
        lead_title = client.get(f"/api/v1/chats/{lead.chat_id}").json()["title"]
        sub_held.set()

        note_row = wait_for_message(client, sub_id, received(note))
        wait_for_message(client, lead.chat_id, finished_reply("Seven invoices."))
        [report] = [
            m
            for m in history_of(client, lead.chat_id)["messages"].values()
            if m["role"] == "user" and "[ASYNC SUBAGENT COMPLETE" in str(m["content"])
        ]

    assert_shows_sender(note_row, note, lead.chat_id, lead_title)
    read = wait_for_request(upstream, reply.answering(note))
    assert any(entry.get("role") == "tool" for entry in reply.conversation(read)), (
        "the sub-agent got the message only after it had finished, not after its tool call"
    )
    assert sub_answer in report["content"], (
        f"the report to the lead lacks the answer given after the message: {report['content']!r}"
    )


def test_a_subagent_asks_the_idle_lead_and_its_report_still_follows(
    subagents_on, make_user, upstream
):
    start, prompt = unique("start the lead"), unique("hand this over")
    task, question = unique("draft the agenda"), unique("Should the agenda include the budget?")
    # named apart from the task, so the lead's request for the question does not answer the task
    sub_title = unique("Demo agenda")
    sub_answer = unique("The agenda is drafted.")
    lead_idle = threading.Event()
    with make_user().client() as client:
        upstream.queue(reply.text("Ready.", match=reply.answering(start)))
        lead, _ = ask(client, start)
        upstream.queue(
            reply.tool_call(
                "delegate_task", {"task": task, "title": sub_title}, match=reply.answering(prompt)
            ),
            message_to(lead.chat_id, question, task, hold=lead_idle),
            reply.text(sub_answer, match=reply.answering(task)),
            reply.text("Handed over.", match=reply.answering(prompt)),
            reply.text("Yes, include the budget.", match=reply.answering(question)),
            reply.text("The agenda is in.", match=report_of(task)),
        )
        turn = send_message(
            client, prompt, chat_id=lead.chat_id, parent_id=lead.assistant_message_id
        )
        stored = wait_for_reply(client, turn)
        sub_id = dispatch_handle(stored)["subagent_chat_id"]
        briefed = wait_for_request(upstream, reply.answering(task))
        lead_idle.set()

        question_row = wait_for_message(client, lead.chat_id, received(question))
        wait_for_message(client, lead.chat_id, finished_reply("Yes, include the budget."))
        wait_for_message(client, lead.chat_id, finished_reply("The agenda is in."))
        sub_reply = wait_for_message(client, sub_id, finished_reply(sub_answer))

    system = " ".join(
        str(entry["content"]) for entry in briefed["messages"] if entry["role"] == "system"
    )
    assert lead.chat_id in system, "the sub-agent was never told which chat its lead is"
    assert_shows_sender(question_row, question, sub_id, sub_title)
    assert question_row["parentId"] == turn.assistant_message_id
    assert [o for o in tool_outputs(sub_reply) if o.startswith("Error")] == []


def test_a_chat_of_another_account_cannot_be_messaged(subagents_on, make_user, upstream):
    owner_prompt, prompt = unique("keep my chat"), unique("message that chat")
    note = unique("Hello from a stranger.")
    upstream.queue(reply.text("Kept.", match=reply.answering(owner_prompt)))
    with make_user().client() as owner_client:
        owned, _ = ask(owner_client, owner_prompt)
        upstream.queue(
            message_to(owned.chat_id, note, prompt),
            reply.text("Could not send.", match=reply.answering(prompt)),
            reply.text("A stranger wrote.", match=reply.answering(note)),
        )
        with make_user().client() as client:
            _, stored = ask(client, prompt)
        outputs = tool_outputs(stored)
        messages = history_of(owner_client, owned.chat_id)["messages"].values()

    assert len(outputs) == 1 and outputs[0].startswith("Error"), outputs
    assert [m for m in messages if note in str(m["content"])] == [], (
        "a message from another account reached the chat"
    )
    assert not any(reply.answering(note)(body) for body in upstream.chat_requests())
