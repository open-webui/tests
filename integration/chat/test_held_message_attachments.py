"""Regression: a message sent while a reply ran was answered without the chat's attachments.

Since 0.13.4 a message sent while the reply before it still runs is held and answered once that
reply ends. It should be answered with the attachments a normal follow-up at that point gets: the
files on the messages of its branch that the chat still holds, plus its own. The held answer got
only its own files, so with Preload files ticked the file's text never reached the model and the
answer had no source, with it off the model was not offered the attachment tools, and the chat's
stored file list was emptied (open-webui/open-webui#34933). Each narrow test sends the same
question afterwards as a normal follow-up as well and expects the same. The nearby tests pin a
file on the held message itself, a file the person removed from the chat's files while the reply
ran, which stays out, and a file attached to the held message again, which is sent and kept once.

Discriminates: fails on dev c94eb96b7 (the held answer gets no file text, no source and no
attachment tools, in both modes and wherever in the branch the file is); passes with the fix
b19a727a1 (#34933). The nearby tests pass on both; the removed-file test fails when the held
answer simply takes the running reply's files, and the attached-again test when the chat's files
are added without dropping the held message's own.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask, send_message, wait_for_reply
from harness.model_presets import preset_with_capabilities
from harness.upstream import conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ATTACHMENT_TOOLS = {"list_chat_files", "query_chat_files", "view_file"}


def upload(client, name: str, text: str) -> dict:
    """Upload a file as the chat input does and return the item the client attaches."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = uploaded.json()
    return {
        "type": "file",
        "id": stored["id"],
        "url": f"/api/v1/files/{stored['id']}",
        "name": name,
        "content_type": "text/plain",
        "size": len(text),
        "status": "uploaded",
    }


def letter(secret: str) -> str:
    return "Dear tenant,\n" + "The heating works as before.\n" * 20 + f"{secret}\n" + "Regards\n"


def requests_for(upstream, question: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(question)(body)]


def wait_until_asked(upstream, prompt: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not requests_for(upstream, prompt):
        if time.monotonic() > deadline:
            raise AssertionError(f"the provider never received the request for {prompt!r}")
        time.sleep(0.05)


def answer_after(client, chat_id: str, reply_id: str, timeout: float = 60.0) -> dict:
    """The finished answer to the message held after `reply_id`."""
    deadline = time.monotonic() + timeout
    answer: dict = {}
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        asked = [messages[child] for child in messages[reply_id].get("childrenIds") or []]
        answers = [
            messages[child] for message in asked for child in message.get("childrenIds") or []
        ]
        answer = answers[-1] if answers else {}
        if answer.get("done"):
            return answer
        time.sleep(0.1)
    raise AssertionError(f"the held message was never answered; last stored answer: {answer}")


def scripted_answer(file_context: bool, file_id: str, question: str, answer: str) -> list:
    """With Preload files ticked the model answers; off, it first reads the file with its tool."""
    answering = reply.answering(question)
    if file_context:
        return [reply.text(answer, match=answering)]
    return [
        reply.tool_call("view_file", {"file_id": file_id}, match=answering),
        reply.text(answer, match=answering),
    ]


def assert_answered_with_the_file(
    requests: list[dict], answer: dict, file_context: bool, secret: str, what: str
) -> None:
    assert requests, f"the provider never received the {what}"
    first = requests[0]
    if file_context:
        assert secret in json.dumps(conversation(first)), (
            f"the {what} was answered without the attached file's text: {conversation(first)}"
        )
        assert secret in json.dumps(answer.get("sources") or []), (
            f"the answer to the {what} cites nothing from the file: {answer.get('sources')}"
        )
    else:
        offered = {tool["function"]["name"] for tool in first.get("tools") or []}
        assert ATTACHMENT_TOOLS <= offered, (
            f"the {what} was not offered the attachment tools, only {sorted(offered)}"
        )
        results = [
            entry["content"] for entry in requests[-1]["messages"] if entry["role"] == "tool"
        ]
        assert any(secret in str(result) for result in results), (
            f"reading the attached file for the {what} did not return its text: {results}"
        )


@contextlib.contextmanager
def held_question(
    person: Actor,
    upstream,
    question: str,
    *,
    earlier_files: list[dict] | None = None,
    running_files: list[dict] | None = None,
    held_files: list[dict] | None = None,
    while_running=None,
    model: str,
):
    """Start a chat, send `question` while its reply runs and yield what answered it."""
    released = threading.Event()
    story = f"write a long story about a lighthouse keeper {uuid.uuid4().hex[:8]}"
    upstream.queue(
        reply.text(
            ["Once upon a time ", "the keeper lit the lamp."],
            hold=released,
            hold_after=1,
            match=reply.answering(story),
        )
    )
    try:
        with person.client() as client:
            parent_id, chat_id = None, None
            if earlier_files:
                intro = f"here is my letter {uuid.uuid4().hex[:8]}"
                upstream.queue(reply.text("Thanks.", match=reply.answering(intro)))
                first, _ = ask(client, intro, model=model, chat_files=earlier_files)
                parent_id, chat_id = first.assistant_message_id, first.chat_id
            chat_files = [*(earlier_files or []), *(running_files or [])]
            running = send_message(
                client,
                story,
                model=model,
                chat_id=chat_id,
                parent_id=parent_id,
                message_files=running_files,
                **({"files": chat_files} if chat_files else {}),
            )
            wait_until_asked(upstream, story)
            if while_running:
                chat_files = while_running(client, running.chat_id)
            # the web client asks to queue a message sent while a reply runs, with the chat's files
            send_message(
                client,
                question,
                model=model,
                chat_id=running.chat_id,
                parent_id=running.assistant_message_id,
                message_files=held_files,
                files=[*chat_files, *(held_files or [])],
                queue=True,
            )
            time.sleep(0.5)
            answered_too_soon = requests_for(upstream, question)
            released.set()
            wait_for_reply(client, running)
            answer = answer_after(client, running.chat_id, running.assistant_message_id)
            stored_files = (
                client.get(f"/api/v1/chats/{running.chat_id}").json()["chat"].get("files")
            )
        assert not answered_too_soon, "the message sent while the reply ran was not held"
        yield {
            "chat_id": running.chat_id,
            "answer": answer,
            "requests": requests_for(upstream, question),
            "stored_files": stored_files or [],
        }
    finally:
        released.set()


def follow_up(person: Actor, upstream, chat_id: str, after: dict, question: str, files, model: str):
    """Ask `question` as a normal follow-up after the stored reply `after`."""
    with person.client() as client:
        turn = send_message(
            client, question, model=model, chat_id=chat_id, parent_id=after["id"], files=files
        )
        answer = wait_for_reply(client, turn)
    return answer, requests_for(upstream, question)


MODES = pytest.mark.parametrize("file_context", [True, False], ids=["preload_on", "preload_off"])
WHERE = pytest.mark.parametrize("where", ["running_turn", "earlier_turn"])


@MODES
@WHERE
def test_a_held_message_is_answered_with_the_chats_attachments(
    file_context, where, admin, make_user, upstream
):
    person = make_user()
    secret = f"Kostenbeitrag: {uuid.uuid4().hex[:6]} Euro"
    with person.client() as client:
        attachment = upload(client, "letter.txt", letter(secret))
    question = f"how much is the Kostenbeitrag in the letter? {uuid.uuid4().hex[:8]}"
    again = f"once more, how much is the Kostenbeitrag? {uuid.uuid4().hex[:8]}"
    upstream.queue(*scripted_answer(file_context, attachment["id"], question, secret))
    upstream.queue(*scripted_answer(file_context, attachment["id"], again, secret))
    files = {"earlier_files" if where == "earlier_turn" else "running_files": [attachment]}
    capabilities = {"file_context": True} if file_context else {}
    with preset_with_capabilities(admin, capabilities) as model:
        with held_question(person, upstream, question, model=model, **files) as held:
            pass
        normal_answer, normal_requests = follow_up(
            person, upstream, held["chat_id"], held["answer"], again, [attachment], model
        )

    assert_answered_with_the_file(
        normal_requests, normal_answer, file_context, secret, "normal follow-up"
    )
    assert_answered_with_the_file(
        held["requests"], held["answer"], file_context, secret, "held message"
    )
    assert attachment["id"] in [file.get("id") for file in held["stored_files"]], (
        f"answering the held message dropped the file from the chat's files: {held['stored_files']}"
    )


def test_a_held_message_is_answered_with_its_own_attachment(admin, make_user, upstream):
    person = make_user()
    secret = f"Kostenbeitrag: {uuid.uuid4().hex[:6]} Euro"
    with person.client() as client:
        attachment = upload(client, "letter.txt", letter(secret))
    question = f"how much is the Kostenbeitrag in this letter? {uuid.uuid4().hex[:8]}"
    upstream.queue(*scripted_answer(True, attachment["id"], question, secret))
    with preset_with_capabilities(admin, {"file_context": True}) as model:
        with held_question(
            person, upstream, question, model=model, held_files=[attachment]
        ) as held:
            pass

    assert_answered_with_the_file(held["requests"], held["answer"], True, secret, "held message")


def test_a_file_removed_from_the_chat_while_the_reply_runs_stays_out(admin, make_user, upstream):
    person = make_user()
    secret = f"Kostenbeitrag: {uuid.uuid4().hex[:6]} Euro"
    with person.client() as client:
        attachment = upload(client, "letter.txt", letter(secret))
    question = f"anything else to note? {uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("Nothing else.", match=reply.answering(question)))

    def remove_the_file(client, chat_id: str) -> list[dict]:
        # the chat's file list, as the Controls panel saves it after the file's remove button
        removed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"files": []}})
        assert removed.status_code == 200, removed.text
        return []

    with preset_with_capabilities(admin, {"file_context": True}) as model:
        with held_question(
            person,
            upstream,
            question,
            model=model,
            running_files=[attachment],
            while_running=remove_the_file,
        ) as held:
            pass

    [request] = held["requests"]
    assert secret not in json.dumps(conversation(request)), (
        "the held message was answered with a file removed from the chat while the reply ran"
    )
    assert not held["answer"].get("sources"), f"the answer cites a removed file: {held['answer']}"


def test_a_file_attached_again_to_the_held_message_is_sent_once(admin, make_user, upstream):
    person = make_user()
    secret = f"Kostenbeitrag: {uuid.uuid4().hex[:6]} Euro"
    with person.client() as client:
        attachment = upload(client, "letter.txt", letter(secret))
    question = f"look at this letter again, what is the amount? {uuid.uuid4().hex[:8]}"
    upstream.queue(*scripted_answer(True, attachment["id"], question, secret))
    with preset_with_capabilities(admin, {"file_context": True}) as model:
        with held_question(
            person,
            upstream,
            question,
            model=model,
            running_files=[attachment],
            held_files=[attachment],
        ) as held:
            pass

    [request] = held["requests"]
    assert json.dumps(conversation(request)).count(secret) == 1, (
        f"the file attached twice was sent to the model more than once: {conversation(request)}"
    )
    stored = [file for file in held["stored_files"] if file.get("id") == attachment["id"]]
    assert len(stored) == 1, f"the chat's files hold the file {len(stored)} times"
