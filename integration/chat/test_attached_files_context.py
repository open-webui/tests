"""Regression: attached files moved to the wrong message after compaction or a tool approval.

Commit af6b82a18 (issue open-webui/open-webui#31411). With native function calling each user
message sent to the model starts with an `<attached_files>` block naming the files uploaded with
it. The block was built by pairing the request's user messages with the stored chat's user
messages by position. Once context compaction dropped the oldest turns, the kept messages were
paired with the dropped ones, so a new upload was announced with an older message's file id or
not at all. The request sent after a tool approval was rebuilt from the database without the
block, so the model lost the file ids mid-turn. The block is now built from each message's own
files, on the first request and on the one after an approval.

Discriminates: passes on dev ef67cc3fa; with af6b82a18 reverted in a backend copy the compacting
turn, the turn after the checkpoint, both approval tests and the compaction-then-approval test
fail (a message lists another message's file, or none). The uncompacted, several-files and
own-tools cases pass on both. The compacting-turn test also fails on dev 97ca2144b, where the
compacting turn sent the whole chat plus the summary (open-webui/open-webui#33915), and passes
with the fix.
"""

from __future__ import annotations

import re
import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message, wait_for_reply
from harness.chat_history import seed_chat

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SUMMARY = "SUMMARY OF THE EARLIER TURNS"
APPROVAL_WAIT = 20.0
FILE_ID = re.compile(r'<file [^>]*\bid="([^"]+)"')


def attachment(name: str) -> dict:
    return {"type": "file", "id": f"file-{name}", "url": f"/api/v1/files/file-{name}", "name": name}


def turns(count: int, files: dict[int, list[dict]] | None = None) -> list[dict]:
    """`count` long question and answer pairs; `files` maps a question number to its uploads."""
    messages = []
    for number in range(1, count + 1):
        question = {"role": "user", "content": f"question {number} " + "q" * 400}
        if files and number in files:
            question["files"] = files[number]
        messages.append(question)
        messages.append({"role": "assistant", "content": f"answer {number} " + "a" * 400})
    return [{**message, "id": str(uuid.uuid4())} for message in messages]


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def listed_files(request: dict) -> dict[str, list[str]]:
    """Each user message the provider was sent, by its first words, with the file ids it lists."""
    listed = {}
    for message in request["messages"]:
        if message["role"] != "user":
            continue
        text = str(message["content"])
        question = re.search(r"question \d+", text)
        listed[question.group(0) if question else text[:20]] = FILE_ID.findall(text)
    return listed


def last_chat_call(upstream) -> dict:
    return [body for body in upstream.chat_requests() if body.get("stream")][-1]


@pytest.fixture
def chat_settings(admin, preserve):
    """`chat_settings(**settings)` saves those global chat settings for this test."""
    preserve(CHAT_CONFIG)

    def configure(**settings) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            client.post(CHAT_CONFIG[1], json={**current, **settings}).raise_for_status()

    return configure


def compaction(threshold: int = 50) -> dict:
    return {
        "ENABLE_CONTEXT_COMPACTION": True,
        "CONTEXT_COMPACTION_TOKEN_THRESHOLD": threshold,
        "CONTEXT_COMPACTION_TOKEN_CAP": threshold,
    }


def approve_every_call(client: httpx.Client, turn: ChatTurn) -> dict:
    """Approve the pending tool call once it waits, then return the finished reply."""
    deadline = time.monotonic() + APPROVAL_WAIT
    while time.monotonic() < deadline:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
        pending = [
            item["call_id"]
            for item in output
            if item.get("type") == "function_call" and item.get("status") == "pending"
        ]
        if pending:
            resolved = client.post(
                f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
                json={"call_id": pending[0], "action": "approve"},
            )
            assert resolved.status_code == 200, resolved.text
            return wait_for_reply(client, turn)
        time.sleep(0.2)
    raise AssertionError("the tool call never waited for approval")


def compact_with_a_new_upload(client: httpx.Client, upstream) -> ChatTurn:
    """Three long turns with uploads on the first two, then a fourth with its own upload."""
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    history = turns(3, files={1: [attachment("a")], 2: [attachment("b")]})
    chat_id, last_id = seed_chat(client, history)
    turn, _ = ask(
        client, "question 4", chat_id=chat_id, parent_id=last_id, message_files=[attachment("new")]
    )
    return turn


def test_the_compacting_turn_lists_its_own_upload(chat_settings, make_user, upstream):
    chat_settings(**compaction())
    with make_user().client() as client:
        compact_with_a_new_upload(client, upstream)

    assert listed_files(last_chat_call(upstream)) == {
        "question 3": [],
        "question 4": ["file-new"],
    }, "a kept message was given a compacted message's files (#31411)"


def test_the_turn_after_the_checkpoint_keeps_every_upload_in_place(
    chat_settings, make_user, upstream
):
    chat_settings(**compaction())
    with make_user().client() as client:
        turn = compact_with_a_new_upload(client, upstream)
        chat_settings(**compaction(threshold=1_000_000))
        upstream.queue(reply.text("answer 5"))
        ask(
            client,
            "question 5",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
            message_files=[attachment("five")],
        )

    assert listed_files(last_chat_call(upstream)) == {
        "question 3": [],
        "question 4": ["file-new"],
        "question 5": ["file-five"],
    }, "the replayed checkpoint shifted the uploads onto other messages (#31411)"


def test_uploads_are_still_listed_after_a_tool_approval(chat_settings, make_user, upstream):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, turns(1, files={1: [attachment("a")]}))
        turn = send_message(
            client,
            "question 2",
            chat_id=chat_id,
            parent_id=last_id,
            message_files=[attachment("b")],
            params={"tool_approval_mode": "ask"},
        )
        message = approve_every_call(client, turn)

    assert message["content"].endswith("It is late.")
    first, after_approval = [body for body in upstream.chat_requests() if body.get("stream")][-2:]
    expected = {"question 1": ["file-a"], "question 2": ["file-b"]}
    assert listed_files(first) == expected
    assert listed_files(after_approval) == expected, (
        "the request after the approval lost the uploads (af6b82a18)"
    )


def test_a_first_message_upload_survives_a_tool_approval(chat_settings, make_user, upstream):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is late."))
    with make_user().client() as client:
        turn = send_message(
            client,
            "question 1",
            message_files=[attachment("only")],
            params={"tool_approval_mode": "ask"},
        )
        approve_every_call(client, turn)

    assert listed_files(last_chat_call(upstream)) == {"question 1": ["file-only"]}


def test_every_message_keeps_its_uploads_after_compaction_and_approval(
    chat_settings, make_user, upstream
):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True, **compaction())
    upstream.queue(
        reply.text(SUMMARY, match=is_summary_request),
        reply.tool_call("get_current_timestamp", {}, match=lambda body: bool(body.get("stream"))),
        reply.text("It is late.", match=lambda body: bool(body.get("stream"))),
    )
    with make_user().client() as client:
        history = turns(3, files={1: [attachment("a")], 2: [attachment("b")]})
        chat_id, last_id = seed_chat(client, history)
        turn = send_message(
            client,
            "question 4",
            chat_id=chat_id,
            parent_id=last_id,
            message_files=[attachment("new")],
            params={"tool_approval_mode": "ask"},
        )
        approve_every_call(client, turn)

    after_approval = listed_files(last_chat_call(upstream))
    expected = {"question 1": ["file-a"], "question 2": ["file-b"], "question 3": []}
    for question, files in after_approval.items():
        assert files == {**expected, "question 4": ["file-new"]}[question], after_approval
    assert after_approval["question 4"] == ["file-new"]


def test_an_uncompacted_chat_lists_each_upload_on_its_message(make_user, upstream):
    upstream.queue(reply.text("answer 3"))
    with make_user().client() as client:
        history = turns(2, files={1: [attachment("a")], 2: [attachment("b")]})
        chat_id, last_id = seed_chat(client, history)
        ask(
            client,
            "question 3",
            chat_id=chat_id,
            parent_id=last_id,
            message_files=[attachment("c")],
        )

    assert listed_files(last_chat_call(upstream)) == {
        "question 1": ["file-a"],
        "question 2": ["file-b"],
        "question 3": ["file-c"],
    }


def test_several_uploads_on_one_message_are_listed_in_order(make_user, upstream):
    upstream.queue(reply.text("answer 1"))
    uploads = [attachment("first"), attachment("second"), attachment("third")]
    with make_user().client() as client:
        ask(client, "question 1", message_files=uploads)

    assert listed_files(last_chat_call(upstream)) == {
        "question 1": ["file-first", "file-second", "file-third"]
    }


def test_a_caller_sending_its_own_tools_gets_no_upload_list(make_user, upstream):
    own_tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
    upstream.queue(reply.text("answer 1"))
    with make_user().client() as client:
        ask(client, "question 1", message_files=[attachment("a")], tools=[own_tool])

    assert listed_files(last_chat_call(upstream)) == {"question 1": []}
