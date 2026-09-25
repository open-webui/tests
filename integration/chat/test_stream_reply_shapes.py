"""Journey: reply shapes the streaming handler turns into what a user sees and keeps.

Many local models write their reasoning into the text itself between tags (`<think>` and its
kin, or a pair the chat settings name), often split across chunks, with attributes, or never
closed. The handler lifts that into a reasoning item and keeps the answer, unless the chat
turned tag detection off. A reply of only whitespace is stored empty, an admin can stop a reply
mid-stream and keep what arrived, and a model that keeps calling tools is cut off at the
configured number of rounds with a message saying so.

Discriminates: in a backend copy, dropping the attribute parsing fails the attributes case,
ignoring the chat's own tag pair fails the custom-tags case, looking for the end tag only past
the text already scanned fails the split case, leaving stopped items `in_progress` fails the
stop test and removing the iteration limit message fails the limit test.
"""

from __future__ import annotations

import time

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.inflight import start_slow_reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _items(message: dict, kind: str) -> list[dict]:
    return [item for item in message["output"] if item["type"] == kind]


def _reasoning_text(message: dict) -> list[str]:
    return [item["content"][0]["text"] for item in _items(message, "reasoning")]


def _ask_streamed(make_user, upstream, pieces: list[str], **options) -> dict:
    upstream.queue(reply.text(pieces))
    with make_user().client() as client:
        _, message = ask(client, "think it over", **options)
    return message


def test_tags_split_across_chunks_become_a_reasoning_item(make_user, upstream):
    pieces = ["Sure. <thi", "nk>pondering", " more</thi", "nk>The answer", " is 4."]

    message = _ask_streamed(make_user, upstream, pieces)

    [reasoning] = _items(message, "reasoning")
    assert reasoning["content"][0]["text"] == "pondering more"
    assert reasoning["status"] == "completed"
    assert (reasoning["start_tag"], reasoning["end_tag"]) == ("<think>", "</think>")
    assert [item["type"] for item in message["output"]] == ["message", "reasoning", "message"]
    assert message["output"][-1]["content"][0]["text"] == "The answer is 4."
    assert "pondering" not in message["content"]
    assert "<think>" not in message["content"]


def test_a_tag_keeps_its_attributes(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ['<think effort="high">deep</think>ok'])

    [reasoning] = _items(message, "reasoning")
    assert reasoning["attributes"] == {"effort": "high"}
    assert _reasoning_text(message) == ["deep"]
    assert message["content"] == "ok"


def test_the_chats_own_tag_pair_is_detected(make_user, upstream):
    message = _ask_streamed(
        make_user,
        upstream,
        ["<plan>step one</plan>Done."],
        params={"reasoning_tags": ["<plan>", "</plan>"]},
    )

    assert _reasoning_text(message) == ["step one"]
    assert message["content"] == "Done."


def test_detection_switched_off_keeps_the_tags_as_text(make_user, upstream):
    message = _ask_streamed(
        make_user, upstream, ["<think>visible</think>kept"], params={"reasoning_tags": False}
    )

    assert _items(message, "reasoning") == []
    assert message["content"] == "<think>visible</think>kept"


def test_an_unclosed_tag_is_closed_when_the_stream_ends(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["<think>never ", "closes"])

    [reasoning] = _items(message, "reasoning")
    assert reasoning["status"] == "completed"
    assert reasoning["content"][0]["text"] == "never closes"
    assert not message.get("error")


def test_a_reply_of_only_whitespace_is_stored_empty(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["   ", "\n"])

    assert message["content"] == ""
    assert [item["content"][0]["text"] for item in _items(message, "message")] == [""]


def test_an_admin_stops_a_reply_and_keeps_what_arrived(admin, upstream):
    with admin.client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.2)
        [task_id] = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        time.sleep(0.5)
        stopped = client.post(f"/api/tasks/stop/{task_id}")
        time.sleep(0.5)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()
        remaining = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]

    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["status"] is True
    message = stored["chat"]["history"]["messages"][turn.assistant_message_id]
    assert message["content"].startswith("part-0 ")
    assert "part-19" not in message["content"]
    assert [item["status"] for item in message["output"]] == ["incomplete"]
    assert remaining == []


def test_only_an_admin_may_stop_a_task_by_id(make_user, upstream):
    with make_user().client() as client:
        turn = start_slow_reply(client, upstream, chunk_delay=0.05)
        [task_id] = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        refused = client.post(f"/api/tasks/stop/{task_id}")

    assert refused.status_code in (401, 403), refused.text


@pytest.mark.slow
def test_a_model_that_keeps_calling_tools_is_cut_off(instance_with):
    limited = instance_with({"CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS": "2"})
    limited.upstream.queue(
        *(reply.tool_call("get_current_timestamp", {}, call_id=f"call_{n}") for n in range(3)),
    )
    with admin_of(limited).client() as client:
        _, message = ask(client, "what time is it, again and again?")

    # the third call is asked for but never run
    assert len(limited.upstream.chat_requests()) == 3
    assert len(_items(message, "function_call")) == 3
    assert len(_items(message, "function_call_output")) == 2
    assert "Tool-call limit reached (2 iterations)." in str(message.get("error"))
