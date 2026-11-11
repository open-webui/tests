"""Regression: the collapsed steps of a reply read "Explored" while the model was still thinking.

Fix commit `19957bc19` (open-webui/open-webui#29531) in the frontend's group of collapsed steps.
The group showed "Exploring" with a spinner only while one of its tool calls was running. Once
the tool call finished and the model went on thinking inside the same group, the header already
read "Explored", so the chat looked stuck until the answer arrived. An unfinished thinking
block in a reply that is still streaming now keeps the group on "Exploring".

The scripted model calls a tool, then thinks for a few seconds before it answers, so the page
shows the finished tool call and the open thinking block together.

Discriminates: passes on the dev bc2416c5d build; with the group's active check back to tool
calls only, the header reads "Explored" throughout the thinking and never "Exploring".

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "what time is it, after some thought?"
ANSWER = "It is now, having thought about it."
THINKING_SECONDS = 4.0


def steps_header(chat, text: str):
    return last_reply(chat).get_by_text(text, exact=True)


def wait_for_the_tool_result_to_be_sent(upstream, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(
            entry["role"] == "tool"
            for request in upstream.chat_requests()
            for entry in request.get("messages", [])
        ):
            return
        time.sleep(0.1)
    raise AssertionError("the tool call never finished")


def test_the_steps_read_exploring_while_the_model_thinks_after_a_tool(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(PROMPT)),
        reply.text(
            [ANSWER],
            reasoning="weighing the timestamp",
            chunk_delay=THINKING_SECONDS,
            match=reply.answering(PROMPT),
        ),
    )

    send(chat, PROMPT)
    wait_for_the_tool_result_to_be_sent(upstream)

    # the group forms once the thinking starts and stays open THINKING_SECONDS
    expect(steps_header(chat, "Exploring")).to_be_visible(timeout=THINKING_SECONDS * 2 * 1000)
    expect_reply(chat, ANSWER)
    expect(steps_header(chat, "Explored")).to_be_visible()
    expect(steps_header(chat, "Exploring")).to_have_count(0)


def test_the_steps_of_a_finished_reply_read_explored(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(PROMPT)),
        reply.text(ANSWER, reasoning="weighing the timestamp", match=reply.answering(PROMPT)),
    )

    send(chat, PROMPT)

    expect_reply(chat, ANSWER)
    expect(steps_header(chat, "Explored")).to_be_visible()
    expect(steps_header(chat, "Exploring")).to_have_count(0)
