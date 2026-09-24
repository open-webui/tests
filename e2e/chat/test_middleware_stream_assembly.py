"""The stream assembly regressions as a person sees them in the chat page.

* #27414 (`381149ea5`): an outlet filter's in-place edit to the reply's output was not stored, so
  the reloaded chat showed the reply without it.
* #26645 (`051a1f6`): reasoning that arrived after the answer rendered below it, and a
  `reasoning_details` delta with no text rendered an empty thinking block.
* #27411 (`8ab44ed`): a provider that hung up while the reply continued after a tool call left
  the reply silently unfinished.
* #27074 (`dd514ee20`): a plain JSON error line was not stored, so the reloaded chat showed no
  error.
* #29040 (`87bed3f0b`): a tool round sent the page an empty message item, and the next round's
  first thinking chunk was written into it, so the thinking streamed as the answer until the
  turn finished. The stored reply was right, so only the live page shows it.

Twin of unit/chat/test_middleware_stream_assembly.py.

Discriminates: passes on dev `bbfa876af`; each test fails with its fix reverted in the backend and
the dev frontend build unchanged (one mutation each).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.filters import global_filter
from harness.second_provider import OPENAI_CONFIG, attach, sse, tool_call_delta
from harness.socket_client import connected
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SECOND_MODEL = "second-model"
THOUGHTS = re.compile(r"^(Thinking|Thought)")


def _chat_id(page) -> str:
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    return page.url.rsplit("/", 1)[-1]


def thinking_blocks(page):
    return last_reply(page).get_by_role("button", name=THOUGHTS)


@pytest.fixture
def second_provider(preserve, admin, listener):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        attach(client, listener, SECOND_MODEL)
    return listener


@pytest.fixture
def second_model_page(page_for, make_user, second_provider):
    """A signed-in admin's chat page with the second connection's model selected."""
    page = page_for(make_user(role="admin"))
    page.goto(f"/?models={SECOND_MODEL}")
    return page


def test_an_outlet_edit_to_the_output_survives_a_reload(page_for, make_user, admin, upstream):
    appends = """
class Filter:
    def outlet(self, body: dict) -> dict:
        body["messages"][-1]["output"][-1]["content"][0]["text"] += " [reviewed]"
        return body
"""
    upstream.queue(reply.text("hello there"))
    person = make_user()
    with admin.client() as client, global_filter(client, appends), connected(person) as socket:
        page = page_for(person)
        send(page, "hi")
        expect_reply(page, "hello there")
        socket.wait_for(_chat_id(page), "chat:outlet")

        page.reload()
        expect_reply(page, "hello there [reviewed]")


def test_reasoning_after_the_answer_shows_above_it(second_model_page, second_provider):
    events = ({"content": "the answer is 42"}, {"reasoning": "second thoughts"})
    second_provider.route("POST", "/v1/chat/completions", sse(*events))
    send(second_model_page, "think late")
    expect_reply(second_model_page, "the answer is 42")

    block = thinking_blocks(second_model_page)
    expect(block).to_have_count(1)
    answer = last_reply(second_model_page).get_by_text("the answer is 42")
    assert block.bounding_box()["y"] < answer.bounding_box()["y"], (
        "the thinking block that arrived after the answer rendered below it (#26645)"
    )


def test_reasoning_details_without_text_show_no_thinking_block(second_model_page, second_provider):
    empty_details = {"reasoning_details": [{"type": "reasoning.text", "index": 0}]}
    second_provider.route(
        "POST", "/v1/chat/completions", sse(empty_details, {"content": "plain answer"})
    )
    send(second_model_page, "no thoughts")
    expect_reply(second_model_page, "plain answer")

    expect(thinking_blocks(second_model_page)).to_have_count(0)


def test_a_provider_that_hangs_up_after_a_tool_call_shows_an_error(
    second_model_page, second_provider
):
    def tool_call_then_hang_up(_request):
        if len(second_provider.requests_to("/v1/chat/completions")) == 1:
            clock = tool_call_delta("get_current_timestamp", {})
            return sse(clock, finish_reason="tool_calls")
        raise ConnectionResetError("the provider hangs up")

    second_provider.route("POST", "/v1/chat/completions", tool_call_then_hang_up)
    send(second_model_page, "what time is it?")

    expect_reply(second_model_page, "Server Connection Error")


def test_a_plain_json_error_line_shows_after_a_reload(page_for, make_user, second_provider):
    second_provider.route("POST", "/v1/chat/completions", sse('{"error": "rate limit exceeded"}'))
    owner = make_user(role="admin")
    page = page_for(owner)
    page.goto(f"/?models={SECOND_MODEL}")
    with connected(owner) as socket:
        send(page, "hello")
        socket.wait_for(_chat_id(page), "chat:active", active=False)

    page.reload()
    expect_reply(page, "rate limit exceeded")


def test_thinking_after_a_tool_call_does_not_stream_as_the_answer(chat_page_for_user, upstream):
    page = chat_page_for_user
    pieces = [f"piece-{index} " for index in range(15)]
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, reasoning="I should check the clock"),
        reply.text(pieces, reasoning="the clock says now", chunk_delay=0.4),
    )
    send(page, "what time is it?")
    expect(last_reply(page)).to_contain_text("piece-1 ", timeout=30_000)

    thought_shown_as_answer = conversation(page).get_by_text("the clock says now").is_visible()
    assert "piece-14" not in last_reply(page).inner_text(), "the reply finished before the check"
    assert not thought_shown_as_answer, (
        "the thinking after a tool call streamed into the empty message item the tool round "
        "left behind, so it rendered as the answer until the turn finished (#29040)"
    )


@pytest.fixture
def chat_page_for_user(page_for, make_user):
    page = page_for(make_user())
    page.goto("/")
    return page
