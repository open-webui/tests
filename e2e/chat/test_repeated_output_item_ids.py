"""Regression: a chat whose reply repeats output item ids froze the tab when it was reopened.

Fix commit `8b4d3ac52` (open-webui/open-webui#31838, issue #31837) in the frontend's display items
of a structured reply. Some Responses API providers (OpenVINO Model Server) number their output
items from zero in every response, so a reply that wrote text, called a tool and then answered
was stored with two message items of one id, and showing it locked the tab. Repeated ids are now
renamed when the reply is shown, saved chats included.

One test has the Responses stand-in answer both rounds with the same message id and reload the
chat it made; the other opens a chat stored in that shape. Every wait is bounded, so a frozen
tab fails the test and does not hang the run.

Discriminates: passes on the dev b859124f9 build; in the build with 8b4d3ac52 reverted the
reopened chat never shows its reply (the tab is unresponsive).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), a stored chat
dragged into a tab beside a new chat.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import responses_provider as responses_api
from harness.chat_history import seed_chat
from harness.second_provider import OPENAI_CONFIG
from utils.chat_surface import ChatSurface
from utils.chat_ui import expect_reply, last_reply, send
from utils.model_selector import select_model

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

MODEL = responses_api.RESPONSES_MODEL
BEFORE_THE_TOOL = "Let me check the clock."
AFTER_THE_TOOL = "It is late."
RESPONSIVE_MS = 10_000


@pytest.fixture
def responses(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        yield responses_api.connect_responses(client, listener)


def expect_responsive(chat: ChatSurface) -> None:
    # a frozen renderer never answers, so the wait ends at its timeout
    chat.page.wait_for_function("() => true", timeout=RESPONSIVE_MS)


def expect_whole_reply(chat: ChatSurface) -> None:
    expect(last_reply(chat)).to_contain_text(BEFORE_THE_TOOL, timeout=RESPONSIVE_MS)
    expect(last_reply(chat)).to_contain_text(AFTER_THE_TOOL, timeout=RESPONSIVE_MS)
    expect(last_reply(chat).get_by_text("View Result from get_current_timestamp")).to_be_visible(
        timeout=RESPONSIVE_MS
    )
    expect_responsive(chat)


def test_a_reply_with_the_provider_reusing_item_ids_shows_after_a_reload(
    chat_for, make_user, responses
):
    # an admin account, so the Responses connection's model needs no access grant
    chat = chat_for(make_user(role="admin"))
    select_model(chat, MODEL)
    responses.answer(
        responses_api.events_stream(
            *responses_api.message(BEFORE_THE_TOOL),
            *responses_api.function_call("get_current_timestamp", {}, index=1),
            responses_api.completed(),
        ),
        responses_api.events_stream(
            *responses_api.message(AFTER_THE_TOOL), responses_api.completed()
        ),
    )

    send(chat, "what time is it?")
    expect_reply(chat, AFTER_THE_TOOL)
    chat = chat.reload()

    expect_whole_reply(chat)


def test_a_stored_reply_with_repeated_item_ids_opens_with_text_and_tool_call(chat_for, make_user):
    account = make_user()
    call = {
        "type": "function_call",
        "id": "item_1",
        "call_id": "call_1",
        "name": "get_current_timestamp",
        "arguments": "{}",
        "status": "completed",
    }
    result = {
        "type": "function_call_output",
        "id": "item_1_out",
        "call_id": "call_1",
        "output": [{"type": "input_text", "text": '{"current_timestamp": 1}'}],
        "status": "completed",
    }
    output = [
        _message("item_0", BEFORE_THE_TOOL),
        call,
        result,
        _message("item_0", AFTER_THE_TOOL),
    ]
    history = [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": AFTER_THE_TOOL, "output": output},
    ]
    with account.client() as client:
        chat_id, _ = seed_chat(client, history, model=MODEL)
    chat = chat_for(account, chat_id=chat_id)

    expect_whole_reply(chat)


def _message(item_id: str, text: str) -> dict:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
