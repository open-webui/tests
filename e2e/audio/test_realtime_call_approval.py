"""Journey: a spoken request whose answer needs a tool approval, settled in the chat.

With tool permissions on and the composer set to Ask for approval, a realtime call hands a spoken
question to the chat model, which calls a tool. The approval card shows in the chat as for a
typed message, the call panel shows Waiting for approval and the voice provider is asked to tell
the user to review it in chat. Allowing the call in the chat runs the tool, the model's answer
shows and is then spoken, and the panel goes back to listening. Here the provider is
`harness.realtime_provider`.

Discriminates: passes on the dev ebc6add67 build; fails on a frontend copy whose call never
treats a tool call waiting for approval as an approval (no approval status is spoken and the
panel keeps showing Thinking...).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.realtime_provider import serving_realtime_provider, using_realtime
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REVIEW_IN_CHAT = "Please review the approval or question in chat. I will wait for you there."


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


@pytest.fixture
def realtime(admin, e2e_instance):
    with serving_realtime_provider() as provider:
        with admin.client() as client, using_realtime(client, provider):
            yield provider


def ask_for_approval(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    # the sidebar has a "More" button of its own
    page.locator("#input-menu-button").click()
    page.get_by_role("button", name="Tool Permissions").click()
    page.get_by_role("button", name="Ask for approval").click()
    page.keyboard.press("Escape")


def test_a_spoken_request_waits_for_the_approval_in_chat_and_is_then_answered(
    tool_approval_on, voice_page_for, make_user, realtime, upstream
):
    question, answer = f"what time is it {uuid.uuid4().hex[:6]}", "It is half past six."
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )
    page = voice_page_for(make_user())
    ask_for_approval(page)
    realtime.hears(question)

    start_call(page)

    allow = conversation(page).get_by_role("button", name="Allow", exact=True)
    expect(allow).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    expect(call_status(page, "Waiting for approval")).to_be_visible()
    realtime.wait_for(lambda: realtime.spoken, "a spoken status")
    assert REVIEW_IN_CHAT in realtime.spoken[0]

    allow.click()

    expect_reply(page, answer)
    realtime.wait_for(lambda: answer in realtime.spoken, "the answer spoken")
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
