"""Regression: pressing M during a voice call typed an "m" into the chat box instead of muting.

Issue open-webui/open-webui#30406, fix e14f24558 (PR open-webui/open-webui#30421). The call
overlay mutes on M unless focus is in a text field, so typing still works. After every voice
message the chat input took keyboard focus back, so the next M went into the chat box. The chat
input now leaves focus alone while a call is open.

Discriminates: passes on the dev efe63bd34 build, fails on that build with e14f24558 reverted
(after the first turn M types into the chat box and the call stays unmuted).

Red on dev c4c10aa25 (open-webui/open-webui#33557): a call started in a new chat loses its Voice tab
and End call button when the first message saves the chat, and a standard call's reply stays blank.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input, expect_reply
from utils.voice_call import (
    TURN_TIMEOUT_MS,
    call_status,
    show_chat,
    start_call,
    wait_for_transcriptions,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def finish_turn(page: Page, speech_engine, turn: int) -> None:
    wait_for_transcriptions(speech_engine, turn)
    expect(page.get_by_label("Chat Conversation").locator(".chat-assistant")).to_have_count(
        turn, timeout=TURN_TIMEOUT_MS
    )
    expect_reply(page, "ok")
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)


def hidden_chat_box(page: Page) -> Locator:
    # the chat's box behind the selected Voice tab, which the shortcut must not type into
    return page.locator("#chat-input")


def test_m_mutes_the_call_after_a_turn(voice_page_for, make_user, speech_engine):
    page = voice_page_for(make_user())
    start_call(page)
    finish_turn(page, speech_engine, 1)

    page.keyboard.press("m")

    expect(call_status(page, "Muted")).to_be_visible()
    expect(page.get_by_role("button", name="Unmute")).to_be_visible()
    expect(hidden_chat_box(page)).to_have_text("")


def test_m_keeps_working_turn_after_turn(voice_page_for, make_user, speech_engine):
    page = voice_page_for(make_user())
    start_call(page)
    finish_turn(page, speech_engine, 1)
    page.keyboard.press("m")
    expect(call_status(page, "Muted")).to_be_visible()
    page.keyboard.press("m")
    finish_turn(page, speech_engine, 2)

    page.keyboard.press("m")

    expect(call_status(page, "Muted")).to_be_visible()
    expect(hidden_chat_box(page)).to_have_text("")


def test_typing_m_into_the_chat_box_during_a_call_still_types(
    voice_page_for, make_user, speech_engine
):
    page = voice_page_for(make_user())
    start_call(page)
    finish_turn(page, speech_engine, 1)
    show_chat(page)

    chat_input(page).click()
    page.keyboard.type("mm")

    expect(chat_input(page)).to_have_text("mm")
    expect(call_status(page, "Muted")).to_have_count(0)
