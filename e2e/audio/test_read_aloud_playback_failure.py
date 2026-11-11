"""Regression: a failed Read Aloud playback left the speaker button stuck in "speaking".

Issue open-webui/open-webui#30262, fix 7924ce7d7 (PR open-webui/open-webui#30373). The audio
queue ignored a rejected `play()`, so when the browser refused to play the speech (as iOS does
for playback started outside a tap) its state never reset: the button stayed in "speaking", no
error showed and the next tap only stopped a playback that never started. A rejected playback
now resets the queue, returns the button to idle and shows "Audio playback failed". The iOS
unlock itself needs WebKit on a device and is not covered here.

Discriminates: passes on the dev efe63bd34 build, fails on that build with 7924ce7d7 reverted
(no error shows and the second tap asks for no speech).

The test runs on the saved chat in the main chat and dragged into a tab beside another chat
(`utils/chat_surface.py`), the only places a saved chat opens. A side chat answer had no Read
Aloud (open-webui/open-webui#34014, fix PR open-webui/open-webui#34064): the tab variants are red
on dev c15f28649.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.chat_history import seed_chat
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT, ChatSurface

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The tide comes in twice a day."
UNPLAYABLE = b"this is not audio at all"


@pytest.fixture
def unplayable_speech(speech_engine):
    speech_engine.speech = UNPLAYABLE
    return speech_engine


def open_answer(chat_for, make_user) -> ChatSurface:
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "tides?"}, {"role": "assistant", "content": ANSWER}],
        )
    chat = chat_for(owner, chat_id=chat_id)
    expect(chat.conversation.get_by_text(ANSWER)).to_be_visible()
    return chat


def read_aloud(chat: ChatSurface):
    """Press Read Aloud; returns the speech request the page made."""
    with chat.page.expect_request(
        lambda request: request.url.endswith("/api/v1/audio/speech")
    ) as asked:
        chat.panel.get_by_role("button", name="Read Aloud").click()
    return asked.value


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_failed_playback_shows_an_error_and_the_next_tap_reads_again(
    chat_for, make_user, unplayable_speech
):
    """In a side chat: open-webui/open-webui#34014."""
    chat = open_answer(chat_for, make_user)

    read_aloud(chat)
    expect(chat.page.get_by_text("Audio playback failed").first).to_be_visible()

    # a wedged button only stops on this tap and asks for no speech
    again = read_aloud(chat)
    assert again.post_data_json["input"] == ANSWER
