"""Regression: a voice call hung in "speaking" with no sound when text-to-speech failed.

Issue open-webui/open-webui#30052, fix 24c01bd76 (PR open-webui/open-webui#30372). When the
speech provider failed during a call, the failed sentence never reached the audio cache and the
playback loop queued it again every 200 ms, forever. The overlay stayed on "Tap to interrupt",
showed no error and ignored the microphone until tapped. A failed sentence now marks its reply
as failed: the provider's error shows once, and the call goes back to listening.

Discriminates: passes on the dev efe63bd34 build, fails on that build with 24c01bd76 reverted
(no error shows and the call never takes the next turn).

Red on dev c4c10aa25 (open-webui/open-webui#33557): a call started in a new chat loses its Voice tab
and End call button when the first message saves the chat, and a standard call's reply stays blank.
"""

from __future__ import annotations

import json
import re
import time
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import Actor
from harness.audio_engine import AudioEngine
from harness.listener import json_answer
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call, wait_for_transcriptions

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROVIDER_ERROR = "the speech provider is down"
SPEECH_ERROR = re.compile(r"External: 500.*/audio/speech")  # how the server reports the failure


@pytest.fixture
def words() -> str:
    # the server caches speech by its text, so a reply spoken in an earlier test never fails
    return uuid.uuid4().hex


@pytest.fixture
def caller(make_user, words) -> Actor:
    """An account whose chats carry `words` in their system prompt, so its replies are its own."""
    account = make_user()
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"system": words}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["ui"]["system"] == words, saved.text
    return account


@pytest.fixture(autouse=True)
def fresh_replies(upstream, words):
    # a turn left over from an earlier test's call must not take these replies
    def from_this_call(body: dict) -> bool:
        return words in json.dumps(body.get("messages", []))

    upstream.queue(
        reply.text(f"First {words}.", match=from_this_call),
        reply.text(f"Second {words}.", match=from_this_call),
    )


def wait_for_speech(engine: AudioEngine, text: str) -> None:
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1000
    while not any(request["input"] == text for request in engine.speech_requests()):
        if time.monotonic() > deadline:
            spoken = [request["input"] for request in engine.speech_requests()]
            raise AssertionError(f"{text!r} was never spoken, only {spoken}")
        time.sleep(0.1)


@pytest.fixture
def failing_speech(speech_engine, listener):
    listener.route(
        "POST", "/audio/speech", json_answer({"error": {"message": PROVIDER_ERROR}}, status=500)
    )
    return speech_engine


def test_a_speech_failure_shows_the_error_and_the_call_listens_again(
    voice_page_for, caller, failing_speech
):
    page = voice_page_for(caller)
    start_call(page)
    wait_for_transcriptions(failing_speech, 1)

    expect(page.get_by_text(SPEECH_ERROR).first).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    wait_for_transcriptions(failing_speech, 2)


def test_the_next_turn_asks_for_speech_again(voice_page_for, caller, words, failing_speech):
    page = voice_page_for(caller)
    start_call(page)
    wait_for_transcriptions(failing_speech, 2)

    wait_for_speech(failing_speech, f"Second {words}.")
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)


def test_a_working_speech_provider_keeps_the_call_going(voice_page_for, caller, speech_engine):
    page = voice_page_for(caller)
    start_call(page)
    wait_for_transcriptions(speech_engine, 2)

    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert speech_engine.speech_requests(), "the reply was never spoken"
