"""Regression: interrupting a Voice mode reply spoke its remaining sentences anyway.

Issue open-webui/open-webui#31876, fix 7dd9fc707 (PR open-webui/open-webui#31877). With Response
Splitting on Punctuation or Paragraphs a reply is spoken one part at a time by a loop that runs
until its abort signal fires. Tapping to interrupt stopped the sentence being spoken but never
fired the signal, so the loop went on to the next sentence a moment later and the call kept
talking over the user. Interrupting now aborts the loop.

Discriminates: passes on the dev b859124f9 build, fails on that build with 7dd9fc707 reverted
(the second sentence starts playing right after the tap).
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.audio_engine import AUDIO_CONFIG, AudioEngine
from utils.speech_audio import silent_wav
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# how long the next sentence is given to start; the call's next recording takes longer to arrive
QUIET_SECONDS = 1.5
RECORD_PLAYS = """() => {
    window.sentencesPlayed = 0;
    document.addEventListener(
        'playing',
        (event) => {
            if (event.target.src.startsWith('blob:')) window.sentencesPlayed += 1;
        },
        true
    );
}"""
SENTENCES_PLAYED = "() => window.sentencesPlayed"
SEPARATORS = {"punctuation": " ", "paragraphs": "\n\n"}


@pytest.fixture(params=sorted(SEPARATORS))
def split_on(request, admin, speech_engine) -> str:
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        saved = client.post(
            AUDIO_CONFIG[1],
            json={"tts": {**current["tts"], "SPLIT_ON": request.param}, "stt": current["stt"]},
        )
    assert saved.status_code == 200, saved.text
    return request.param


@pytest.fixture
def long_sentences(speech_engine) -> AudioEngine:
    speech_engine.speech = silent_wav(8)
    return speech_engine


def queue_reply(upstream, make_user, split_on: str):
    """A caller whose reply has three sentences of its own; returns the caller and the reply."""
    words = uuid.uuid4().hex[:8]
    text = SEPARATORS[split_on].join(
        f"The {ordinal} keeper lights the harbour lamp at dusk {words}."
        for ordinal in ("first", "second", "third")
    )
    upstream.queue(
        reply.text(text, match=lambda body: words in json.dumps(body.get("messages", [])))
    )
    caller = make_user()
    with caller.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"system": words}})
    assert saved.status_code == 200, saved.text
    return caller, words


def wait_for_sentence_audio(page: Page, count: int) -> None:
    page.wait_for_function(f"() => window.sentencesPlayed >= {count}", timeout=TURN_TIMEOUT_MS)


def test_interrupting_the_call_stops_the_rest_of_the_reply(
    voice_page_for, make_user, upstream, long_sentences, split_on
):
    caller, words = queue_reply(upstream, make_user, split_on)
    page = voice_page_for(caller)
    page.add_init_script(f"({RECORD_PLAYS})()")
    page.goto("/")
    start_call(page)
    wait_for_sentence_audio(page, 1)

    # the status line is a live region since 093bfce2b, so the button it sits in has no name
    call_status(page, "Tap to interrupt").click()
    expect(call_status(page, "Listening...")).to_be_visible()
    transcribed = len(long_sentences.transcription_requests())
    time.sleep(QUIET_SECONDS)

    assert page.evaluate(SENTENCES_PLAYED) == 1, "the call spoke on after it was interrupted"
    assert len(long_sentences.transcription_requests()) == transcribed, "a new turn began"
    spoken = [request["input"] for request in long_sentences.speech_requests()]
    assert any(words in text for text in spoken), "the reply was never spoken"
