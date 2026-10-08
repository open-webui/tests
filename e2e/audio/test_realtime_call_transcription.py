"""Regression: speech that came to nothing still made the realtime voice talk.

Commit 87a937459: a segment of a realtime voice call whose transcription failed, or came back
without words, used to have the voice say "I could not transcribe that. Please repeat it.";
now an empty segment is passed over in silence, and a failed one shows the error "A voice
segment could not be transcribed. Please try again." with no reply. The call goes on, so the
next question is answered as usual. And a transcript of an earlier segment that arrives while
the caller is already speaking again no longer counts as the end of that newer speech: the voice
waits for the caller to finish before it answers anything.

Discriminates: passes on the dev 87a937459 build. With 87a937459 reverted (a frontend copy built
without its change to the call, on a backend copy without its change) all three go red: the
voice speaks the old "could not transcribe" status before the answer, no error shows for the
failed segment, and the earlier question's answer is asked for while the caller still speaks.
"""

from __future__ import annotations

import time
import uuid

import pytest
from playwright.sync_api import expect

from harness.realtime_provider import serving_realtime_provider, using_realtime
from utils.chat_ui import conversation
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FAILED_SEGMENT = "A voice segment could not be transcribed. Please try again."
STILL_SPEAKING_SECONDS = 2


@pytest.fixture
def realtime(admin, e2e_instance):
    with serving_realtime_provider() as fake, admin.client() as client:
        with using_realtime(client, fake):
            yield fake


def words() -> str:
    return uuid.uuid4().hex[:6]


def answers_asked_for(call) -> list[dict]:
    return call.received("response.create")


def test_a_segment_without_words_gets_no_reply_and_the_next_question_is_answered(
    voice_page_for, make_user, realtime
):
    question, answer = f"is the harbour open {words()}", f"The harbour is open {words()}."
    realtime.hears("")
    realtime.hears(question, answers=answer)
    page = voice_page_for(make_user())

    start_call(page)

    expect(conversation(page).get_by_text(question)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    realtime.wait_for(lambda: answer in realtime.spoken, "the answer spoken")
    assert realtime.spoken == [answer], "the voice spoke about the segment without words"
    expect(conversation(page).locator(".chat-user")).to_have_count(1)
    expect(page.get_by_text(FAILED_SEGMENT)).to_have_count(0)


def test_a_segment_that_cannot_be_transcribed_shows_an_error_and_gets_no_reply(
    voice_page_for, make_user, realtime
):
    question, answer = f"is the harbour open {words()}", f"The harbour is open {words()}."
    realtime.mishears()
    realtime.hears(question, answers=answer)
    page = voice_page_for(make_user())

    start_call(page)

    expect(page.get_by_text(FAILED_SEGMENT)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect(conversation(page).get_by_text(question)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    realtime.wait_for(lambda: answer in realtime.spoken, "the answer spoken")
    assert realtime.spoken == [answer], "the voice spoke about the failed segment"
    expect(conversation(page).locator(".chat-user")).to_have_count(1)
    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)


def test_the_voice_waits_for_speech_that_runs_on_past_an_earlier_transcript(
    voice_page_for, make_user, realtime
):
    first, second = f"is the harbour open {words()}", f"and the lighthouse {words()}"
    first_answer, second_answer = f"The harbour is open {words()}.", f"It is lit {words()}."
    still_speaking = realtime.hears_over(first, second, answers=(first_answer, second_answer))
    page = voice_page_for(make_user())

    start_call(page)

    expect(conversation(page).get_by_text(first)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    call = realtime.calls[-1]
    deadline = time.monotonic() + STILL_SPEAKING_SECONDS
    while time.monotonic() < deadline:  # bounded: no answer while the caller still speaks
        assert not answers_asked_for(call), "the voice answered while the caller was speaking"
        time.sleep(0.1)
    still_speaking.set()
    expect(conversation(page).get_by_text(second)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    realtime.wait_for(
        lambda: first_answer in realtime.spoken and second_answer in realtime.spoken,
        "both answers spoken",
    )
