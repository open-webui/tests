"""Regression: a realtime voice call hung up in the middle of a long spoken answer.

A voice provider streams its speech faster than it plays, so a long answer piles up in the
browser waiting to be played. Once more than 120 seconds of it were waiting, the call ended with
Call disconnected and "Voice playback exceeded the 120 second buffer." although nothing was wrong
with the answer. Here the provider (`harness.realtime_provider`) streams a 150 second answer in
one-second chunks as fast as it can; the call stays connected and keeps speaking.

Fixed on dev by b8b3fb906 (open-webui/open-webui#32070): playback no longer has a cap and no
longer ends the call.

Discriminates: passes on the dev b8b3fb906 build, fails on the build of its parent 5729d7ad8,
which is that build with b8b3fb906 reverted (the overlay reads Call disconnected with the buffer
error).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import expect

from harness.realtime_provider import SAMPLE_RATE, serving_realtime_provider, using_realtime
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_ANSWER_SECONDS = 150
BUFFER_ERROR = "Voice playback exceeded"


@pytest.fixture
def realtime(admin, e2e_instance):
    with serving_realtime_provider() as provider, admin.client() as client:
        with using_realtime(client, provider):
            yield provider


def test_a_long_spoken_answer_keeps_the_call_connected(voice_page_for, make_user, realtime):
    realtime.speech = b"\x00\x00" * (SAMPLE_RATE * LONG_ANSWER_SECONDS)
    question = f"read me the whole tide table {uuid.uuid4().hex[:6]}"
    answer = f"Here is the tide table {uuid.uuid4().hex[:6]}."
    realtime.hears(question, answers=answer)
    page = voice_page_for(make_user())

    start_call(page)
    call = realtime.wait_for_call()
    realtime.wait_for(lambda: answer in realtime.spoken, "the long answer sent", timeout=30.0)
    # the whole answer arrived; give the browser a moment to queue it before checking
    page.wait_for_timeout(3000)

    overlay = page.get_by_role("region", name="Voice call")
    expect(overlay).not_to_contain_text(BUFFER_ERROR)
    expect(overlay.get_by_role("alert")).to_have_count(0)
    expect(call_status(page, "Tap to interrupt")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert not call.ended.is_set(), "the call to the voice provider closed during the answer"
