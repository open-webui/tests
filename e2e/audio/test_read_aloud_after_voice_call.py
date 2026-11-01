"""Regression: Read Aloud stayed silent after a Voice mode call had ended.

Issue open-webui/open-webui#31874, fix 35b0db7ed (PR open-webui/open-webui#31875). Read Aloud and
Voice mode play through one hidden audio element on the chat page. Ending a call mutes it, and
the Read Aloud player never unmuted it, so every reply read aloud after a call played with no
sound until the page was reloaded. The player now unmutes the element before each sentence.

Since 78613d86f a call plays through an audio element of its own and leaves the shared one alone,
so the test mutes the shared element itself after the call and requires Read Aloud to unmute it.

The same ended call must let the screen go: since a9227421d (open-webui/open-webui#32251) the
call's screen wake lock is released on hang-up, where a standard call used to keep it after
closing. A fake `navigator.wakeLock` counts requests and releases.

Discriminates: passes on the dev b859124f9 build, fails on that build with 35b0db7ed reverted
(Read Aloud plays on an audio element that is still muted). Retargeted: passes on dev 4ca8fdc3f
and fails on a build of it without the unmute. Passes on dev c31423955 and fails with a9227421d
reverted (the wake lock is still held after the call ends).
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import last_reply
from utils.speech_audio import silent_wav
from utils.voice_call import (
    TURN_TIMEOUT_MS,
    end_call_button,
    expect_screen_kept_on,
    record_wake_locks,
    start_call,
    wait_for_transcriptions,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

IS_MUTED = "() => document.getElementById('audioElement').muted"
IS_PLAYING = """() => {
    const audio = document.getElementById('audioElement');
    return !audio.paused && audio.played.length > 0;
}"""


def test_read_aloud_plays_unmuted_after_a_call_ends(
    voice_page_for, make_user, speech_engine, upstream
):
    speech_engine.speech = silent_wav(6)
    words = uuid.uuid4().hex[:8]
    upstream.queue(
        reply.text(
            f"Heard you {words}.",
            match=lambda body: words in json.dumps(body.get("messages", [])),
        )
    )
    caller = make_user()
    with caller.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"system": words}})
    assert saved.status_code == 200, saved.text
    page = voice_page_for(caller)
    record_wake_locks(page)
    start_call(page)
    expect_screen_kept_on(page, True)
    wait_for_transcriptions(speech_engine, 1)
    expect(last_reply(page)).to_contain_text(f"Heard you {words}.", timeout=TURN_TIMEOUT_MS)

    end_call_button(page).click()
    expect(page.get_by_role("button", name="End call")).to_have_count(0)
    expect_screen_kept_on(page, False)
    # what an ended call did to the shared player before 78613d86f, and anything else may do
    page.evaluate("() => { document.getElementById('audioElement').muted = true; }")

    page.get_by_role("button", name="Read Aloud").last.click()

    page.wait_for_function(IS_PLAYING)
    assert not page.evaluate(IS_MUTED), "Read Aloud is playing on a muted audio element"
