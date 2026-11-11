"""Regression: Read Aloud and auto-playback ignored the Speech Playback Speed setting.

Issue open-webui/open-webui#31870, fix 273180a0d (PR open-webui/open-webui#31881). The audio queue
plays a reply one sentence at a time through a single audio element, and the browser puts the
element's playback rate back to its default each time a new source is loaded. The queue only
set the current rate, so the chosen speed was lost at every sentence. It now sets the default
rate too.

Discriminates: passes on the dev b859124f9 build, fails on that build with 273180a0d reverted
(the sentences play at normal speed).

Both tests run in the main chat and in a side chat (`utils/chat_surface.py`): Read Aloud on a saved
chat also dragged into a tab, auto-playback in every way a side chat starts and every state it can
be in. A side chat answer had no Read Aloud (open-webui/open-webui#34014) and ignored
Auto-Playback Response (open-webui/open-webui#34024), fix PR open-webui/open-webui#34064: the side
chat variants are red on dev c15f28649. The rate recorder starts in the open chat page.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import DRAGGED_TAB, EVERYWHERE, MAIN_CHAT, SIDE_CHAT
from utils.chat_ui import expect_reply, send
from utils.speech_audio import silent_wav

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SPEED = 2
SENTENCES = 3
# every time the element starts playing a sentence, with the rate it plays at; the silent clip
# the player plays to unlock the element on the first gesture is no sentence
RECORD_RATES = """() => {
    window.playedRates = [];
    document.addEventListener(
        'playing',
        (event) => {
            if (!event.target.src.startsWith('blob:')) return;
            window.playedRates.push(event.target.playbackRate);
        },
        true
    );
}"""
PLAYED_RATES = "() => window.playedRates"


@pytest.fixture(autouse=True)
def short_sentences(speech_engine):
    speech_engine.speech = silent_wav(1.5)


def reply_text() -> str:
    # the server caches speech by its text, so the reply is this test's own
    tag = uuid.uuid4().hex[:6]
    # the player merges sentences shorter than 50 characters into the one before
    return " ".join(
        f"The {ordinal} keeper lights the harbour lamp at dusk {tag}."
        for ordinal in ("first", "second", "third")
    )


def set_speed(page: Page) -> None:
    page.goto("/?settings=audio")
    tab = page.locator("#tab-audio")
    speed = tab.get_by_role("spinbutton", name="Speech Playback Speed")
    expect(speed).to_be_visible()
    speed.fill(str(SPEED))
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        tab.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def played_rates(page: Page) -> list[float]:
    page.wait_for_function(f"() => window.playedRates.length >= {SENTENCES}")
    return page.evaluate(PLAYED_RATES)


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_every_sentence_read_aloud_plays_at_the_chosen_speed(page_for, chat_for, make_user):
    """In a side chat: open-webui/open-webui#34014."""
    owner = make_user()
    set_speed(page_for(owner))
    text = reply_text()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "lamps?"}, {"role": "assistant", "content": text}],
        )
    chat = chat_for(owner, chat_id=chat_id)
    chat.page.evaluate(RECORD_RATES)
    expect(chat.conversation.get_by_text(text)).to_be_visible()

    chat.button("Read Aloud").click()

    assert played_rates(chat.page)[:SENTENCES] == [SPEED] * SENTENCES


# one model: auto-playback reads the answer that ends, however many models were asked
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_every_sentence_of_an_auto_played_reply_plays_at_the_chosen_speed(
    page_for, chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34024."""
    account = make_user()
    page = page_for(account)
    set_speed(page)
    switch = page.locator("#tab-audio").get_by_role("switch", name="Auto-Playback Response")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_be_checked()
    question = "which lamps burn?"
    text = reply_text()
    upstream.queue(reply.text(text, match=reply.answering(question)))
    chat = chat_for(account)
    chat.page.evaluate(RECORD_RATES)

    send(chat, question)
    expect_reply(chat, text)

    assert played_rates(chat.page)[:SENTENCES] == [SPEED] * SENTENCES, (
        f"the {chat.place} did not read its answer aloud at the chosen speed"
    )
