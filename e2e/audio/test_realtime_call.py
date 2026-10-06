"""Journey: a realtime voice call from the chat input, end to end against a voice provider.

With Call mode set to Realtime under Admin Settings > Audio, Voice mode opens the call panel,
which connects to the instance's realtime call and through it to the provider (here
`harness.realtime_provider`, heard through Chromium's fake microphone). What the provider
transcribes lands in the chat as the user's message; a request it hands on goes to the chat's
own model, whose answer shows in the chat as usual and is then spoken, its spoken transcript
saved on that reply; an answer the voice model gives itself is saved as a reply of its own.
Both survive a reload. Mute stops the microphone reaching the provider, End call closes the
panel and the provider's session, and a provider that refuses or drops the call shows its
message and closes the panel.

Discriminates: passes on the dev ebc6add67 build; in a frontend copy, a transcript that is never
added to the chat turns the transcript tests red, a spoken answer never saved turns the
metadata test red, a mute that keeps sending audio turns the mute test red and a call failure
that leaves the panel open turns the provider failure tests red.
"""

from __future__ import annotations

import time
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.realtime_provider import serving_realtime_provider, using_realtime
from utils.chat_ui import conversation, expect_reply
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def provider():
    with serving_realtime_provider() as fake:
        yield fake


@pytest.fixture
def realtime(admin, provider, e2e_instance):
    with admin.client() as client, using_realtime(client, provider):
        yield provider


def words() -> str:
    return uuid.uuid4().hex[:6]


def chat_id_of(page: Page) -> str:
    page.wait_for_url("**/c/**", timeout=TURN_TIMEOUT_MS)
    return page.url.rsplit("/c/", 1)[1].split("?")[0]


def stored_messages(actor, chat_id: str) -> list[dict]:
    with actor.client() as client:
        chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    return list(chat["history"]["messages"].values())


def wait_for_stored(actor, chat_id: str, condition, what: str) -> list[dict]:
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        messages = stored_messages(actor, chat_id)
        if condition(messages):
            return messages
        time.sleep(0.2)
    raise AssertionError(f"the chat never stored {what}: {stored_messages(actor, chat_id)}")


def spoken_of(message: dict) -> list[str]:
    voice = (message.get("meta") or {}).get("voice") or {}
    return [entry["transcript"] for entry in voice.get("speech", [])]


def test_a_spoken_question_is_answered_by_the_chat_model_and_spoken(
    voice_page_for, make_user, realtime, upstream
):
    question, answer = f"when is high tide {words()}", f"High tide is at six {words()}."
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    realtime.hears(question)
    caller = make_user()
    page = voice_page_for(caller)

    start_call(page)

    expect(conversation(page).get_by_text(question)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect_reply(page, answer)
    realtime.wait_for(lambda: answer in realtime.spoken, "the answer spoken")
    chat_id = chat_id_of(page)
    messages = wait_for_stored(
        caller,
        chat_id,
        lambda stored: any(answer in spoken_of(message) for message in stored),
        "the spoken answer",
    )
    by_role = {message["role"]: message for message in messages}
    assert by_role["user"]["content"] == question
    assert by_role["user"]["meta"]["voice"]["input_item_id"]
    assert by_role["assistant"]["content"] == answer

    page.reload()
    expect(conversation(page).get_by_text(question)).to_be_visible()
    expect_reply(page, answer)


def test_an_answer_the_voice_model_gives_itself_is_saved_as_a_reply(
    voice_page_for, make_user, realtime, upstream
):
    greeting, answer = f"hello there {words()}", f"Hello, sailor {words()}."
    realtime.hears(greeting, answers=answer)
    caller = make_user()
    page = voice_page_for(caller)

    start_call(page)

    expect(conversation(page).get_by_text(greeting)).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect_reply(page, answer)
    chat_id = chat_id_of(page)
    messages = wait_for_stored(
        caller,
        chat_id,
        lambda stored: any(message["content"] == answer for message in stored),
        "the voice model's own answer",
    )
    assert upstream.chat_requests() == [], "the chat model was asked although the voice answered"
    reply_message = next(message for message in messages if message["role"] == "assistant")
    assert spoken_of(reply_message) == [answer]

    page.reload()
    expect(conversation(page).get_by_text(greeting)).to_be_visible()
    expect_reply(page, answer)


def test_the_call_panel_connects_with_the_admins_voice_and_listens(
    voice_page_for, make_user, realtime
):
    page = voice_page_for(make_user())

    start_call(page)

    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    call = realtime.wait_for_call()
    realtime.wait_for(lambda: call.received("input_audio_buffer.append"), "microphone audio")
    assert call.session["audio"]["output"]["voice"] == "harbour"


def test_muting_stops_the_microphone_reaching_the_provider(voice_page_for, make_user, realtime):
    page = voice_page_for(make_user())
    start_call(page)
    call = realtime.wait_for_call()
    realtime.wait_for(lambda: call.received("input_audio_buffer.append"), "microphone audio")

    page.get_by_role("button", name="Mute").click()

    expect(call_status(page, "Muted")).to_be_visible()
    realtime.wait_for(lambda: call.received("input_audio_buffer.clear"), "the buffer cleared")
    sent_while_muted = len(call.received("input_audio_buffer.append"))
    time.sleep(1)
    assert len(call.received("input_audio_buffer.append")) == sent_while_muted

    page.get_by_role("button", name="Unmute").click()

    expect(call_status(page, "Listening...")).to_be_visible()
    realtime.wait_for(
        lambda: len(call.received("input_audio_buffer.append")) > sent_while_muted,
        "microphone audio after unmuting",
    )


def test_ending_the_call_closes_the_panel_and_the_provider_session(
    voice_page_for, make_user, realtime
):
    page = voice_page_for(make_user())
    start_call(page)
    call = realtime.wait_for_call()
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    page.get_by_role("button", name="End call").click()

    expect(page.get_by_role("button", name="End call")).to_have_count(0)
    realtime.wait_for(call.ended.is_set, "the call end")


def test_a_provider_refusing_the_session_shows_why_and_closes_the_panel(
    voice_page_for, make_user, realtime
):
    realtime.refuse_session = True
    page = voice_page_for(make_user())

    page.get_by_role("button", name="Voice mode").click()

    expect(page.get_by_text("Provider rejected voice configuration")).to_be_visible()
    expect(page.get_by_role("button", name="End call")).to_have_count(0)


def test_a_dropped_call_shows_why_and_closes_the_panel(voice_page_for, make_user, realtime):
    page = voice_page_for(make_user())
    start_call(page)
    call = realtime.wait_for_call()
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    call.drop()

    expect(page.get_by_text("Voice provider connection closed")).to_be_visible()
    expect(page.get_by_role("button", name="End call")).to_have_count(0)
