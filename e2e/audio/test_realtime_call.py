"""Journey: a realtime voice call from the chat input, end to end against a voice provider.

With Call mode set to Realtime under Admin Settings > Audio, Voice mode opens the call panel,
which connects to the instance's realtime call and through it to the provider (here
`harness.realtime_provider`, heard through Chromium's fake microphone). A call in an existing
chat first tells the provider the conversation so far. What the provider transcribes lands in
the chat as the user's message; a request it hands on goes to the chat's own model, whose answer
shows in the chat as usual and is then spoken, its spoken transcript saved on that reply; an
answer the voice model gives itself is saved as a reply of its own. Both survive a reload. A
turn that cannot be transcribed is asked again and adds no message of the user, and stopping a
spoken answer tells the provider where playback stopped and saves the answer as interrupted. While
the chat model works the panel shows Thinking... and its Stop cancels the request and tells the
provider so. Mute stops the microphone reaching the provider, hiding the side panel keeps the
call on and the voice button, reading Return to call while it runs, brings it back (since
fa4c7fe5e there is no Review in chat button for this), End call closes the panel and the
provider's session, and a provider that refuses or drops the call shows its message and closes
the panel. Realtime calls open on the browser's speech-to-text engine too, which the standard
call refuses, and without the call permission there is no Voice mode at all.

The stop test is red now and then with Postgres and Redis on dev 0f5a58f5f: the panel offers Stop as
soon as the request to the chat model starts, but a press before the server has accepted that
request is dropped without a word and the chat model answers on. The slower backends widen that
moment enough for the test's press to land in it.

Discriminates: passes on the dev 0f5a58f5f build. In one frontend copy a spoken answer never
saved turns the chat model test red, the voice model's own answer never added turns that test
red, a mute that keeps sending audio, End call leaving the call connected, a failure leaving the
panel open, an empty conversation sent to the provider, a panel Stop that does nothing and Voice
mode refusing realtime calls on the browser's speech-to-text each turn their test red, while the
panel test stays green. In another, closing the panel ending the call turns the return test red,
a failed transcription left unanswered turns that test red and an interruption that never tells
the provider where playback stopped turns the interrupt test red, while the chat model test stays
green. In a third, a voice button that still reads Voice mode and starts a new call while one
runs turns the return test red.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.realtime_provider import SAMPLE_RATE, serving_realtime_provider, using_realtime
from utils.chat_ui import chat_input, conversation, expect_reply
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_ANSWER_SECONDS = 6


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


def call_buttons(page: Page):
    """The call panel's row of buttons; the chat input has a Stop button of its own."""
    return page.get_by_role("button", name="End call").locator("xpath=..")


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


def test_a_call_in_an_existing_chat_tells_the_provider_the_conversation(
    voice_page_for, make_user, realtime
):
    question, answer = f"where is the pier {words()}", f"The pier is north {words()}."
    caller = make_user()
    with caller.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        )
    page = voice_page_for(caller)
    page.goto(f"/c/{chat_id}")
    expect_reply(page, answer)

    start_call(page)

    call = realtime.wait_for_call()
    realtime.wait_for(lambda: call.received("conversation.item.create"), "the conversation")
    told = [event["item"] for event in call.received("conversation.item.create")]
    assert [(item["role"], item["content"][0]["text"]) for item in told] == [
        ("user", question),
        ("assistant", answer),
    ]


def test_hiding_the_call_keeps_it_running_and_return_to_call_brings_it_back(
    voice_page_for, make_user, realtime
):
    page = voice_page_for(make_user())
    start_call(page)
    call = realtime.wait_for_call()
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    page.get_by_role("navigation").get_by_role("button", name="Controls").click()

    expect(page.get_by_role("button", name="End call")).to_have_count(0)
    assert not call.ended.wait(1), "hiding the call panel ended the call"
    expect(page.get_by_role("button", name="Voice mode")).to_have_count(0)
    page.get_by_role("button", name="Return to call").click()
    expect(page.get_by_role("button", name="End call")).to_be_visible()
    expect(call_status(page, "Listening...")).to_be_visible()
    assert len(realtime.calls) == 1, "returning to the call opened a new one"


def test_stopping_the_chat_model_tells_the_provider_it_was_cancelled(
    voice_page_for, make_user, realtime, upstream
):
    question = f"read me the whole almanac {words()}"
    pages = [f"page {number} " for number in range(60)]
    upstream.queue(reply.text(pages, chunk_delay=0.5, match=reply.answering(question)))
    realtime.hears(question)
    page = voice_page_for(make_user())
    start_call(page)
    expect(call_status(page, "Thinking...")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    call_buttons(page).get_by_role("button", name="Stop", exact=True).click()

    call = realtime.calls[-1]
    realtime.wait_for(lambda: call.results, "the function result")
    assert list(call.results.values()) == [
        "The backend request was cancelled. Completed actions have not been undone."
    ]
    outputs = [
        json.loads(event["item"]["output"])
        for event in call.received("conversation.item.create")
        if event["item"]["type"] == "function_call_output"
    ]
    assert [output["status"] for output in outputs] == ["cancelled"]
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)


def test_realtime_calls_work_with_the_browsers_speech_to_text(
    voice_page_for, make_user, admin, realtime
):
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        current["stt"]["ENGINE"] = "web"
        saved = client.post(AUDIO_CONFIG[1], json=current)
    assert saved.status_code == 200, saved.text
    page = voice_page_for(make_user())

    start_call(page)

    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert realtime.wait_for_call()


def test_without_the_call_permission_there_is_no_voice_mode(
    voice_page_for, make_user, admin, realtime, preserve
):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["call"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    page = voice_page_for(make_user())

    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("button", name="Voice mode")).to_have_count(0)


def test_a_turn_that_cannot_be_transcribed_is_asked_again_with_no_message_of_the_user(
    voice_page_for, make_user, realtime
):
    realtime.mishears()
    page = voice_page_for(make_user())

    start_call(page)

    realtime.wait_for(lambda: realtime.spoken, "a spoken status")
    assert "I could not transcribe that. Please repeat it." in realtime.spoken[0]
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect(conversation(page).locator(".chat-user")).to_have_count(0)


def test_interrupting_the_spoken_answer_cuts_it_short_and_saves_it_as_interrupted(
    voice_page_for, make_user, realtime
):
    realtime.speech = b"\x00\x00" * (SAMPLE_RATE * LONG_ANSWER_SECONDS)
    greeting, answer = f"tell me a story {words()}", f"Once upon a tide {words()}."
    realtime.hears(greeting, answers=answer)
    caller = make_user()
    page = voice_page_for(caller)
    start_call(page)
    expect(call_status(page, "Tap to interrupt")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    page.get_by_role("button", name="Stop speaking").click()

    expect(call_status(page, "Listening...")).to_be_visible()
    call = realtime.calls[-1]
    realtime.wait_for(lambda: call.received("conversation.item.truncate"), "the answer cut")
    [cut] = call.received("conversation.item.truncate")
    assert 0 <= cut["audio_end_ms"] < LONG_ANSWER_SECONDS * 1000
    messages = wait_for_stored(
        caller,
        chat_id_of(page),
        lambda stored: any(
            entry.get("interrupted")
            for message in stored
            for entry in ((message.get("meta") or {}).get("voice") or {}).get("speech", [])
        ),
        "the answer marked as interrupted",
    )
    reply_message = next(message for message in messages if message["role"] == "assistant")
    assert reply_message["content"] == answer
