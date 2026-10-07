"""Journey: a voice call on the Prompt Caching page's setup keeps every request append-only.

The Prompt Caching docs page says voice mode is usually fine for the provider's prefix cache:
the voice prompt is prepended to the system message once and stays the same while the call
lasts. A person starts a call in a new chat on a model set up as the page's checklist says and
speaks several turns; every request the provider gets carries the voice prompt and repeats the
one before it byte for byte, only adding to the end. A call started in the middle of a typed chat
is the one place the voice prompt arrives late: it lands in the system message of the first
spoken turn and rewrites the prefix the typed turns built, as the page's table says.

A realtime call (Call mode Realtime, the voice provider in `harness.realtime_provider`) adds no
voice prompt: each question the voice hands on goes to the chat model as an ordinary turn, and
what the voice spoke, its answer read out or one it gave itself, follows as a historical voice
transcript. Mid-call the requests only append, through a tool call, an answer the voice gives
itself and a message typed while the call runs, in a new chat and in one with typed turns
before the call.

Discriminates: passes on dev 30f3f6a8f. In backend copies, a clock value added to the model's
system prompt turned both tests red (the call started mid-chat because the typed turns before it
stopped appending); on dev 176d31d1d the tool list shuffled per request turned the first red too.
The realtime tests pass on dev 4b9d31b39; a backend copy of it that adds a clock value to every
message carrying a voice record turns both red.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.prompt_caching import (
    assert_append_only,
    cache_optimal_model,
    first_break,
    turn_off_memory_system_context,
)
from harness.realtime_provider import serving_realtime_provider, using_realtime
from utils.cached_chat import ask, called_tools, calling, chat_requests
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call, wait_for_transcriptions

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

VOICE_PROMPT = "Everything you say will be spoken aloud."
HISTORICAL = "[Historical voice assistant transcript; not current task status]\n"
TURNS = 3


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def answered_turns(upstream, count: int, timeout: float = 60.0) -> list[dict]:
    """The first `count` requests of the call, once the reply to the last has been asked for."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        requests = [body for body in upstream.chat_requests() if body.get("stream")]
        if len(requests) >= count:
            return requests[:count]
        time.sleep(0.2)
    raise AssertionError(f"the call reached the model {len(requests)} times, not {count}")


def test_a_voice_call_only_appends(
    voice_page_for, make_user, cached_setup, speech_engine, upstream
):
    page = voice_page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    start_call(page)
    wait_for_transcriptions(speech_engine, TURNS, timeout=60.0)

    requests = answered_turns(upstream, TURNS)
    assert all(VOICE_PROMPT in request["messages"][0]["content"] for request in requests)
    assert [message["role"] for message in requests[-1]["messages"]][-3:] == [
        "user",
        "assistant",
        "user",
    ]
    assert_append_only(requests)


def test_a_voice_call_started_in_a_typed_chat_rewrites_the_prefix(
    voice_page_for, make_user, cached_setup, speech_engine, upstream
):
    page = voice_page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(page, upstream, "when is high tide?", reply.text("At noon."))
    typed = chat_requests(upstream)
    assert VOICE_PROMPT not in typed[0]["messages"][0]["content"]
    assert_append_only(typed)

    start_call(page)
    wait_for_transcriptions(speech_engine, 1, timeout=60.0)
    requests = answered_turns(upstream, len(typed) + 1)

    assert VOICE_PROMPT in requests[-1]["messages"][0]["content"]
    broken = first_break(requests)
    assert broken is not None, "the voice prompt left the prefix append-only"
    assert "messages[0] (system)" in broken, broken


@pytest.fixture
def realtime(admin, e2e_instance):
    with serving_realtime_provider() as provider, admin.client() as client:
        with using_realtime(client, provider):
            yield provider


def chat_id_of(page: Page) -> str:
    page.wait_for_url("**/c/**", timeout=TURN_TIMEOUT_MS)
    return page.url.rsplit("/c/", 1)[1].split("?")[0]


def wait_for_spoken_record(actor, page: Page, answer: str) -> None:
    """Wait until `answer` was spoken and its transcript saved on the chat, as a person hears it."""
    chat_id = chat_id_of(page)
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        with actor.client() as client:
            messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        speech = [
            entry["transcript"]
            for message in messages.values()
            for entry in ((message.get("meta") or {}).get("voice") or {}).get("speech", [])
        ]
        if answer in speech:
            return
        time.sleep(0.2)
    raise AssertionError(f"the call never saved the spoken answer {answer!r}")


def speak(page: Page, caller, realtime, upstream, question: str, *replies) -> None:
    """Say `question` in the call; the chat model answers with `replies` and the voice speaks it."""
    for scripted in replies:
        scripted.match = reply.answering(question)
        upstream.queue(scripted)
    realtime.hears(question)
    answer = replies[-1].content
    realtime.wait_for(lambda: answer in realtime.spoken, f"{answer!r} spoken", timeout=30.0)
    wait_for_spoken_record(caller, page, answer)


def test_a_realtime_call_only_appends(voice_page_for, make_user, cached_setup, realtime, upstream):
    caller = make_user()
    page = voice_page_for(caller)
    page.goto(f"/?models={cached_setup.id}")
    start_call(page)
    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    speak(page, caller, realtime, upstream, "when does the ferry leave", reply.text("At 06:40."))
    speak(
        page,
        caller,
        realtime,
        upstream,
        "what time is it now",
        calling("get_current_timestamp", {}),
        reply.text("Just after six."),
    )
    realtime.hears("thank you", answers="You are welcome.")
    realtime.wait_for(lambda: "You are welcome." in realtime.spoken, "the voice's own answer")
    wait_for_spoken_record(caller, page, "You are welcome.")
    # the call panel covers the chat input until it is hidden
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    ask(page, upstream, "is pier 7 covered?", reply.text("Pier 7 is covered."))
    speak(page, caller, realtime, upstream, "and is it heated", reply.text("It is not heated."))

    requests = chat_requests(upstream)
    assert len(requests) == 5, [request["messages"][-1] for request in requests]
    assert called_tools(requests[-1]) == {"get_current_timestamp"}
    told = [message["content"] for message in requests[-1]["messages"]]
    assert {HISTORICAL + "At 06:40.", HISTORICAL + "You are welcome."} <= set(told)
    assert VOICE_PROMPT not in told[0]
    assert_append_only(requests)


def test_a_realtime_call_in_a_typed_chat_only_appends(
    voice_page_for, make_user, cached_setup, realtime, upstream
):
    caller = make_user()
    page = voice_page_for(caller)
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(page, upstream, "when is high tide?", reply.text("At noon."))
    start_call(page)
    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)

    speak(page, caller, realtime, upstream, "and low tide", reply.text("At six."))
    speak(page, caller, realtime, upstream, "is the harbour open", reply.text("It is open."))
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    ask(page, upstream, "until when?", reply.text("Until dusk."))
    speak(page, caller, realtime, upstream, "thanks, and tomorrow", reply.text("Same hours."))

    requests = chat_requests(upstream)
    assert len(requests) == 6, [request["messages"][-1] for request in requests]
    assert HISTORICAL + "At six." in [message["content"] for message in requests[-1]["messages"]]
    assert_append_only(requests)
