"""Journey: a voice call on the Prompt Caching page's setup keeps every request append-only.

The Prompt Caching docs page says voice mode is usually fine for the provider's prefix cache:
the voice prompt is prepended to the system message once and stays the same while the call
lasts. A person starts a call in a new chat on a model set up as the page's checklist says and
speaks several turns; every request the provider gets carries the voice prompt and repeats the
one before it byte for byte, only adding to the end. A call started in the middle of a typed chat
is the one place the voice prompt arrives late: it lands in the system message of the first
spoken turn and rewrites the prefix the typed turns built, as the page's table says.

Discriminates: passes on dev 30f3f6a8f; in backend copies, a clock value added to the model's
system prompt and the tool list shuffled per request each turn it red.
"""

from __future__ import annotations

import time

import pytest

from harness import upstream as reply
from harness.prompt_caching import (
    assert_append_only,
    cache_optimal_model,
    first_break,
    turn_off_memory_system_context,
)
from utils.cached_chat import ask, chat_requests
from utils.voice_call import start_call, wait_for_transcriptions

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

VOICE_PROMPT = "Everything you say will be spoken aloud."
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
