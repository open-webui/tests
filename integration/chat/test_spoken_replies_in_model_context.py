"""Journey: what the chat model is told of the spoken side of a realtime voice call.

Since d989375b4 the server replays each earlier message's `meta` from the stored chat (and the
web client sends `model` and `meta` along) and turns the voice record a call saved
(`meta.voice`) into context for the chat model: a reply the voice model spoke on its own (its
`model` is the voice's) reaches it only as a historical voice transcript, never as an answer of
its own, and the spoken transcript of a chat answer follows that answer as one more such
message. A message without a voice record is passed on as before, its `meta` left out.

Discriminates: in a backend copy that passes messages on without reading their voice record, the
spoken reply test and the spoken answer test turn red; the third passes there as well.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.realtime_provider import VOICE_MODEL
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HISTORICAL = "[Historical voice assistant transcript; not current task status]\n"


def voice_record(*transcripts: str) -> dict:
    speech = [
        {"item_id": f"item_{number}", "transcript": text, "interrupted": False}
        for number, text in enumerate(transcripts)
    ]
    return {"voice": {"model": VOICE_MODEL, "input_item_id": "item_in", "speech": speech}}


def sent_to_chat_model(make_user, upstream, history: list[dict]) -> list[dict]:
    """What the chat model gets for a question asked after `history`, seeded as a chat."""
    question = f"and after that {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("noted", match=reply.answering(question)))
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, history)
        ask(client, question, chat_id=chat_id, parent_id=last_id, history=history)
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return [message for message in request["messages"] if message["role"] != "system"]


def test_a_reply_the_voice_spoke_on_its_own_reaches_the_chat_model_as_a_transcript(
    make_user, upstream
):
    spoken = "Hello, sailor. The tide turns at six."
    history = [
        {"role": "user", "content": "hello there"},
        {
            "role": "assistant",
            "content": spoken,
            "model": VOICE_MODEL,
            "meta": voice_record(spoken),
        },
    ]

    sent = sent_to_chat_model(make_user, upstream, history)

    assert sent[:2] == [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": HISTORICAL + spoken},
    ]


def test_a_spoken_chat_answer_is_followed_by_its_transcript(make_user, upstream):
    answer, spoken = "High tide is at six.", "High tide is at six, sailor."
    history = [
        {"role": "user", "content": "when is high tide"},
        {
            "role": "assistant",
            "content": answer,
            "model": MOCK_MODEL_ID,
            "meta": voice_record(spoken),
        },
    ]

    sent = sent_to_chat_model(make_user, upstream, history)

    assert sent[:3] == [
        {"role": "user", "content": "when is high tide"},
        {"role": "assistant", "content": answer},
        {"role": "assistant", "content": HISTORICAL + spoken},
    ]


def test_a_message_without_a_voice_record_is_passed_on_without_its_meta(make_user, upstream):
    history = [
        {"role": "user", "content": "where is the pier"},
        {
            "role": "assistant",
            "content": "The pier is north.",
            "model": MOCK_MODEL_ID,
            "meta": {"source": "almanac"},
        },
    ]

    sent = sent_to_chat_model(make_user, upstream, history)

    assert sent[:2] == [
        {"role": "user", "content": "where is the pier"},
        {"role": "assistant", "content": "The pier is north."},
    ]
