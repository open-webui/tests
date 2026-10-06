"""Journey: what a realtime voice call saves on a chat message, seen over the message route.

A realtime call keeps the spoken side of a turn on the message it belongs to: the browser posts
`{"voice": {...}}` to `/api/v1/chats/{id}/messages/{message_id}`, which stores it as the
message's `meta.voice` beside whatever else the message's meta holds, leaves the content alone
and tells the owner's open tabs with a `chat:message:voice` event. A later voice save replaces
the voice part only, a content edit keeps it, and a body with neither part, an unknown message,
an oversized voice record or someone else's chat is refused.

Discriminates: in a backend copy, voice saves that replace the whole meta turn the store and
replace tests red, and the route storing no voice part turns every test that saves one red.
"""

from __future__ import annotations

import pytest

from harness.chat_history import seed_chat
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SPEECH = {
    "call_id": "call-harbour",
    "input_item_id": "item_1",
    "model": "gpt-realtime-harbour",
    "speech": [{"item_id": "item_2", "transcript": "High tide is at six.", "interrupted": False}],
}


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def answered_chat(owner) -> tuple[str, str]:
    """A question and its answer, the answer carrying meta of its own; (chat id, answer id)."""
    with owner.client() as client:
        return seed_chat(
            client,
            [
                {"role": "user", "content": "when is high tide?"},
                {
                    "role": "assistant",
                    "content": "High tide is at six.",
                    "meta": {"source": "almanac"},
                },
            ],
        )


def save_voice(actor, chat_id: str, message_id: str, body: dict):
    with actor.client() as client:
        return client.post(f"/api/v1/chats/{chat_id}/messages/{message_id}", json=body)


def stored_message(actor, chat_id: str, message_id: str) -> dict:
    with actor.client() as client:
        chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    return chat["history"]["messages"][message_id]


def test_a_voice_record_is_stored_beside_the_messages_meta(owner, answered_chat):
    chat_id, answer_id = answered_chat

    saved = save_voice(owner, chat_id, answer_id, {"voice": SPEECH})

    assert saved.status_code == 200, saved.text
    message = stored_message(owner, chat_id, answer_id)
    assert message["meta"] == {"source": "almanac", "voice": SPEECH}
    assert message["content"] == "High tide is at six."


def test_a_later_voice_save_replaces_only_the_voice_part(owner, answered_chat):
    chat_id, answer_id = answered_chat
    save_voice(owner, chat_id, answer_id, {"voice": SPEECH})
    interrupted = {**SPEECH, "speech": [{**SPEECH["speech"][0], "interrupted": True}]}

    saved = save_voice(owner, chat_id, answer_id, {"voice": interrupted})

    assert saved.status_code == 200, saved.text
    assert stored_message(owner, chat_id, answer_id)["meta"] == {
        "source": "almanac",
        "voice": interrupted,
    }


def test_a_content_edit_keeps_the_voice_record(owner, answered_chat):
    chat_id, answer_id = answered_chat
    save_voice(owner, chat_id, answer_id, {"voice": SPEECH})

    edited = save_voice(owner, chat_id, answer_id, {"content": "High tide is at seven."})

    assert edited.status_code == 200, edited.text
    message = stored_message(owner, chat_id, answer_id)
    assert message["content"] == "High tide is at seven."
    assert message["meta"]["voice"] == SPEECH


def test_the_owners_open_tabs_hear_of_the_voice_record(owner, answered_chat):
    chat_id, answer_id = answered_chat
    with connected(owner) as tab:
        save_voice(owner, chat_id, answer_id, {"voice": SPEECH})
        event = tab.wait_for(chat_id, "chat:message:voice")

    assert event["data"] == {"chat_id": chat_id, "message_id": answer_id, "voice": SPEECH}


def test_a_save_with_neither_content_nor_voice_is_refused(owner, answered_chat):
    chat_id, answer_id = answered_chat

    saved = save_voice(owner, chat_id, answer_id, {})

    assert saved.status_code == 400
    assert saved.json()["detail"] == "No message changes supplied"


def test_a_voice_record_for_an_unknown_message_is_refused(owner, answered_chat):
    chat_id, _ = answered_chat

    saved = save_voice(owner, chat_id, "no-such-message", {"voice": SPEECH})

    assert saved.status_code == 404
    with owner.client() as client:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    assert "no-such-message" not in messages


def test_an_oversized_voice_record_is_refused(owner, answered_chat):
    chat_id, answer_id = answered_chat
    oversized = {**SPEECH, "speech": [{"item_id": "item_2", "transcript": "x" * 100_001}]}

    saved = save_voice(owner, chat_id, answer_id, {"voice": oversized})

    assert saved.status_code == 400
    assert saved.json()["detail"] == "Voice metadata is too large"
    assert stored_message(owner, chat_id, answer_id)["meta"] == {"source": "almanac"}


def test_a_voice_record_on_someone_elses_chat_is_refused(owner, answered_chat, make_user):
    chat_id, answer_id = answered_chat

    saved = save_voice(make_user(), chat_id, answer_id, {"voice": SPEECH})

    assert saved.status_code == 401
    assert stored_message(owner, chat_id, answer_id)["meta"] == {"source": "almanac"}
