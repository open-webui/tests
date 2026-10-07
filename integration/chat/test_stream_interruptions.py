"""Journey: what the server keeps of a reply when the provider or the browser goes away mid-stream.

A provider that drops the connection after three pieces leaves a reply the server stores as
finished, with the error and with the three pieces the browser already showed. A browser tab
that disconnects while the reply streams does not stop it: the reply streams to its end and is
stored whole, as a person closing the tab and opening the chat later expects.

The provider test is red on dev ebc6add67: the stream's progress lives in memory until the reply
finishes, and the error path stores only the error, so the stored reply has lost the pieces.

Twin of e2e/chat/test_provider_drop_mid_reply.py and e2e/chat/test_reply_while_away.py.

Discriminates: passes on dev ebc6add67 except the provider test, which passes on a backend copy
that saves the output on a stream error; the disconnect test fails on a backend copy that
cancels a user's replies when their socket disconnects.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.inflight import SLOW_PIECES
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

WHOLE_REPLY = "".join(SLOW_PIECES).strip()


def _stored_text(message: dict) -> str:
    """The reply's text as the chat stores it, from its output items or its content."""
    texts = [
        part.get("text", "")
        for item in message.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    ]
    return "".join(texts) or message.get("content") or ""


def test_a_reply_the_provider_dropped_is_stored_with_what_arrived(make_user, upstream):
    prompt = "tell me the story of the lighthouse"
    upstream.queue(
        reply.text(SLOW_PIECES, chunk_delay=0.1, hang_up_after=3, match=reply.answering(prompt))
    )
    with make_user().client() as client:
        stored = wait_for_reply(client, send_message(client, prompt))

    assert "payload is not completed" in str(stored.get("error"))
    assert "part-0 part-1 part-2" in _stored_text(stored), (
        f"the pieces sent before the provider went down are not stored: {stored}"
    )
    assert "part-3" not in _stored_text(stored)


def test_a_reply_streams_to_its_end_after_the_tab_disconnects(make_user, upstream):
    account = make_user()
    prompt = "tell me a long story"
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.1, match=reply.answering(prompt)))
    with account.client() as client:
        with connected(account) as tab:
            turn = send_message(client, prompt, session_id=tab.client.get_sid("/"))
            tab.wait_for(turn.chat_id, "chat:completion")
        stored = wait_for_reply(client, turn)

    assert _stored_text(stored) == WHOLE_REPLY
