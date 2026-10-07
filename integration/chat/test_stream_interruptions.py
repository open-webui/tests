"""Journey: what the server keeps of a reply when the browser goes away mid-stream.

A browser tab that disconnects while the reply streams does not stop it: the reply streams to its
end and is stored whole, as a person closing the tab and opening the chat later expects.

Twin of e2e/chat/test_reply_while_away.py.

Discriminates: passes on dev ebc6add67; the disconnect test fails on a backend copy that cancels
a user's replies when their socket disconnects.
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
