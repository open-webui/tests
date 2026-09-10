"""Guard: the server-side cost of a streamed reply grows linearly with its length.

Same contract as `unit/footprint/test_payload_reference_sharing.py`, seen from outside: every
chunk of a streamed reply passes through the filter, output-item and resume-store code on the
server, and if any step there rebuilds the whole reply so far, the time to finish a reply
grows with the square of its chunk count. The mock upstream streams its chunks as fast as the
server takes them, so the wall time from request to the saved assistant message being done is
server work; the poll in between asks for the chat's task list, which does not grow. Six
times the chunks may cost at most twelve times the wall time. The instance is shared with the
memory test, which runs first and leaves its retained plugin sources behind; that is heap,
not per-chunk work, so it does not move the ratio.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where every flushed delta copies the
whole reply text twice in `utils/middleware.py`: the delta is appended onto the output item's
text string, and the content parts are re-joined for the resume store in
`save_current_response_stream`. Six times the chunks cost about twenty times the time; strict
`xfail`. Unmarked: no issue filed yet.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from integration.conftest import MOCK_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

SHORT, LONG = 4000, 24000
CHUNK_TEXT = "lorem-ipsum-dolor-sit-amet-" * 4  # no trailing space: the saved reply is stripped
ALLOWED_RATIO = 2 * LONG / SHORT


def _stream_a_reply(instance, chunks: int, deadline_seconds: float) -> float:
    instance.upstream.reset(mode="stream", chunks=chunks, chunk_text=CHUNK_TEXT)
    assistant_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
    with instance.client() as client:
        started = time.perf_counter()
        accepted = client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
                "parent_id": None,
                "id": assistant_id,
                "user_message": {"id": user_id, "role": "user", "content": "go"},
                "session_id": "integration",
                "background_tasks": {
                    "title_generation": False,
                    "tags_generation": False,
                    "follow_up_generation": False,
                },
            },
        )
        accepted.raise_for_status()
        chat_id = accepted.json()["chat_id"]
        _wait_until_done(client, chat_id, deadline_seconds)
        elapsed = time.perf_counter() - started
        chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    message = chat["history"]["messages"][assistant_id]
    if not message.get("done") or message["content"].count(CHUNK_TEXT) != chunks - 1:
        pytest.fail("the reply is incomplete")
    return elapsed


def _wait_until_done(client: httpx.Client, chat_id: str, deadline_seconds: float) -> None:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if not client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]:
            return
        time.sleep(0.02)
    pytest.fail("the streamed reply never finished")


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="every delta copies the reply")
def test_six_times_the_chunks_cost_at_most_twelve_times_the_time(launched_instance):
    _stream_a_reply(launched_instance, SHORT, 60)  # warm caches, first chat, first model load
    short = _stream_a_reply(launched_instance, SHORT, 60)
    long = _stream_a_reply(launched_instance, LONG, 20 * ALLOWED_RATIO * short + 30)

    assert long / short < ALLOWED_RATIO, f"{SHORT} chunks: {short:.2f}s, {LONG} chunks: {long:.2f}s"
