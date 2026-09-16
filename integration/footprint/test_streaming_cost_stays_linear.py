"""Guard: the server-side cost of a streamed reply grows linearly with its length.

Same contract as `unit/footprint/test_payload_reference_sharing.py`, seen from outside: every
chunk of a streamed reply passes through the filter, output-item and resume-store code on the
server, and if any step there rebuilds the whole reply so far, the work a reply costs grows
with the square of its chunk count. Six times the chunks may cost at most twelve times the
work. The instance is shared with the memory test, which runs first and leaves its retained
plugin sources behind; that is heap, not per-chunk work, so it does not move the ratio.

What counts is the CPU the server process burns, read from the kernel around each reply, not
the wall time the client waits. Wall time carries whatever else the runner was doing: it read
under the bound twice on a ref whose backend was byte-identical to a run that read over it,
and the second of those took the 0.11.4 release gate red on 2026-09-14. Copying the reply per
delta is CPU work, so CPU is both the thing the guard is about and the stable way to see it.

Two further steps keep the divisor honest, because the short reply is small enough that its
noise decides the verdict: the cost of answering at all is measured on a 100-chunk reply and
taken off both sides, and each of those two is read twice with the cheaper reading kept. Left
in, that fixed part alone moved the measured ratio between 11 and 19 on identical code.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where every flushed delta copies the
whole reply text twice in `utils/middleware.py`: the delta is appended onto the output item's
text string, and the content parts are re-joined for the resume store in
`save_current_response_stream`. Six times the chunks cost about twenty times the work; strict
`xfail`. Unmarked: no issue filed yet.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from integration.conftest import MOCK_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

BASELINE, SHORT, LONG = 100, 4000, 24000
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


def _server_cpu_for_a_reply(instance, chunks: int, deadline_seconds: float) -> tuple[float, float]:
    before = instance.cpu_seconds()
    elapsed = _stream_a_reply(instance, chunks, deadline_seconds)
    return instance.cpu_seconds() - before, elapsed


def _cheapest_reply_cpu(instance, chunks: int, deadline_seconds: float) -> tuple[float, float]:
    """The lowest of two readings: the one least charged for whatever else ran alongside it."""
    readings = [_server_cpu_for_a_reply(instance, chunks, deadline_seconds) for _ in range(2)]
    return min(readings)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="every delta copies the reply")
def test_six_times_the_chunks_cost_at_most_twelve_times_the_work(launched_instance):
    _stream_a_reply(launched_instance, SHORT, 60)  # warm caches, first chat, first model load
    per_request, _ = _cheapest_reply_cpu(launched_instance, BASELINE, 60)
    short, short_wall = _cheapest_reply_cpu(launched_instance, SHORT, 60)
    long, _ = _server_cpu_for_a_reply(launched_instance, LONG, 20 * ALLOWED_RATIO * short_wall + 30)

    # Answering at all costs the same whatever the length, and that fixed part is the noisy
    # one: at 4000 chunks it has read anywhere from a quarter to half of the total.
    short -= per_request
    long -= per_request

    measured = (
        f"{SHORT} chunks: {short:.2f}s CPU, {LONG} chunks: {long:.2f}s CPU, "
        f"per request: {per_request:.2f}s"
    )
    print(measured)  # the reading is the point; `-s` shows it whichever way the guard lands

    assert long / short < ALLOWED_RATIO, measured
