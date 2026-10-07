"""Journey: a slow or stalled provider as the user sees it, under the connection timeouts.

`AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST` bounds how long the model list waits for each connection,
`AIOHTTP_CLIENT_STREAM_IDLE_TIMEOUT` how long a streamed answer may go silent and
`AIOHTTP_CLIENT_TIMEOUT` how long a whole answer may take (docs: env-configuration). On an
instance booted with all three set low, a connection slower than the list's limit leaves the
selector with the other models and without its own. An answer that goes silent mid-stream is cut
off with the timeout shown under the text that arrived; an answer that keeps streaming runs past
the idle limit and is cut at the total one. The provider is a second connection that streams its
pieces with the pauses each test sets.

Discriminates: passes on the dev ebc6add67 build except the two tests below; in a backend copy,
the model list's timeout dropped turns the list test red (the selector waits for the slow
connection), the idle timeout dropped from streamed requests turns the stall test red (the answer
finishes after the pause) and the idle limit applied as the total turns the steady test red. Red
on dev, real bugs: after a reload an answer cut off by a timeout has lost the text that arrived
(the error is stored, the streamed text never is; Stop keeps it), and an answer cut at the total
limit says only "Error submitting message" (the timeout's empty message is stored as the error).
"""

from __future__ import annotations

import json
import re
import time
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import Listener, json_answer
from harness.second_provider import OPENAI_CONFIG, attach
from utils.chat_ui import chat_input, expect_reply, last_reply, send, stop_button
from utils.model_selector import model_options, select_model

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

IDLE_LIMIT_SECONDS = 2
TOTAL_LIMIT_SECONDS = 8
TIMEOUTS = {
    "AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST": "1",
    "AIOHTTP_CLIENT_STREAM_IDLE_TIMEOUT": str(IDLE_LIMIT_SECONDS),
    "AIOHTTP_CLIENT_TIMEOUT": str(TOTAL_LIMIT_SECONDS),
}
SLOW_LISTING_SECONDS = 6
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
MODEL = "slow.talker"


@pytest.fixture
def impatient(instance_with):
    return instance_with(TIMEOUTS)


@pytest.fixture
def talker(impatient, preserve, listener) -> Listener:
    """The listener as a connection every account may chat with, as `slow.talker`."""
    preserve(OPENAI_CONFIG, on=impatient)
    with admin_of(impatient).client() as client:
        attach(client, listener, "talker", prefix_id="slow")
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={"id": MODEL, "name": MODEL, "access_grants": [EVERYONE_READS]},
        )
    assert shared.status_code == 200, shared.text
    return listener


def stream_with_pauses(listener: Listener, pieces: list[tuple[float, str]]) -> None:
    """Answer every chat with `pieces`, each sent after its pause in seconds."""

    def answer(_request):
        def body():
            for pause, text in pieces:
                time.sleep(pause)
                delta = {"index": 0, "delta": {"content": text}, "finish_reason": None}
                chunk = {"object": "chat.completion.chunk", "choices": [delta]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return 200, {"Content-Type": "text/event-stream"}, body()

    listener.route("POST", "/v1/chat/completions", answer)


def ask_until_cut_off(page: Page, first_text: str) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    select_model(page, MODEL)
    send(page, f"Tell me a story, {uuid.uuid4().hex[:6]}.")
    expect_reply(page, first_text)
    expect(stop_button(page)).to_be_hidden(timeout=(TOTAL_LIMIT_SECONDS + 10) * 1000)


# --------------------------------------------------------------------------- model list


def test_a_connection_slower_than_the_list_limit_leaves_the_other_models(
    page_for, impatient, talker
):
    def slow_models(_request):
        time.sleep(SLOW_LISTING_SECONDS)
        return json_answer({"object": "list", "data": [{"id": "sluggish"}]})

    talker.route("GET", "/v1/models", slow_models)
    page = page_for(create_user(impatient))
    started = time.monotonic()
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(1)
    waited = time.monotonic() - started

    expect(model_options(page, "slow.sluggish")).to_have_count(0)
    assert talker.requests_to("/v1/models"), "the slow connection was never asked"
    assert waited < SLOW_LISTING_SECONDS - 1, f"the selector waited {waited:.1f}s for the list"


# --------------------------------------------------------------------------- idle limit


def test_an_answer_that_goes_silent_is_cut_off_with_the_timeout_shown(page_for, impatient, talker):
    stream_with_pauses(talker, [(0, "Once upon a time"), (IDLE_LIMIT_SECONDS + 3, " THE END")])
    page = page_for(create_user(impatient))

    ask_until_cut_off(page, "Once upon a time")

    expect(last_reply(page)).to_contain_text("Timeout on reading data from socket")
    expect(last_reply(page)).not_to_contain_text("THE END")


def test_the_text_that_arrived_before_a_timeout_is_still_there_after_a_reload(
    page_for, impatient, talker
):
    stream_with_pauses(talker, [(0, "Once upon a time"), (IDLE_LIMIT_SECONDS + 3, " THE END")])
    page = page_for(create_user(impatient))
    ask_until_cut_off(page, "Once upon a time")

    page.reload()
    expect(last_reply(page)).to_contain_text("Timeout on reading data from socket")
    shown = last_reply(page).inner_text()
    assert "Once upon a time" in shown, f"after a reload the cut-off answer reads only {shown!r}"


# --------------------------------------------------------------------------- total limit


def test_a_steady_answer_runs_past_the_idle_limit_and_stops_at_the_total_limit(
    page_for, impatient, talker
):
    pieces = [(1, f" part{number}") for number in range(TOTAL_LIMIT_SECONDS + 4)]
    stream_with_pauses(talker, pieces)
    page = page_for(create_user(impatient))

    ask_until_cut_off(page, "part0")

    expect(last_reply(page)).to_contain_text(f"part{IDLE_LIMIT_SECONDS + 2}")
    expect(last_reply(page)).not_to_contain_text(f"part{TOTAL_LIMIT_SECONDS + 3}")


def test_an_answer_cut_at_the_total_limit_says_it_timed_out(page_for, impatient, talker):
    pieces = [(1, f" part{number}") for number in range(TOTAL_LIMIT_SECONDS + 4)]
    stream_with_pauses(talker, pieces)
    page = page_for(create_user(impatient))

    ask_until_cut_off(page, "part0")

    shown = last_reply(page).inner_text()
    assert re.search(r"time ?out|timed out", shown, re.IGNORECASE), (
        f"the answer was cut at the total limit and reads {shown!r}, naming no timeout"
    )
