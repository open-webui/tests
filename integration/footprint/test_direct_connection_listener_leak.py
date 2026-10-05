"""Regression: a direct-connection streaming request left a listener behind on every bad exit.

open-webui 0.11.4 PR #29509 (commit `aa48106fc`): a direct-connection streaming request
registered a per-request listener before asking the browser to start the completion, and that
listener was only removed once the response had been fully streamed. Every other way the
request could end left it behind for the life of the process: the tab refusing, an empty or
malformed acknowledgement, no live tab at all, the chat stopped while the tab had not answered.
A client that kept hitting a failing direct connection grew the server's listener table without
bound. The fix removes the listener on every exit path.

Each test sends its chats through the real route with a tab from `harness.direct_connection`
and asks the probe of `harness.process_probe` how long every module-level container of the
server is before and after; the leak is one entry per request, far below the process size. A
batch of non-streamed chats, which never register a listener, is the reference for what every
chat leaves behind anyway.

The finished stream test is red on dev b859124f9 on purpose (open-webui/open-webui#31953): since
24e30d1cb the socket router checks the tab's session token again for every event the tab sends, so
the reply's pieces can overtake each other while those checks run and the streamed reply comes back
scrambled or empty. It passes on dev 015dbc861 and on b859124f9 with that check taken back out of
the router.

Twin of unit/footprint/test_direct_connection_listener_leak.py.

Discriminates: passes on dev ef67cc3fa, fails with the fix's `EVENT_QUEUES.pop` calls removed
from the raising and refused paths (every failing case grows the queue table by one entry per
chat); the finished stream passes on both.
"""

from __future__ import annotations

import threading
import time

import pytest

from harness.chat import send_message, wait_for_reply
from harness.direct_connection import answering, chunk_line, direct_model
from harness.process_probe import grown, probing

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MODEL = "my-own-model"
CHATS = 10
COMPLETION = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "whole"}}]}


def _send(client, session_id: str, content: str, **extra):
    return send_message(
        client,
        content,
        model=MODEL,
        model_item=direct_model(MODEL),
        session_id=session_id,
        **extra,
    )


def _chat_until_done(client, session_id: str, **extra) -> dict:
    return wait_for_reply(client, _send(client, session_id, "hello", **extra))


def _retained_beyond_reference(probe, client, tab, send_batch) -> dict[str, int]:
    """What a batch of streamed chats left behind that as many non-streamed ones did not.

    Every finished chat leaves its own trace (see test_unbounded_process_state.py), so a
    batch of non-streamed chats, which never register a listener, is the reference.
    """
    start = probe.containers()
    for _ in range(CHATS):
        tab.complete(COMPLETION)
        assert _chat_until_done(client, tab.session_id, stream=False)["content"] == "whole"
    middle = probe.containers()
    send_batch()
    end = probe.containers()
    reference = grown(start, middle)
    return {name: n for name, n in grown(middle, end, by=CHATS).items() if name not in reference}


def _assert_nothing_retained(retained: dict[str, int]) -> None:
    assert not retained, (
        f"{CHATS} failed direct-connection chats left an entry each in {retained}: the stream "
        "listener outlives its request, so a failing connection grows the server without "
        "bound (#29509)"
    )


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(("refuse", {"error": {"message": "invalid api key"}}), id="refused"),
        pytest.param(("complete", {}), id="empty_acknowledgement"),
        pytest.param(("complete", "not an acknowledgement"), id="malformed_acknowledgement"),
    ],
)
def test_a_failed_stream_request_leaves_no_listener_behind(admin, make_user, answer):
    person = make_user()

    def send_batch():
        for _ in range(CHATS):
            tab.answers.append(answer)
            message = _chat_until_done(client, tab.session_id)
            assert message.get("error"), f"the tab's {answer[0]} answer did not fail: {message}"

    with probing(admin) as probe, answering(person) as tab, person.client() as client:
        retained = _retained_beyond_reference(probe, client, tab, send_batch)

    _assert_nothing_retained(retained)


def test_a_chat_without_a_live_tab_leaves_no_listener_behind(admin, make_user):
    person = make_user()

    def send_batch():
        for _ in range(CHATS):
            message = _chat_until_done(client, "no-such-session")
            assert "Client session disconnected" in str(message.get("error")), message

    with probing(admin) as probe, answering(person) as tab, person.client() as client:
        retained = _retained_beyond_reference(probe, client, tab, send_batch)

    _assert_nothing_retained(retained)


def test_a_chat_stopped_before_the_tab_answers_leaves_no_listener_behind(admin, make_user):
    person = make_user()
    release = threading.Event()

    def send_batch():
        for round_number in range(CHATS):
            tab.hold(release)
            turn = _send(client, tab.session_id, "hello")
            _wait_until(lambda: len(tab.requests) > CHATS + round_number, "the tab is asked")
            client.post(f"/api/tasks/chat/{turn.chat_id}/stop").raise_for_status()
            _wait_until(lambda: not _running(client, turn.chat_id), "the chat stops")

    with probing(admin) as probe, answering(person) as tab, person.client() as client:
        try:
            retained = _retained_beyond_reference(probe, client, tab, send_batch)
        finally:
            release.set()

    _assert_nothing_retained(retained)


def test_a_finished_stream_leaves_no_listener_behind(admin, make_user):
    """Control: the clean path removed its listener before the fix too."""
    person = make_user()

    def send_batch():
        for _ in range(CHATS):
            tab.stream(chunk_line({"content": "hi"}), chunk_line({}, "stop"))
            reply = _chat_until_done(client, tab.session_id)["content"]
            assert reply == "hi", f"the streamed reply came back as {reply!r} (#31953)"

    with probing(admin) as probe, answering(person) as tab, person.client() as client:
        retained = _retained_beyond_reference(probe, client, tab, send_batch)

    _assert_nothing_retained(retained)


def _running(client, chat_id: str) -> list[str]:
    return client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]


def _wait_until(condition, what: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting until {what}")
        time.sleep(0.05)
