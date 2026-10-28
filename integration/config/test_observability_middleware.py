"""Regression: what the server reports about a request, in headers, the audit log and its log.

open-webui 0.11.0 fixes, each read where an operator reads it:

* `X-Process-Time` (#27368): the elapsed time was truncated with `int()`, so every sub-second
  response reported `0`.
* Pure-ASGI security headers (#26924, issue #26922): the configured security headers are
  stamped once on the response start, streamed replies included.
* Empty audit exclusion list (#27370, commit 2ef6c76): an emptied `AUDIT_EXCLUDED_PATHS` became
  `['']`, whose alternation matched every path and silently switched auditing off.
* Readable audit bodies (#27369): the audit middleware was registered after compression, so it
  sat outside it and recorded gzip bytes as the response.
* Blocked webhook targets (commit 0671b7a, issue #26975): the URL check shared a try block with
  the POST, so a blocked target logged a full traceback instead of one warning.
* Provider rejections (#27238, issues #27237/#26253): the reason, model and status of an upstream
  refusal never reached the server log.
* Feedback events (commit 300302d): the event read `feedback.rating`, an attribute the model does
  not have, so every feedback event said the rating was None.
* Values in error logs (commit 6aebfd8, #26814): loguru's `diagnose` was left at its default, so
  every logged traceback printed the local variables beside each frame, a user's chat message
  among them. `LOGURU_DIAGNOSE` now switches it on, off by default.
* Transcription chunk order (#27417, issue #27143): a recording over the 20 MB upload limit of
  speech-to-text engines is split into chunks, and their transcripts were joined in the order the
  engine finished them, not the order they were spoken. Splitting runs ffmpeg and ffprobe, so
  that test skips on a host without them.
* Licensed startup (commits 8f77533, 0c7ddbd): the lifespan fetched the license inline, so an
  instance with a `LICENSE_KEY` was not ready until the license server answered or timed out,
  and an unreachable license host raised out of the fetch instead of falling through to the next.
  The license hosts are fixed, so the instance reaches them through a local HTTPS proxy that
  refuses the first host and holds the connection to the second.

open-webui 0.12.4 journey (commit 52e155019): with `/chat` audited, every model call of a chat
request writes a `model_usage` audit entry with the model and its input, output and total tokens,
linked to the request's own entry by `request_id`. A tool call run writes one per round; a reply
that reports no usage writes none.

The header, audit and webhook tests run on an instance of their own, booted with security
headers, request-and-response auditing, an emptied exclusion list and a legacy `WEBHOOK_URL` on
loopback, which the webhook sender refuses.

Twin of unit/config/test_observability_middleware.py; the pure-ASGI audit of the middleware
stack stays there.

Discriminates: passes on bbfa876af; formatting the process time with `int()` fails the process
time test, reverting 2ef6c76 the audit tests, registering audit after compression the gzip test,
reverting 0671b7a the webhook test, dropping the rejection log line the provider tests, reading
`feedback.rating` again the feedback test, joining chunks with `asyncio.as_completed` the
transcription test, defaulting `LOGURU_DIAGNOSE` to true the traceback
test, fetching the license inline (or awaiting it without a timeout) the readiness test and
letting the first host's error escape the fetch the fall-through test. The header, GET, sign-in
and opted-in diagnose tests pass on both. The model usage tests pass on dev 948432b43; returning
early from `log_model_usage` fails the chat, tool call and API usage tests, dropping the usage
write after a tool call follow-up round fails the tool call test, and writing an entry for empty
usage fails the no-usage test.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.audio_engine import serve_audio_engine, using_audio_engine
from harness.chat import ask
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD
from harness.listener import ReceivedRequest, json_answer
from harness.plugins import installed_function
from harness.prepared_data import serving
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

OBSERVED = {
    "XFRAME_OPTIONS": "DENY",
    "AUDIT_LOG_LEVEL": "REQUEST_RESPONSE",
    "AUDIT_EXCLUDED_PATHS": "",
    "WEBHOOK_URL": "http://127.0.0.1:9/hook",
}
WAIT_SECONDS = 10.0

FORWARD_FEEDBACK_EVENTS = """
import json
import urllib.request


class Event:
    def event(self, event):
        if event["event"].startswith("feedback."):
            body = json.dumps(event).encode()
            headers = {{"Content-Type": "application/json"}}
            request = urllib.request.Request("{url}", data=body, headers=headers)
            urllib.request.urlopen(request, timeout=10).close()
"""


@pytest.fixture
def observed(instance_with):
    return instance_with(OBSERVED)


def _eventually(read, timeout: float = WAIT_SECONDS):
    """The first truthy `read()` within the timeout, else its last value."""
    deadline = time.monotonic() + timeout
    value = read()
    while not value and time.monotonic() < deadline:
        time.sleep(0.1)
        value = read()
    return value


def _audit_entries(instance, path_suffix: str) -> list[dict]:
    audit_log = instance.data_dir / "audit.log"
    if not audit_log.exists():
        return []
    entries = [json.loads(line) for line in audit_log.read_text().splitlines() if line.strip()]
    return [entry for entry in entries if entry["request_uri"].split("?")[0].endswith(path_suffix)]


def _create_note(client, title: str, words: int = 20, **options):
    return client.post(
        "/api/v1/notes/create",
        json={"title": title, "data": {"content": {"md": "lorem ipsum " * words}}},
        **options,
    )


# response headers


def test_a_fast_response_reports_its_fractional_process_time(user):
    with user.client() as client:
        response = client.get("/api/config")

    reported = response.headers.get_list("x-process-time")
    assert len(reported) == 1, reported
    assert 0 < float(reported[0]) < 30, (
        f"a sub-second response reported X-Process-Time {reported[0]!r}; int() truncation "
        "reports 0 for every one of them (#27368)"
    )


def test_a_streamed_reply_arrives_whole_with_each_header_once(observed):
    pieces = [f"piece-{index} " for index in range(20)]
    observed.upstream.queue(reply.text(pieces))
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": "stream to me"}],
        "stream": True,
    }

    with admin_of(observed).client() as client:
        with client.stream("POST", "/api/chat/completions", json=request) as response:
            body = "".join(response.iter_text())

    assert response.headers.get_list("x-frame-options") == ["DENY"]
    assert len(response.headers.get_list("x-process-time")) == 1
    assert all(piece.strip() in body for piece in pieces), body
    assert "[DONE]" in body


# audit log


def test_an_emptied_exclusion_list_still_audits_every_path(observed):
    title = f"audited-{uuid.uuid4().hex[:8]}"
    with admin_of(observed).client() as client:
        created = _create_note(client, title)
    assert created.status_code == 200, created.text

    audited = _eventually(
        lambda: [
            entry
            for entry in _audit_entries(observed, "/notes/create")
            if title in entry["request_object"]
        ]
    )
    assert audited, (
        "an emptied AUDIT_EXCLUDED_PATHS switched auditing off, because the empty entry "
        "matched every path (#27370)"
    )
    assert audited[0]["verb"] == "POST"
    assert audited[0]["user"]["email"] == ADMIN_EMAIL


def test_a_compressed_response_is_recorded_readable(observed):
    title = f"compressed-{uuid.uuid4().hex[:8]}"
    with admin_of(observed).client() as client:
        # between the compression threshold and the audit body cap, so it is both
        created = _create_note(client, title, words=30, headers={"Accept-Encoding": "gzip"})
    assert created.headers.get("content-encoding") == "gzip", "the response was not compressed"

    audited = _eventually(
        lambda: [
            entry
            for entry in _audit_entries(observed, "/notes/create")
            if title in entry["request_object"]
        ]
    )
    assert audited, "the request was never audited"
    recorded = audited[0]["response_object"]
    assert title in recorded, (
        "the audit log recorded the compressed bytes, not the response, because auditing sat "
        f"outside compression (#27369): {recorded[:120]!r}"
    )
    assert json.loads(recorded)["title"] == title


def test_a_get_is_not_audited_and_a_sign_in_is(observed):
    with admin_of(observed).client() as client:
        client.get("/api/v1/notes/").raise_for_status()
        signed_in = client.post(
            "/api/v1/auths/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
    assert signed_in.status_code == 200, signed_in.text

    sign_ins = _eventually(lambda: _audit_entries(observed, "/auths/signin"))
    assert sign_ins, "a sign-in was not audited"
    assert ADMIN_PASSWORD not in sign_ins[-1]["request_object"]
    assert _audit_entries(observed, "/api/v1/notes/") == []


# model usage in the audit log

ADD_TOOL = '''
class Tools:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
'''


def _tokens(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _audited_chat_request(instance, prompt: str) -> dict:
    """The HTTP audit entry of the chat request that carried `prompt`."""
    entries = _eventually(
        lambda: [
            entry
            for entry in _audit_entries(instance, "/api/chat/completions")
            if entry["extra"].get("event") != "model_usage" and prompt in entry["request_object"]
        ]
    )
    assert len(entries) == 1, f"the chat request was audited {len(entries)} times: {entries}"
    return entries[0]


def _model_usage_entries(instance, request_id: str) -> list[dict]:
    return [
        entry
        for entry in _audit_entries(instance, "/api/chat/completions")
        if entry["extra"].get("event") == "model_usage"
        and entry["extra"].get("request_id") == request_id
    ]


def _usage_of(prompt: int, completion: int) -> dict:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def test_a_chat_reply_writes_its_model_usage_linked_to_the_request(observed):
    prompt = f"count my tokens {uuid.uuid4().hex[:8]}"
    observed.upstream.queue(
        reply.text("counted", usage=_tokens(11, 7), match=reply.answering(prompt))
    )

    with admin_of(observed).client() as client:
        ask(client, prompt)
    request = _audited_chat_request(observed, prompt)
    usages = _eventually(lambda: _model_usage_entries(observed, request["id"]))

    assert len(usages) == 1, (
        f"a reply with usage wrote {len(usages)} model usage entries for its request, not one"
    )
    assert usages[0]["model_id"] == MOCK_MODEL_ID, usages[0]
    assert usages[0]["usage"] == _usage_of(11, 7), (
        f"the model usage entry did not carry the reply's token counts: {usages[0]['usage']}"
    )
    assert usages[0]["id"] != request["id"]
    assert usages[0]["user"]["email"] == ADMIN_EMAIL


def test_each_tool_call_round_writes_its_own_model_usage(observed):
    prompt = f"add two and three {uuid.uuid4().hex[:8]}"
    answering = reply.answering(prompt)
    owner = admin_of(observed)

    with python_tool(owner, ADD_TOOL, name="Adder") as tool_id, owner.client() as client:
        observed.upstream.queue(
            reply.tool_call("add", {"a": 2, "b": 3}, usage=_tokens(20, 5), match=answering),
            reply.text("five", usage=_tokens(30, 2), match=answering),
        )
        _, message = ask(client, prompt, tool_ids=[tool_id])
    assert message["content"] == "five", message
    request = _audited_chat_request(observed, prompt)
    _eventually(lambda: len(_model_usage_entries(observed, request["id"])) >= 2)
    usages = _model_usage_entries(observed, request["id"])

    assert [entry["usage"] for entry in usages] == [_usage_of(20, 5), _usage_of(30, 2)], (
        "a tool call run did not write one model usage entry per round with that round's "
        f"tokens: {usages}"
    )
    assert all(entry["model_id"] == MOCK_MODEL_ID for entry in usages), usages


def test_a_reply_without_usage_writes_no_model_usage(observed):
    prompt = f"no counts here {uuid.uuid4().hex[:8]}"
    observed.upstream.queue(reply.text("uncounted", match=reply.answering(prompt)))

    with admin_of(observed).client() as client:
        _, message = ask(client, prompt)
    assert message["content"] == "uncounted", message
    request = _audited_chat_request(observed, prompt)
    # nothing more is expected; the pause gives a late entry time to land
    time.sleep(1)

    assert _model_usage_entries(observed, request["id"]) == [], (
        "a reply that reported no usage still wrote a model usage entry"
    )


@pytest.mark.parametrize("stream", [True, False])
def test_an_api_completion_writes_its_model_usage(observed, stream):
    prompt = f"api tokens {uuid.uuid4().hex[:8]}"
    observed.upstream.queue(
        reply.text("counted", usage=_tokens(13, 4), match=reply.answering(prompt))
    )
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }

    with admin_of(observed).client() as client:
        answered = client.post("/api/chat/completions", json=request)
    assert answered.status_code == 200, answered.text
    audited = _audited_chat_request(observed, prompt)
    usages = _eventually(lambda: _model_usage_entries(observed, audited["id"]))

    assert [entry["usage"] for entry in usages] == [_usage_of(13, 4)], (
        f"an API completion (stream={stream}) did not write its model usage once: {usages}"
    )
    assert usages[0]["model_id"] == MOCK_MODEL_ID, usages[0]


# server log


def test_a_blocked_webhook_target_logs_one_warning_without_a_traceback(observed):
    offset = observed.log_size()
    create_user(observed)

    _eventually(
        lambda: any(mark in observed.log_since(offset) for mark in ("Webhook skipped", "Traceback"))
    )
    logged = observed.log_since(offset)
    assert "Traceback" not in logged, (
        "a blocked webhook target logged a full traceback instead of one warning (#26975):\n"
        f"{logged[-2000:]}"
    )
    assert "Webhook skipped" in logged, f"the blocked webhook was never reported:\n{logged[-2000:]}"


@pytest.mark.parametrize("status, level", [(400, "WARNING"), (429, "WARNING"), (503, "ERROR")])
def test_a_provider_rejection_reaches_the_server_log(instance, user, upstream, status, level):
    reason = f"max_tokens is too large ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.error(status, reason))
    offset = instance.log_size()

    with user.client() as client:
        ask(client, "hello")

    lines = _eventually(
        lambda: [line for line in instance.log_since(offset).splitlines() if reason in line]
    )
    assert lines, (
        f"the provider's HTTP {status} reason never reached the server log (#27238):\n"
        f"{instance.log_since(offset)[-2000:]}"
    )
    assert f"HTTP {status}" in lines[0] and MOCK_MODEL_ID in lines[0]
    assert f"| {level}" in lines[0], lines[0]


# events


def _feedback_events(listener, event_name: str) -> list[dict]:
    received = [request.json() for request in listener.requests_to("/events")]
    return [event for event in received if event["event"] == event_name]


def test_feedback_events_carry_the_rating(admin, user, listener):
    listener.route("POST", "/events", json_answer({}))
    source = FORWARD_FEEDBACK_EVENTS.format(url=f"{listener.base_url}/events")

    with installed_function(admin, source), user.client() as client:
        created = client.post(
            "/api/v1/evaluations/feedback",
            json={"type": "rating", "data": {"rating": 1, "model_id": MOCK_MODEL_ID}},
        )
        assert created.status_code == 200, created.text
        feedback_id = created.json()["id"]
        created_events = _eventually(lambda: _feedback_events(listener, "feedback.created"))

        updated = client.post(
            f"/api/v1/evaluations/feedback/{feedback_id}",
            json={"type": "rating", "data": {"rating": -1, "model_id": MOCK_MODEL_ID}},
        )
        assert updated.status_code == 200, updated.text
        updated_events = _eventually(lambda: _feedback_events(listener, "feedback.updated"))

    assert created_events, "no feedback.created event reached the event function"
    assert created_events[0]["data"]["rating"] == 1, (
        "the feedback event reported no rating, because it read an attribute the feedback "
        f"model does not have: {created_events[0]['data']}"
    )
    assert updated_events and updated_events[0]["data"]["rating"] == -1, updated_events


# tracebacks in the server log

# Short enough to fall inside loguru's truncated rendering of the chat body.
PRIVATE_MESSAGE = "pw hunter2"

FAILING_OUTLET = """
class Filter:
    def outlet(self, body):
        raise RuntimeError("the outlet filter broke")
"""


def _outlet_failure_log(target, admin) -> str:
    offset = target.log_size()
    with installed_function(admin, FAILING_OUTLET, is_global=True), admin.client() as client:
        ask(client, PRIVATE_MESSAGE)
        _eventually(lambda: "the outlet filter broke" in target.log_since(offset))
    logged = target.log_since(offset)
    assert "Traceback" in logged, f"the failing filter logged no traceback:\n{logged[-2000:]}"
    return logged


def test_a_logged_traceback_does_not_print_the_users_message(instance, admin, upstream):
    logged = _outlet_failure_log(instance, admin)

    assert PRIVATE_MESSAGE not in logged, (
        "a logged traceback printed the local variables beside its frames, the user's chat "
        f"message among them (#26814):\n{logged[-3000:]}"
    )


def test_an_operator_who_opts_in_sees_the_values(instance_with):
    diagnosing = instance_with({"LOGURU_DIAGNOSE": "true"})

    logged = _outlet_failure_log(diagnosing, admin_of(diagnosing))

    assert PRIVATE_MESSAGE in logged, (
        "with LOGURU_DIAGNOSE=true the traceback no longer shows the chat body beside its frame; "
        "retarget the probe message, the test above proves nothing without this"
    )


# transcription of a long recording

# 32 kbit/s for 93 minutes is still over 20 MB after the server's own compression.
LONG_RECORDING_SECONDS = 5600


@pytest.fixture(scope="module")
def long_recording(tmp_path_factory) -> Path:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("splitting a long recording needs ffmpeg and ffprobe on PATH")
    recording = tmp_path_factory.mktemp("recording") / "meeting.mp3"
    tone = f"sine=frequency=440:sample_rate=16000:duration={LONG_RECORDING_SECONDS}"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", tone]
        + ["-ac", "1", "-b:a", "32k", str(recording)],
        check=True,
        timeout=300,
    )
    return recording


def _chunk_index(request: ReceivedRequest) -> int:
    match = re.search(rb'filename="[^"]*_chunk_(\d+)\.', request.body)
    assert match, f"the chunk upload named no chunk: {request.body[:300]!r}"
    return int(match.group(1))


def test_a_long_recording_is_transcribed_in_spoken_order(
    admin, make_user, listener, long_recording
):
    engine = serve_audio_engine(listener)
    later_chunks_done = threading.Event()

    def later_chunks_answer_first(request: ReceivedRequest):
        index = _chunk_index(request)
        if index == 0:
            later_chunks_done.wait(timeout=60)
        else:
            later_chunks_done.set()
        return json_answer({"text": f"part{index}"})

    listener.route("POST", "/audio/transcriptions", later_chunks_answer_first)
    with admin.client() as admin_client, using_audio_engine(admin_client, engine):
        with make_user().client() as client, long_recording.open("rb") as audio:
            transcribed = client.post(
                "/api/v1/audio/transcriptions",
                files={"file": ("meeting.mp3", audio, "audio/mpeg")},
                timeout=600,
            )

    assert transcribed.status_code == 200, transcribed.text
    chunks = len(engine.transcription_requests())
    assert chunks > 1, "the recording was not split; lengthen it past the upload limit"
    spoken_order = " ".join(f"part{index}" for index in range(chunks))
    assert transcribed.json()["text"] == spoken_order, (
        "the chunks were joined in the order the engine finished them, not the order they were "
        f"spoken (#27143): {transcribed.json()['text']!r}"
    )


def test_a_chunk_the_engine_fails_fails_the_whole_transcription(
    admin, make_user, listener, long_recording
):
    engine = serve_audio_engine(listener)

    def the_last_chunk_fails(request: ReceivedRequest):
        if _chunk_index(request) == 0:
            return json_answer({"text": "part0"})
        return json_answer({"error": {"message": "the engine is overloaded"}}, status=500)

    listener.route("POST", "/audio/transcriptions", the_last_chunk_fails)
    with admin.client() as admin_client, using_audio_engine(admin_client, engine):
        with make_user().client() as client, long_recording.open("rb") as audio:
            transcribed = client.post(
                "/api/v1/audio/transcriptions",
                files={"file": ("meeting.mp3", audio, "audio/mpeg")},
                timeout=600,
            )

    assert len(engine.transcription_requests()) > 1
    assert transcribed.status_code >= 400, (
        f"a chunk the engine refused still produced a transcript: {transcribed.text}"
    )
    assert "part0" not in transcribed.text


# licensed startup

REFUSED_LICENSE_HOST = "api.openwebui.com:443"
HELD_LICENSE_HOST = "licenses.api.openwebui.com:443"


class LicenseProxy:
    """An HTTPS proxy: refuses CONNECT to the first license host, holds the second one open."""

    def __init__(self) -> None:
        self.server = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.server.getsockname()[1]}"
        self.targets: list[str] = []
        self.open_targets: list[str] = []
        self.lock = threading.Lock()

    def serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            request = b""
            while b"\r\n\r\n" not in request:
                received = connection.recv(4096)
                if not received:
                    return
                request += received
            target = request.split(b" ")[1].decode()
            with self.lock:
                self.targets.append(target)
            if target == REFUSED_LICENSE_HOST:
                connection.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                return
            with self.lock:
                self.open_targets.append(target)
            try:
                while connection.recv(4096):
                    pass
            except OSError:
                pass
            with self.lock:
                self.open_targets.remove(target)

    def pending(self) -> list[str]:
        with self.lock:
            return list(self.open_targets)


@contextmanager
def license_proxy() -> Iterator[LicenseProxy]:
    proxy = LicenseProxy()
    threading.Thread(target=proxy.serve, daemon=True).start()
    try:
        yield proxy
    finally:
        proxy.server.close()


@pytest.fixture(scope="module")
def licensed_boot(tmp_path_factory) -> dict:
    """What the license proxy saw by the time a licensed instance first answered `/health`."""
    with license_proxy() as proxy:
        env = {
            "LICENSE_KEY": "test-license-key",
            "HTTPS_PROXY": proxy.url,
            "https_proxy": proxy.url,
            "NO_PROXY": "127.0.0.1,localhost",
        }
        with serving(tmp_path_factory.mktemp("licensed"), env) as backend:
            ready = {"pending": proxy.pending(), "targets": list(proxy.targets)}
            ready["log"] = backend.log()
    return ready


def test_startup_does_not_wait_on_the_license_server(licensed_boot):
    assert licensed_boot["pending"] == [HELD_LICENSE_HOST], (
        "the instance only became ready once the license request had ended, so startup waits "
        f"on the license server; proxy saw {licensed_boot['targets']}"
    )


def test_an_unreachable_license_host_falls_through_to_the_next(licensed_boot):
    assert licensed_boot["targets"][:2] == [REFUSED_LICENSE_HOST, HELD_LICENSE_HOST], (
        "the refused first license host ended the lookup instead of falling through to the "
        f"second: {licensed_boot['targets']}\n{licensed_boot['log'][-2000:]}"
    )
