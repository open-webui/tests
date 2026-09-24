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

The header, audit and webhook tests run on an instance of their own, booted with security
headers, request-and-response auditing, an emptied exclusion list and a legacy `WEBHOOK_URL` on
loopback, which the webhook sender refuses.

Twin of unit/config/test_observability_middleware.py; the pure-ASGI audit of the middleware
stack, the log diagnose default, transcription order and licensed startup stay there.

Discriminates: passes on bbfa876af; formatting the process time with `int()` fails the process
time test, reverting 2ef6c76 the audit tests, registering audit after compression the gzip test,
reverting 0671b7a the webhook test, dropping the rejection log line the provider tests and
reading `feedback.rating` again the feedback test. The header, GET and sign-in tests pass on both.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD
from harness.listener import json_answer
from harness.plugins import installed_function
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
        created = _create_note(client, title, words=80, headers={"Accept-Encoding": "gzip"})
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
