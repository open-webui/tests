"""Journey: a run started with Run now is recorded with its status and counts as the last run.

`POST /api/v1/automations/{id}/run` starts a run in the background. Its record in
`GET /api/v1/automations/{id}/runs` says `success` and names the chat it made, or `error` with
the reason when the automation's model no longer exists.

Bug, open-webui/open-webui#31580, fixed by PR #31583: the automation's `last_run_at` ("Last
execution time" in the database reference) was only written when the scheduler claimed a due
automation, so after a Run now run the automation and its list entry still said it never ran, and
the automations page showed "Last run Never" above the run it listed.
`test_a_run_now_run_sets_the_last_run_time` pins it; the browser twin is in
e2e/models/test_automation_runs.py.

`test_a_run_is_recorded_as_a_success_with_its_chat` is red on dev 62f70a844: since de73bb830 a chat
request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev a5bc78300; the last run test fails on dev 176d31d1d, before PR #31583.
In a backend copy recording every run as `success` the failed run test goes red, and recording a
success without its chat turns the success test red.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
RUN_WAIT = 30.0


@pytest.fixture
def scheduler(make_user):
    """A fresh admin, whose automations are deleted afterwards so none runs on its own later."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


def _create(client: httpx.Client, model_id: str = MOCK_MODEL_ID) -> dict:
    created = client.post(
        "/api/v1/automations/create",
        json={
            "name": f"digest {uuid.uuid4().hex[:6]}",
            "is_active": True,
            "data": {
                "prompt": f"Summarise the day, batch {uuid.uuid4().hex[:6]}.",
                "model_id": model_id,
                "rrule": SCHEDULE,
            },
        },
    )
    assert created.status_code == 200, created.text
    return created.json()


def _run_and_wait(client: httpx.Client, automation_id: str) -> dict:
    started = client.post(f"/api/v1/automations/{automation_id}/run")
    assert started.status_code == 200, started.text
    deadline = time.monotonic() + RUN_WAIT
    while time.monotonic() < deadline:
        runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
        if runs:
            return runs[0]
        time.sleep(0.2)
    raise AssertionError(f"no run was recorded within {RUN_WAIT}s")


def test_a_run_is_recorded_as_a_success_with_its_chat(scheduler, upstream):
    with scheduler.client() as client:
        automation = _create(client)
        prompt = automation["data"]["prompt"]
        upstream.queue(reply.text("All quiet.", match=reply.answering(prompt)))
        run = _run_and_wait(client, automation["id"])
        chat = client.get(f"/api/v1/chats/{run['chat_id']}")

    assert run["status"] == "success", run
    assert run["error"] is None
    assert chat.status_code == 200, chat.text


def test_a_run_whose_model_is_gone_is_recorded_as_an_error(scheduler):
    with scheduler.client() as client:
        automation = _create(client, model_id=f"retired-{uuid.uuid4().hex[:6]}")
        run = _run_and_wait(client, automation["id"])

    assert run["status"] == "error", run
    assert "Model not found" in run["error"]
    assert run["chat_id"] is None


def test_a_run_now_run_sets_the_last_run_time(scheduler, upstream):
    with scheduler.client() as client:
        automation = _create(client)
        assert automation["last_run_at"] is None
        prompt = automation["data"]["prompt"]
        upstream.queue(reply.text("All quiet.", match=reply.answering(prompt)))
        run = _run_and_wait(client, automation["id"])
        stored = client.get(f"/api/v1/automations/{automation['id']}").json()
        listed = client.get("/api/v1/automations/list").json()["items"]

    listed_entry = next(item for item in listed if item["id"] == automation["id"])
    assert stored["last_run_at"] is not None and listed_entry["last_run_at"] is not None, (
        f"a {run['status']} run was recorded but the automation says it never ran "
        "(open-webui/open-webui#31580)"
    )
