"""An account that may have automations, and a way to give it some.

`scheduler` is a fresh admin, since plain accounts need the automations permission. Its
automations are deleted afterwards so none runs on its own later in the session.
`make_automation(owner, ...)` saves an automation the way the Create dialog does; its schedule
starts in 2099, so the scheduler never runs it and only Run now does.
"""

from __future__ import annotations

import uuid
from typing import Callable

import pytest

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

FUTURE_DAILY = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0"


@pytest.fixture
def scheduler(make_user):
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


@pytest.fixture
def make_automation() -> Callable[..., dict]:
    def create(
        owner: Actor,
        model_id: str = MOCK_MODEL_ID,
        is_active: bool = True,
        rrule: str = FUTURE_DAILY,
    ) -> dict:
        form = {
            "name": f"Weekly report {uuid.uuid4().hex[:6]}",
            "is_active": is_active,
            "data": {
                "prompt": f"Write the weekly report, batch {uuid.uuid4().hex[:6]}.",
                "model_id": model_id,
                "rrule": rrule,
                "target": {"type": "chat"},
            },
        }
        with owner.client() as client:
            created = client.post("/api/v1/automations/create", json=form)
        assert created.status_code == 200, created.text
        return created.json()

    return create
