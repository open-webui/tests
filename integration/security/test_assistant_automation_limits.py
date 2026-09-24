"""Regression: automations the assistant creates for a user obey the same limits as the API.

open-webui 0.11.0 fix `076a84e3f` (#27523, issue #27121): the builtin chat tools
`create_automation` and `update_automation` wrote straight to the automations table, skipping
the `check_automation_limits` that `/api/v1/automations/create` and `/{id}/update` run. A user
could ask the model for any number of automations, past `automations.max_count`, and
reschedule below `automations.min_interval`. Both tools now run the routers' check and hand its
refusal back to the model as an error.

Twin of unit/security/test_assistant_automation_limits.py.

Discriminates: passes on dev bbfa876af; with the two `check_automation_limits` calls removed
from tools/builtin.py the model's create and reschedule go through and the tool results report
success.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MAX_COUNT = 2
MIN_INTERVAL = 3600
HOURLY = "RRULE:FREQ=HOURLY;INTERVAL=1"
EVERY_30_MINUTES = "RRULE:FREQ=MINUTELY;INTERVAL=30"


@pytest.fixture
def limits(admin, preserve):
    """Automations on for users, capped at `MAX_COUNT` and at most hourly."""
    preserve("admin_config", "permissions")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        limited = {
            **config,
            "ENABLE_AUTOMATIONS": True,
            "AUTOMATION_MAX_COUNT": MAX_COUNT,
            "AUTOMATION_MIN_INTERVAL": MIN_INTERVAL,
        }
        client.post("/api/v1/auths/admin/config", json=limited).raise_for_status()
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()


def _create_over_http(client, rrule: str = HOURLY) -> dict:
    form = {
        "name": "weekly report",
        "data": {"prompt": "summarise the week", "model_id": "mock-model", "rrule": rrule},
        "is_active": False,
    }
    created = client.post("/api/v1/automations/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _automations(client) -> list[dict]:
    listed = client.get("/api/v1/automations/list")
    listed.raise_for_status()
    return listed.json()["items"]


def _model_calls(client, upstream, tool: str, arguments: dict) -> dict:
    """The model calls `tool`; returns the result the model is handed back."""
    upstream.queue(reply.tool_call(tool, arguments), reply.text("All set."))
    ask(client, "please take care of my automations")
    replayed = upstream.chat_requests()[-1]["messages"]
    results = [entry["content"] for entry in replayed if entry["role"] == "tool"]
    assert results, f"the {tool} result never reached the model: {replayed}"
    return json.loads(results[-1])


def _create_from_chat(client, upstream, rrule: str = HOURLY) -> dict:
    arguments = {"name": "morning briefing", "prompt": "summarise my inbox", "rrule": rrule}
    return _model_calls(client, upstream, "create_automation", arguments)


def test_the_model_cannot_create_past_the_max_count(limits, make_user, upstream):
    with make_user().client() as client:
        for _ in range(MAX_COUNT):
            _create_over_http(client)

        result = _create_from_chat(client, upstream)

        assert "error" in result and str(MAX_COUNT) in result["error"], (
            f"the assistant created automation number {MAX_COUNT + 1} for a user at the "
            f"automations.max_count limit (#27121): {result}"
        )
        assert len(_automations(client)) == MAX_COUNT


def test_the_model_cannot_create_below_the_min_interval(limits, make_user, upstream):
    with make_user().client() as client:
        result = _create_from_chat(client, upstream, rrule=EVERY_30_MINUTES)

        assert "error" in result and str(MIN_INTERVAL) in result["error"], (
            f"the assistant scheduled an automation every 30 minutes under a {MIN_INTERVAL}s "
            f"floor (#27121): {result}"
        )
        assert _automations(client) == []


def test_the_model_cannot_reschedule_below_the_min_interval(limits, make_user, upstream):
    with make_user().client() as client:
        existing = _create_over_http(client)

        result = _model_calls(
            client,
            upstream,
            "update_automation",
            {"automation_id": existing["id"], "rrule": EVERY_30_MINUTES},
        )

        assert "error" in result and str(MIN_INTERVAL) in result["error"], (
            f"the assistant rescheduled an automation below the {MIN_INTERVAL}s floor "
            f"(#27121): {result}"
        )
        stored = client.get(f"/api/v1/automations/{existing['id']}").json()
        assert stored["data"]["rrule"] == HOURLY


def test_the_http_create_refuses_the_same_automation(limits, make_user):
    with make_user().client() as client:
        for _ in range(MAX_COUNT):
            _create_over_http(client)
        refused = client.post(
            "/api/v1/automations/create",
            json={
                "name": "one too many",
                "data": {"prompt": "p", "model_id": "mock-model", "rrule": HOURLY},
            },
        )

    assert refused.status_code == 403, refused.text


def test_the_model_can_create_below_the_cap_at_exactly_the_min_interval(
    limits, make_user, upstream
):
    with make_user().client() as client:
        _create_over_http(client)

        result = _create_from_chat(client, upstream, rrule=HOURLY)

        assert result.get("status") == "success", result
        assert len(_automations(client)) == 2


def test_deleting_an_automation_frees_a_slot_for_the_model(limits, make_user, upstream):
    with make_user().client() as client:
        first = _create_over_http(client)
        _create_over_http(client)
        deleted = _model_calls(
            client, upstream, "delete_automation", {"automation_id": first["id"]}
        )
        assert deleted.get("status") == "success", deleted

        result = _create_from_chat(client, upstream)

        assert result.get("status") == "success", result


def test_an_admin_is_not_capped_in_chat(limits, make_user, upstream):
    with make_user(role="admin").client() as client:
        for _ in range(MAX_COUNT):
            _create_over_http(client)

        result = _create_from_chat(client, upstream, rrule=EVERY_30_MINUTES)

        assert result.get("status") == "success", result
        assert len(_automations(client)) == MAX_COUNT + 1
