"""Journey: the model keeps a task list, schedules automations and events, reads skills, notifies.

* `create_tasks` and `update_task` keep a checklist on the saved chat; an unknown task or status
  is refused and leaves the list alone.
* `create_automation` stores a schedule on the current model for an account allowed automations,
  `toggle_automation` pauses and resumes it and `delete_automation` removes it; nobody touches
  another account's automation, and an account without the feature is not offered the tools.
* `create_calendar_event` writes in the account's local time into its default calendar or a
  calendar shared with write access, never one shared read-only; `delete_calendar_event` deletes
  only events the account may write.
* `view_skill` loads a skill the account may read and refuses one it may not.
* `get_current_timestamp` adds the account's local time; `notify` delivers to the account's
  default notification target.

The scripted model calls each tool and the test reads the result it was sent back and the rows
it left.

Discriminates: in a backend copy, `update_task` dropping its not-found answer turned the task
list test red; `toggle_automation` and `delete_automation` skipping the owner check turned the
foreign automation test red; `create_calendar_event` checking read in place of write turned the
read-only calendar test red; `_dt_to_ns` reading times as UTC turned the local time test red;
`view_skill` skipping its grant check turned the private skill test red; and `_find_target`
ignoring the default target turned the delivery test red.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.calendar_api import HOUR_NS, default_calendar_id, set_timezone, to_ns
from harness.chat import ask
from harness.listener import json_answer
from harness.tool_calls import offered_tools, run_tool
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

AUTOMATION_TOOLS = {
    "create_automation",
    "update_automation",
    "list_automations",
    "toggle_automation",
    "delete_automation",
}
# first due in 2099, so the scheduler never runs it into another test's script
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"


def call(actor, upstream, tool: str, **arguments):
    """The result a builtin tool returned when the model called it for `actor`, parsed if JSON."""
    with actor.client() as client:
        result = run_tool(client, upstream, tool, arguments)
    try:
        return json.loads(result)
    except ValueError:
        return result


def call_in_turn(actor, upstream, *calls: tuple[str, dict]) -> tuple[str, list]:
    """Have the model call each tool in turn in one reply; the chat id and every result."""
    scripted = [
        reply.tool_call(name, arguments, call_id=f"call_{index}")
        for index, (name, arguments) in enumerate(calls)
    ]
    upstream.queue(*scripted, reply.text("done"))
    with actor.client() as client:
        turn, _ = ask(client, "plan it")
    sent_back = upstream.chat_requests()[-1]["messages"]
    results = [json.loads(entry["content"]) for entry in sent_back if entry["role"] == "tool"]
    return turn.chat_id, results


# --- tasks --------------------------------------------------------------------------------------


def test_a_task_list_is_kept_on_the_chat(make_user, upstream):
    account = make_user()
    tasks = [
        {"content": "draft"},
        {"content": "review", "status": "in_progress"},
        {"content": "   "},
        {"id": "ship", "content": "ship", "status": "someday"},
    ]

    chat_id, (created, updated, unknown, invalid) = call_in_turn(
        account,
        upstream,
        ("create_tasks", {"tasks": tasks}),
        ("update_task", {"id": "1"}),
        ("update_task", {"id": "nope"}),
        ("update_task", {"id": "ship", "status": "done-ish"}),
    )

    assert [(task["id"], task["status"]) for task in created["tasks"]] == [
        ("1", "pending"),
        ("2", "in_progress"),
        ("ship", "pending"),
    ]
    assert updated["summary"] == {
        "total": 3,
        "pending": 1,
        "in_progress": 1,
        "completed": 1,
        "cancelled": 0,
    }
    assert unknown == {"error": 'Task with id "nope" not found'}
    assert invalid["error"].startswith("Invalid status: done-ish")
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}").json()["tasks"]
    assert [task["status"] for task in stored] == ["completed", "in_progress", "pending"]


# --- automations --------------------------------------------------------------------------------


def stored_automation(owner, automation_id: str):
    with owner.client() as client:
        return client.get(f"/api/v1/automations/{automation_id}")


def test_an_automation_is_created_paused_resumed_and_deleted(make_user, upstream):
    owner = make_user(role="admin")

    created = call(
        owner, upstream, "create_automation", name="Digest", prompt="sum up", rrule=SCHEDULE
    )
    paused = call(owner, upstream, "toggle_automation", automation_id=created["id"])
    resumed = call(owner, upstream, "toggle_automation", automation_id=created["id"])
    stored = stored_automation(owner, created["id"]).json()
    deleted = call(owner, upstream, "delete_automation", automation_id=created["id"])

    assert (created["model_id"], created["is_active"]) == ("mock-model", True), created
    assert created["next_runs"], "the schedule produced no upcoming runs"
    assert (paused["is_active"], resumed["is_active"]) == (False, True)
    assert (stored["name"], stored["data"]["prompt"], stored["data"]["rrule"]) == (
        "Digest",
        "sum up",
        SCHEDULE,
    )
    assert deleted == {"status": "success", "message": 'Automation "Digest" deleted'}
    assert stored_automation(owner, created["id"]).status_code == 404


def test_another_accounts_automation_is_left_alone(make_user, upstream):
    owner, other = make_user(role="admin"), make_user(role="admin")
    created = call(owner, upstream, "create_automation", name="Mine", prompt="p", rrule=SCHEDULE)

    toggled = call(other, upstream, "toggle_automation", automation_id=created["id"])
    deleted = call(other, upstream, "delete_automation", automation_id=created["id"])

    assert toggled == deleted == {"error": "Access denied"}
    stored = stored_automation(owner, created["id"])
    assert stored.status_code == 200 and stored.json()["is_active"] is True


def test_a_bad_schedule_or_folder_is_refused(make_user, upstream):
    owner = make_user(role="admin")

    bad_rule = call(owner, upstream, "create_automation", name="x", prompt="p", rrule="whenever")
    bad_folder = call(
        owner,
        upstream,
        "create_automation",
        name="x",
        prompt="p",
        rrule=SCHEDULE,
        folder_id="not-my-folder",
    )

    assert bad_rule["error"].startswith("Invalid schedule"), bad_rule
    assert bad_folder == {"error": "Folder not found"}


def test_automation_tools_are_offered_only_with_the_feature(make_user, upstream):
    with make_user().client() as client:
        for_a_user = offered_tools(client, upstream)
    with make_user(role="admin").client() as client:
        for_an_admin = offered_tools(client, upstream)

    assert not AUTOMATION_TOOLS & for_a_user, f"a user without automations got {for_a_user}"
    assert AUTOMATION_TOOLS <= for_an_admin


# --- calendar -----------------------------------------------------------------------------------


def test_an_event_is_created_in_the_accounts_local_time(make_user, upstream):
    account = make_user()
    with account.client() as client:
        set_timezone(client, "Europe/Berlin")
        calendar_id = default_calendar_id(client)

    created = call(
        account,
        upstream,
        "create_calendar_event",
        title="Dentist",
        start="2026-11-03 09:00",
        location="Main St",
    )

    assert (created["calendar_id"], created["start"]) == (calendar_id, "2026-11-03 09:00")
    with account.client() as client:
        stored = client.get(f"/api/v1/calendars/events/{created['id']}").json()
    nine_in_berlin = to_ns(dt.datetime(2026, 11, 3, 8, 0, tzinfo=dt.timezone.utc))
    assert stored["start_at"] == nine_in_berlin, "the start was not read as the account's time"
    assert stored["end_at"] - stored["start_at"] == HOUR_NS
    assert (stored["location"], stored["meta"]) == ("Main St", {"alert_minutes": 10})


def shared_calendar(owner, member, permission: str) -> str:
    grant = {"principal_type": "user", "principal_id": member.id, "permission": permission}
    with owner.client() as client:
        created = client.post(
            "/api/v1/calendars/create", json={"name": "Team", "access_grants": [grant]}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_a_shared_calendar_takes_events_only_with_write_access(make_user, upstream):
    owner, reader, writer = make_user(), make_user(), make_user()
    read_only = shared_calendar(owner, reader, "read")
    writable = shared_calendar(owner, writer, "write")

    refused = call(
        reader,
        upstream,
        "create_calendar_event",
        title="x",
        start="2026-11-03 09:00",
        calendar_id=read_only,
    )
    accepted = call(
        writer,
        upstream,
        "create_calendar_event",
        title="Offsite",
        start="2026-11-03 09:00",
        calendar_id=writable,
        all_day=True,
    )

    assert refused == {"error": "Access denied to this calendar"}, refused
    assert (accepted["calendar_id"], accepted["all_day"], accepted["end"]) == (
        writable,
        True,
        None,
    )


def test_an_event_is_deleted_only_by_someone_who_may_write_it(make_user, upstream):
    owner, stranger = make_user(), make_user()
    created = call(
        owner, upstream, "create_calendar_event", title="Standup", start="2026-11-03 09:00"
    )

    refused = call(stranger, upstream, "delete_calendar_event", event_id=created["id"])
    deleted = call(owner, upstream, "delete_calendar_event", event_id=created["id"])

    assert refused == {"error": "Access denied"}, f"a stranger deleted an event: {refused}"
    assert deleted == {"status": "success", "message": 'Event "Standup" deleted'}
    with owner.client() as client:
        assert client.get(f"/api/v1/calendars/events/{created['id']}").status_code == 404


def test_an_unparseable_start_is_reported(make_user, upstream):
    result = call(make_user(), upstream, "create_calendar_event", title="x", start="soonish")

    assert result["error"].startswith("Invalid start datetime"), result


# --- skills and time ----------------------------------------------------------------------------


@pytest.fixture
def skills(admin, make_user):
    """(reader, shared skill id, private skill id), both made by the admin."""
    reader = make_user()
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    ids = {kind: f"{kind}-{uuid.uuid4().hex[:8]}" for kind in ("shared", "private")}
    with admin.client() as client:
        for kind, skill_id in ids.items():
            created = client.post(
                "/api/v1/skills/create",
                json={
                    "id": skill_id,
                    "name": f"{kind} skill",
                    "content": f"{kind} instructions",
                    "access_grants": [grant] if kind == "shared" else [],
                },
            )
            assert created.status_code == 200, created.text
        yield reader, ids["shared"], ids["private"]
        for skill_id in ids.values():
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_a_skill_is_loaded_only_when_the_account_may_read_it(skills, upstream):
    reader, shared_id, private_id = skills

    loaded = call(reader, upstream, "view_skill", id=shared_id.upper())
    refused = call(reader, upstream, "view_skill", id=private_id)

    assert loaded == {"name": "shared skill", "content": "shared instructions"}
    assert refused == {"error": "Access denied"}, f"a private skill was loaded: {refused}"


def test_the_current_time_includes_the_accounts_local_time(make_user, upstream):
    account = make_user()
    with account.client() as client:
        set_timezone(client, "Asia/Tokyo")

    now = call(account, upstream, "get_current_timestamp")

    assert now["user_timezone"] == "Asia/Tokyo"
    assert now["user_local_iso"].endswith("+09:00")
    local = dt.datetime.fromisoformat(now["user_local_iso"])
    assert int(local.timestamp()) == now["current_timestamp"]


# --- notify -------------------------------------------------------------------------------------


@pytest.mark.slow
def test_notify_delivers_to_the_default_target(instance_with, preserve, listener):
    fetching = instance_with(LOCAL_WEB_FETCH)
    preserve("admin_config", "permissions", on=fetching)
    with admin_of(fetching).client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_USER_WEBHOOKS": True}
        ).raise_for_status()
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["webhooks"] = True
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()
    listener.route("POST", "/hook", json_answer({}))
    account = create_user(fetching)
    with account.client() as client:
        target = {"id": "phone", "config": {"url": f"{listener.base_url}/hook"}}
        client.post("/api/v1/notifications/targets", json=target).raise_for_status()

    sent = call(account, fetching.upstream, "notify", message="kettle is on", title="Home")
    missing = call(account, fetching.upstream, "notify", message="x", target="pager")

    assert sent == "Notification sent to phone."
    [delivered] = listener.requests_to("/hook")
    assert delivered.json() == {
        "action": "notify",
        "user_id": account.id,
        "message": "kettle is on",
        "title": "Home",
    }
    assert missing == "Notification failed: Notification target not found"
