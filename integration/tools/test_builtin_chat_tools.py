"""Regression: the builtin automation, calendar and file tools a model calls, fixed in 0.11.1.

* `f8ac75d188` (`search_automations`): the folder filter ran whenever `folder_id` was not None,
  and `list_automations` passes `folder_id=""` when the model names no folder, so the query
  became `folder_id IS NULL` and every automation in a folder vanished from the listing.
* `90bb94abf9` (`update_automation`): a blank `folder_id` or `model_id` the model filled in was
  written through, moving the automation out of its folder and clearing its model.
* `1deeaf71d` (PR #27777, `update_calendar_event`): the update form was built with every field,
  so each one the model did not mention was written as None over the stored value.
* `9550731cc1` (issue #27717, `search_calendar_events`): the open-ended upper bound was about
  3.3e19 ns, beyond a signed 64-bit integer, so a search with a start and no end failed; SQLite
  refuses to bind it just as PostgreSQL's bigint does.
* `54cefd2b9` (PR #27642, issue #27641): the file access check and the retrieval tools rebuilt
  the caller from only id and role. The shared-file path failed validation, and the embedding
  call reached an instance forwarding user details without the user's email.

The scripted model calls each tool and the test reads the result sent back, or the stored rows.

Twin of unit/tools/test_builtin_chat_tools.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted: `f8ac75d188` (the blank
folder lists only the loose automation), `90bb94abf9` (the blank folder and model are written),
`1deeaf71d` (the rename is not stored at all), `9550731cc1` (the open-ended search
errors) and `54cefd2b9` (the shared file is refused, the embedding call carries no email).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from harness.actors import create_user
from harness.calendar_api import MINUTE_NS, create_event, default_calendar_id, to_ns
from harness.knowledge_bases import add_text_file, knowledge_base, model_with_knowledge
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# shares one boot with test_connection_headers_and_hosts
FORWARDING = {
    "ENABLE_FORWARD_USER_INFO_HEADERS": "true",
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "ENABLE_API_KEYS": "true",
    "FORWARD_USER_INFO_HEADER_AUTH_TYPE": "X-Caller-Auth",
}


def call(actor, upstream, tool: str, **arguments) -> dict:
    """The JSON a builtin tool returned when the model called it for `actor`."""
    with actor.client() as client:
        return json.loads(run_tool(client, upstream, tool, arguments))


# --- automations: f8ac75d188 and 90bb94abf9 ----------------------------------------------


# first due in 2099, so the scheduler never runs them into another test's script
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"


@pytest.fixture
def automation_owner(make_user):
    """A new admin: automations are an opt-in feature for users."""
    return make_user(role="admin")


@pytest.fixture
def folders(automation_owner) -> list[str]:
    """The ids of the owner's folders "Reports" and "Archive"."""
    with automation_owner.client() as client:
        created = [
            client.post("/api/v1/folders/", json={"name": name}) for name in ("Reports", "Archive")
        ]
    assert all(folder.status_code == 200 for folder in created), [folder.text for folder in created]
    return [folder.json()["id"] for folder in created]


@pytest.fixture
def automations(automation_owner, folders):
    """{kind: id} of an automation in a folder, a loose one and a paused one in the folder."""
    kinds = {
        "in_folder": ("In a folder", folders[0], True),
        "loose": ("Loose", None, True),
        "paused": ("Paused", folders[0], False),
    }
    created = {}
    with automation_owner.client() as client:
        for kind, (name, folder_id, active) in kinds.items():
            form = {
                "name": name,
                "folder_id": folder_id,
                "data": {"prompt": "summarise", "model_id": "mock-model", "rrule": SCHEDULE},
                "is_active": active,
            }
            response = client.post("/api/v1/automations/create", json=form)
            assert response.status_code == 200, response.text
            created[kind] = response.json()["id"]
        yield created
        for automation_id in created.values():
            client.delete(f"/api/v1/automations/{automation_id}/delete")


def _listed_ids(result: dict) -> set[str]:
    return {automation["id"] for automation in result["automations"]}


def _stored_automation(owner, automation_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/automations/{automation_id}").json()


def test_listing_with_a_blank_folder_includes_automations_in_folders(
    automation_owner, automations, upstream
):
    result = call(automation_owner, upstream, "list_automations", folder_id="")

    assert _listed_ids(result) == set(automations.values()), (
        f"a blank folder filter hid the automations that live in a folder: {result}"
    )
    assert result["total"] == 3


def test_a_blank_folder_keeps_the_automation_in_its_folder(
    automation_owner, automations, folders, upstream
):
    call(
        automation_owner,
        upstream,
        "update_automation",
        automation_id=automations["in_folder"],
        name="Nightly digest",
        folder_id="",
    )

    stored = _stored_automation(automation_owner, automations["in_folder"])
    assert stored["name"] == "Nightly digest"
    assert stored["folder_id"] == folders[0], (
        "a blank folder_id from the model moved the automation out of its folder"
    )


def test_a_blank_model_keeps_the_automations_model(automation_owner, automations, upstream):
    call(
        automation_owner,
        upstream,
        "update_automation",
        automation_id=automations["in_folder"],
        model_id="   ",
    )

    stored = _stored_automation(automation_owner, automations["in_folder"])
    assert stored["data"]["model_id"] == "mock-model", (
        f"a blank model_id from the model replaced the automation's model: {stored['data']}"
    )


def test_listing_without_a_folder_includes_every_automation(
    automation_owner, automations, upstream
):
    result = call(automation_owner, upstream, "list_automations")

    assert _listed_ids(result) == set(automations.values())


def test_listing_a_named_folder_filters_to_it(automation_owner, automations, folders, upstream):
    result = call(automation_owner, upstream, "list_automations", folder_id=folders[0])

    assert _listed_ids(result) == {automations["in_folder"], automations["paused"]}


def test_listing_paused_automations_spans_folders(automation_owner, automations, upstream):
    result = call(automation_owner, upstream, "list_automations", status="paused", folder_id="")

    assert _listed_ids(result) == {automations["paused"]}


def test_a_named_folder_and_model_are_applied(automation_owner, automations, folders, upstream):
    archive = folders[1]
    call(
        automation_owner,
        upstream,
        "update_automation",
        automation_id=automations["loose"],
        model_id="other-model",
        folder_id=archive,
    )

    stored = _stored_automation(automation_owner, automations["loose"])
    assert (stored["folder_id"], stored["data"]["model_id"]) == (archive, "other-model")


def test_a_rename_keeps_the_prompt_and_schedule(automation_owner, automations, upstream):
    call(
        automation_owner,
        upstream,
        "update_automation",
        automation_id=automations["loose"],
        name="Renamed",
    )

    stored = _stored_automation(automation_owner, automations["loose"])
    assert stored["name"] == "Renamed"
    assert (stored["data"]["prompt"], stored["data"]["rrule"]) == ("summarise", SCHEDULE)


# --- calendar: 1deeaf71d and 9550731cc1 -------------------------------------------------

EVENT_START = dt.datetime(2026, 11, 3, 9, 0, tzinfo=dt.timezone.utc)
STORED_EVENT = {
    "title": "Standup",
    "description": "daily sync",
    "location": "Room 3",
    "start_at": to_ns(EVENT_START),
    "end_at": to_ns(EVENT_START) + 30 * MINUTE_NS,
}


@pytest.fixture
def calendar_owner(make_user):
    return make_user()


@pytest.fixture
def event_id(calendar_owner):
    """A stored event of the owner's, in their default calendar."""
    with calendar_owner.client() as client:
        created = create_event(client, default_calendar_id(client), **STORED_EVENT)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_event(owner, event_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/calendars/events/{event_id}").json()


def _unchanged(stored: dict, *except_fields: str) -> dict:
    """The stored event's fields that differ from `STORED_EVENT`, ignoring `except_fields`."""
    return {
        field: stored[field]
        for field, value in STORED_EVENT.items()
        if field not in except_fields and stored[field] != value
    }


def test_renaming_an_event_keeps_every_other_field(calendar_owner, event_id, upstream):
    call(
        calendar_owner, upstream, "update_calendar_event", event_id=event_id, title="Standup moved"
    )

    stored = _stored_event(calendar_owner, event_id)
    assert stored["title"] == "Standup moved", f"the rename was not stored: {stored}"
    assert _unchanged(stored, "title") == {}, (
        "fields the model did not mention were overwritten by the rename (#27777)"
    )


@pytest.mark.parametrize("query", [None, "standup"])
def test_an_open_ended_search_finds_the_event(calendar_owner, event_id, upstream, query):
    arguments = {"start": "2026-04-20 00:00", **({"query": query} if query else {})}
    result = call(calendar_owner, upstream, "search_calendar_events", **arguments)

    assert "error" not in result, (
        f"a search with a start and no end failed on its upper bound (#27717): {result}"
    )
    assert [event["id"] for event in result["events"]] == [event_id]


def test_cancelling_an_event_keeps_every_other_field(calendar_owner, event_id, upstream):
    call(calendar_owner, upstream, "update_calendar_event", event_id=event_id, is_cancelled=True)

    stored = _stored_event(calendar_owner, event_id)
    assert stored["is_cancelled"] is True
    assert _unchanged(stored) == {}


def test_every_mentioned_field_is_written(calendar_owner, event_id, upstream):
    call(
        calendar_owner,
        upstream,
        "update_calendar_event",
        event_id=event_id,
        title="Retro",
        description="quarterly",
        location="Room 4",
        all_day=True,
        reminder_minutes=30,
    )

    stored = _stored_event(calendar_owner, event_id)
    written = [stored[field] for field in ("title", "description", "location", "all_day", "meta")]
    assert written == ["Retro", "quarterly", "Room 4", True, {"alert_minutes": 30}]
    assert _unchanged(stored, "title", "description", "location") == {}


def test_moving_an_event_changes_only_its_start(calendar_owner, event_id, upstream):
    call(
        calendar_owner,
        upstream,
        "update_calendar_event",
        event_id=event_id,
        start="2026-11-04 09:00",
    )

    stored = _stored_event(calendar_owner, event_id)
    assert stored["start_at"] == to_ns(EVENT_START + dt.timedelta(days=1))
    assert _unchanged(stored, "start_at") == {}


def test_a_closed_range_around_the_event_finds_it(calendar_owner, event_id, upstream):
    result = call(
        calendar_owner,
        upstream,
        "search_calendar_events",
        start="2026-11-01 00:00",
        end="2026-11-08 00:00",
    )

    assert [event["id"] for event in result["events"]] == [event_id]


def test_a_search_ending_before_the_event_misses_it(calendar_owner, event_id, upstream):
    result = call(calendar_owner, upstream, "search_calendar_events", end="2026-04-27 00:00")

    assert result["events"] == []


def test_an_unparseable_start_is_reported(calendar_owner, upstream):
    result = call(calendar_owner, upstream, "search_calendar_events", start="next tuesday-ish")

    assert "Invalid start datetime" in result["error"]


def test_updating_an_unknown_event_reports_it(calendar_owner, upstream):
    result = call(calendar_owner, upstream, "update_calendar_event", event_id="nope", title="x")

    assert result == {"error": "Event not found"}


# --- the whole user in file access and retrieval: 54cefd2b9 ------------------------------

SHARED_BODY = "the shared body"
PRIVATE_BODY = "the private body"


@pytest.fixture
def shared_library(make_user):
    """A reader with a preset on a knowledge base shared with them, plus two files.

    Yields (reader, model id, shared file id, private file id); the private file sits in a
    second knowledge base the reader cannot open.
    """
    owner, reader = make_user(role="admin"), make_user()
    reader_grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with (
        owner.client() as client,
        knowledge_base(client, "Shared", access_grants=[reader_grant]) as shared_id,
        knowledge_base(client, "Private") as private_id,
    ):
        shared_file = add_text_file(client, shared_id, "shared.md", SHARED_BODY)
        private_file = add_text_file(client, private_id, "private.md", PRIVATE_BODY)
        with model_with_knowledge(client, shared_id) as model_id:
            yield reader, model_id, shared_file, private_file


def _view_file(actor, upstream, preset: str, file_id: str) -> dict:
    with actor.client() as client:
        return json.loads(
            run_tool(client, upstream, "view_file", {"file_id": file_id}, model=preset)
        )


def test_a_file_shared_through_a_knowledge_base_can_be_viewed(shared_library, upstream):
    reader, model_id, shared_file, _ = shared_library

    result = _view_file(reader, upstream, model_id, shared_file)

    assert result.get("content") == SHARED_BODY, (
        f"a reader of the knowledge base could not view its file (#27642): {result}"
    )


def test_a_file_shared_through_a_knowledge_base_can_be_searched(shared_library, upstream):
    reader, model_id, shared_file, _ = shared_library

    with reader.client() as client:
        result = run_tool(
            client,
            upstream,
            "grep_knowledge_files",
            {"pattern": "shared", "file_id": shared_file},
            model=model_id,
        )

    assert SHARED_BODY in result, f"grep over a shared file was refused (#27642): {result}"


@pytest.mark.slow
def test_the_embedding_call_carries_the_whole_user(instance_with):
    forwarding = instance_with(FORWARDING)
    account = create_user(forwarding)

    call(account, forwarding.upstream, "query_knowledge_bases", query="budget")

    embedded = forwarding.upstream.requests_to("/embeddings")
    assert embedded, "query_knowledge_bases never reached the embedding service"
    headers = {name.lower(): value for name, value in embedded[-1].headers.items()}
    assert headers.get("x-openwebui-user-email") == account.email, (
        f"the embedding call did not carry the caller's email (#27641): {headers}"
    )
    assert headers.get("x-openwebui-user-id") == account.id


def test_a_file_in_an_unshared_knowledge_base_stays_hidden(shared_library, upstream):
    reader, model_id, _, private_file = shared_library

    result = _view_file(reader, upstream, model_id, private_file)

    assert "content" not in result, result
    assert PRIVATE_BODY not in json.dumps(result)


def test_the_owner_of_a_file_views_it_without_a_grant(shared_library, upstream):
    reader, model_id, _, _ = shared_library
    with reader.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("mine.md", b"my own body", "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text

    result = _view_file(reader, upstream, model_id, uploaded.json()["id"])

    assert result.get("content") == "my own body"
