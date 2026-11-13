"""Journey: a shared folder's automations, who files, lists, pauses and runs them, and their chats.

Since 6378f94a6 an automation lives in any folder its creator may write to, a shared folder
included, and a member who may only read it is refused with 404. The folder's automations route,
`GET /api/v1/folders/{id}/automations`, lists every automation in the folder, oldest first, whoever
made it, to anyone who may read the folder, with its creator's name and what the caller may do: only
the creator edits it (`can_edit`), while the creator and everyone who may write to the folder pause,
run and read the runs of it (`can_manage`). Editing and deleting stay the creator's. A run files its
chat in the folder while the creator may still write there, and the folder's chat list marks that
chat with the automation that made it (`automation_id`). Once the creator lost write access, the
next run lands outside the folder and the automation leaves it. Who reaches the list at all is
pinned by integration/security/test_folder_access_matrix.py; an automation outside any folder stays
its creator's alone (integration/security/test_automation_access_matrix.py).

Discriminates: journey, no fix. All pass on dev 485103cde. In backend copies of it: the folder check
for a new automation looking only at the caller's own folders turns the filing test red (the writer
gets 404); every listed automation answered `can_edit` turns the listing test red; the manage check
falling back to the creator alone turns the pause and run test red (the writer gets 404); the chat
meta no longer carrying the automation turns the run test red (no `automation_id`); the run's folder
check skipped turns the lost access test red (the chat lands in the folder).
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

# a far-off start, so no automation here runs on its own schedule
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
RUN_WAIT = 30.0


@pytest.fixture
def automations_allowed(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text


@pytest.fixture
def crew(make_user, automations_allowed):
    """An owner's folder shared with a writer and a reader; their automations go afterwards."""
    owner, writer, reader = make_user(), make_user(), make_user()
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"Rota {uuid.uuid4().hex[:6]}"})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={
                "access_grants": [
                    grant("user", writer.id, "read"),
                    grant("user", writer.id, "write"),
                    grant("user", reader.id, "read"),
                ]
            },
        )
    assert shared.status_code == 200, shared.text
    yield folder_id, owner, writer, reader
    for account in (owner, writer, reader):
        with account.client() as client:
            for item in client.get("/api/v1/automations/list").json().get("items", []):
                client.delete(f"/api/v1/automations/{item['id']}/delete")


def _create(account: Actor, folder_id: str | None, name: str = "") -> httpx.Response:
    with account.client() as client:
        return client.post(
            "/api/v1/automations/create",
            json={
                "name": name or f"digest {uuid.uuid4().hex[:6]}",
                "is_active": True,
                "folder_id": folder_id,
                "data": {
                    "prompt": f"Summarise the harbour log, batch {uuid.uuid4().hex[:6]}.",
                    "model_id": MOCK_MODEL_ID,
                    "rrule": SCHEDULE,
                },
            },
        )


def _created(account: Actor, folder_id: str | None, name: str = "") -> dict:
    created = _create(account, folder_id, name)
    assert created.status_code == 200, created.text
    return created.json()


def _folder_automations(account: Actor, folder_id: str) -> dict[str, dict]:
    with account.client() as client:
        listed = client.get(f"/api/v1/folders/{folder_id}/automations")
    assert listed.status_code == 200, listed.text
    return {item["name"]: item for item in listed.json()}


def _run_and_wait(account: Actor, automation: dict, upstream) -> dict:
    """Run now as `account` and return the recorded run once its reply is in."""
    prompt = automation["data"]["prompt"]
    upstream.queue(reply.text("All quiet on the quay.", match=reply.answering(prompt)))
    with account.client() as client:
        started = client.post(f"/api/v1/automations/{automation['id']}/run")
        assert started.status_code == 200, started.text
        deadline = time.monotonic() + RUN_WAIT
        while time.monotonic() < deadline:
            runs = client.get(f"/api/v1/automations/{automation['id']}/runs").json()
            if runs:
                assert runs[0]["status"] == "success", runs[0]
                return runs[0]
            time.sleep(0.2)
    raise AssertionError(f"no run was recorded within {RUN_WAIT}s")


def test_a_writing_member_files_an_automation_in_the_shared_folder_and_a_reader_cannot(crew):
    folder_id, _, writer, reader = crew

    filed = _create(writer, folder_id)
    refused = _create(reader, folder_id)

    assert filed.status_code == 200, f"a writing member could not file an automation: {filed.text}"
    assert filed.json()["folder_id"] == folder_id, filed.json()
    assert refused.status_code == 404, f"a reading member filed an automation: {refused.text}"
    with reader.client() as client:
        own = client.get("/api/v1/automations/list").json()["items"]
    assert own == [], f"the refused automation was saved: {own}"


def test_the_folder_lists_every_automation_with_its_creator_and_what_each_member_may_do(crew):
    folder_id, owner, writer, reader = crew
    owners_name, writers_name = (
        f"Tide table {uuid.uuid4().hex[:4]}",
        f"Ferry {uuid.uuid4().hex[:4]}",
    )
    _created(owner, folder_id, owners_name)
    _created(writer, folder_id, writers_name)
    _created(owner, None, f"Elsewhere {uuid.uuid4().hex[:4]}")

    def rights(account: Actor) -> dict[str, tuple]:
        listed = _folder_automations(account, folder_id)
        return {
            name: (item["user_name"], item["can_edit"], item["can_manage"])
            for name, item in listed.items()
        }

    with owner.client() as client:
        names = [
            item["name"] for item in client.get(f"/api/v1/folders/{folder_id}/automations").json()
        ]

    assert names == [owners_name, writers_name], (
        f"the folder did not list its automations oldest first, and only them: {names}"
    )
    assert rights(owner) == {
        owners_name: (owner.name, True, True),
        writers_name: (writer.name, False, True),
    }, "the owner's view of the folder's automations"
    assert rights(writer) == {
        owners_name: (owner.name, False, True),
        writers_name: (writer.name, True, True),
    }, "the writing member's view of the folder's automations"
    assert rights(reader) == {
        owners_name: (owner.name, False, False),
        writers_name: (writer.name, False, False),
    }, "the reading member's view of the folder's automations"


def test_a_writing_member_pauses_and_runs_the_owners_automation_but_cannot_edit_it(crew, upstream):
    folder_id, owner, writer, reader = crew
    automation = _created(owner, folder_id)
    path = f"/api/v1/automations/{automation['id']}"

    with reader.client() as client:
        reader_toggle = client.post(f"{path}/toggle")
        reader_runs = client.get(f"{path}/runs")
    with writer.client() as client:
        paused = client.post(f"{path}/toggle")
        edited = client.post(
            f"{path}/update",
            json={"name": "taken over", "is_active": True, "data": automation["data"]},
        )
        deleted = client.delete(f"{path}/delete")
    run = _run_and_wait(writer, automation, upstream)
    with owner.client() as client:
        stored = client.get(path).json()

    assert (reader_toggle.status_code, reader_runs.status_code) == (404, 404), (
        f"a reading member managed the owner's automation: {reader_toggle.text}, {reader_runs.text}"
    )
    assert paused.status_code == 200, f"a writing member could not pause it: {paused.text}"
    assert (edited.status_code, deleted.status_code) == (404, 404), (
        f"a writing member edited or deleted the owner's automation: {edited.text} / {deleted.text}"
    )
    assert (stored["name"], stored["is_active"]) == (automation["name"], False), (
        f"the owner's automation after the member's pause: {stored}"
    )
    assert run["chat_id"], run


def test_a_run_files_its_chat_in_the_folder_marked_with_its_automation(crew, upstream):
    folder_id, owner, writer, _ = crew
    automation = _created(writer, folder_id)

    run = _run_and_wait(writer, automation, upstream)
    with owner.client() as client:
        listed = client.get(
            f"/api/v1/folders/{folder_id}/shared/chats", params={"workspace": "true", "page": 1}
        )
    assert listed.status_code == 200, listed.text
    chats = {chat["id"]: chat for chat in listed.json()["chats"]}

    assert run["chat_id"] in chats, (
        f"the run's chat is not in the folder the owner sees: {sorted(chats)}"
    )
    assert chats[run["chat_id"]]["automation_id"] == automation["id"], chats[run["chat_id"]]
    assert chats[run["chat_id"]]["user_id"] == writer.id, chats[run["chat_id"]]


def test_a_run_after_the_creator_lost_write_access_lands_outside_the_folder(crew, upstream):
    folder_id, owner, writer, _ = crew
    automation = _created(writer, folder_id)
    with owner.client() as client:
        narrowed = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": [grant("user", writer.id, "read")]},
        )
    assert narrowed.status_code == 200, narrowed.text

    run = _run_and_wait(writer, automation, upstream)
    with writer.client() as client:
        chat = client.get(f"/api/v1/chats/{run['chat_id']}").json()
        moved = client.get(f"/api/v1/automations/{automation['id']}").json()

    assert chat["folder_id"] is None, (
        f"a run filed its chat in a folder its creator may no longer write to: {chat['folder_id']}"
    )
    assert moved["folder_id"] is None, f"the automation stayed in the folder: {moved}"
