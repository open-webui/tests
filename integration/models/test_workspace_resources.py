"""Workspace resource regressions fixed in 0.11.1, seen through the API.

* A note whose markdown body was stored as a dict or list broke the notes page; the note forms
  and model now render such a body as a fenced JSON block and str() anything else
  (`8d1c205d8e`, #28222).
* The folder duplicate-name check used `ilike(name)`, so `%` and `_` in a new name matched other
  folders and a legitimate name was refused (`1d1c14bd7`, #28695, #28694).
* A skill id outside the URL-path slug charset was stored and then unreachable; creation now
  refuses it with a 400 (`3df485582`, #27660, #27655).
* The read-only filter left out public grants, so a note shared publicly for reading was
  reachable by link but listed nowhere (`c0d09a5de`, #27637, #27487).
* Two folder queries had no ORDER BY, so shared folders came back in storage order; they are
  now listed most recently updated first (`6db64c485`, #28804). The reshuffling the PR fixed
  is PostgreSQL's; on SQLite the test pins the order the fix promises.

Twin of unit/models/test_workspace_resources.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the note sanitizer a
no-op, the folder check back on `ilike`, the skill id check removed, the public grant dropped
from the read-only filter, the ORDER BY removed from either folder query): the note body comes
back a dict, "Notes_" is refused, "my/skill" is stored, the public note is not listed and the
renamed folder is not listed first.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


def _grant(user_id: str, permission: str) -> dict:
    return {"principal_type": "user", "principal_id": user_id, "permission": permission}


# ---------------------------------------------------------------- note bodies


def _stored_note_data(client: httpx.Client, data) -> dict:
    created = client.post("/api/v1/notes/create", json={"title": "shape", "data": data})
    assert created.status_code == 200, created.text
    stored = client.get(f"/api/v1/notes/{created.json()['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()["data"]


@pytest.mark.parametrize(
    ("body", "rendered"),
    [
        ({"type": "doc", "text": "hi"}, '```json\n{\n  "type": "doc",\n  "text": "hi"\n}\n```'),
        ([1, 2], "```json\n[\n  1,\n  2\n]\n```"),
        (42, "42"),
    ],
    ids=["dict", "list", "number"],
)
def test_a_note_body_that_is_not_text_comes_back_as_markdown(make_user, body, rendered):
    with make_user().client() as client:
        data = _stored_note_data(client, {"content": {"md": body}})

    assert data["content"]["md"] == rendered, (
        f"a {type(body).__name__} note body came back as {data['content']['md']!r}, which the "
        "notes page cannot render"
    )


def test_note_data_that_is_plain_text_becomes_the_body(make_user):
    with make_user().client() as client:
        data = _stored_note_data(client, "just some text")

    assert data == {"content": {"md": "just some text"}}


def test_an_update_with_a_dict_body_is_rendered_too(make_user):
    with make_user().client() as client:
        created = client.post("/api/v1/notes/create", json={"title": "shape", "data": None})
        note_id = created.json()["id"]
        updated = client.post(
            f"/api/v1/notes/{note_id}/update",
            json={
                "title": "shape",
                "data": {"versions": [1], "content": {"html": "<p>x</p>", "md": {"a": "über"}}},
            },
        )
        assert updated.status_code == 200, updated.text
        data = client.get(f"/api/v1/notes/{note_id}").json()["data"]

    assert data["content"]["md"] == '```json\n{\n  "a": "über"\n}\n```'
    assert (data["versions"], data["content"]["html"]) == ([1], "<p>x</p>")


@pytest.mark.parametrize(
    "data",
    [{"content": {"md": "# already markdown"}}, {"content": {"html": "<p>x</p>"}}, {}],
    ids=["markdown", "html-only", "empty"],
)
def test_well_formed_note_data_is_stored_as_sent(make_user, data):
    with make_user().client() as client:
        assert _stored_note_data(client, data) == data


# ---------------------------------------------------------------- folder names


def _create_folder(client: httpx.Client, name: str, parent_id: str | None = None):
    return client.post("/api/v1/folders/", json={"name": name, "parent_id": parent_id})


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [("Notes1", "Notes_"), ("Archive 2024", "Archive%"), ("Report", "%")],
)
def test_a_folder_name_with_a_wildcard_is_not_taken_by_another(make_user, existing, candidate):
    with make_user().client() as client:
        assert _create_folder(client, existing).status_code == 200

        created = _create_folder(client, candidate)

    assert created.status_code == 200, (
        f"{candidate!r} was refused as a duplicate of {existing!r}: {created.text}"
    )


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [("Projects", "Projects"), ("Projects", "projects"), ("Projects", "PROJECTS"), ("A_%", "a_%")],
)
def test_the_same_name_in_any_case_is_still_a_duplicate(make_user, existing, candidate):
    with make_user().client() as client:
        assert _create_folder(client, existing).status_code == 200

        assert _create_folder(client, candidate).status_code == 400


def test_a_similar_name_or_another_parent_is_not_a_duplicate(make_user):
    with make_user().client() as client:
        parent = _create_folder(client, "Projects").json()

        assert _create_folder(client, "Project").status_code == 200
        assert _create_folder(client, "Projects", parent_id=parent["id"]).status_code == 200


# ---------------------------------------------------------------- skill ids


@pytest.fixture
def skill_ids(admin):
    """`create(id)` posts a skill as the admin; every stored one is deleted afterwards."""
    stored: list[str] = []
    client = admin.client()

    def create(skill_id: str) -> httpx.Response:
        created = client.post(
            "/api/v1/skills/create",
            json={"id": skill_id, "name": f"Skill {uuid.uuid4().hex[:8]}", "content": "body"},
        )
        if created.status_code == 200:
            stored.append(created.json()["id"])
        return created

    yield create
    for skill_id in stored:
        client.delete(f"/api/v1/skills/id/{skill_id}/delete")
    client.close()


@pytest.mark.parametrize(
    "unreachable", ["my/skill", "../etc", "skill?x", "skill#1", "skill.v1", "скилл"]
)
def test_a_skill_id_that_cannot_be_addressed_is_refused(skill_ids, unreachable):
    created = skill_ids(f"{unreachable}{uuid.uuid4().hex[:4]}")

    assert created.status_code == 400, f"{unreachable!r} was stored: {created.text}"
    assert "Invalid skill ID" in created.json()["detail"]


def test_a_skill_id_with_spaces_is_still_slugged(skill_ids):
    suffix = uuid.uuid4().hex[:6]
    created = skill_ids(f"My Skill {suffix}")

    assert created.status_code == 200, created.text
    assert created.json()["id"] == f"my-skill-{suffix}"


def test_a_taken_skill_id_still_says_so(skill_ids):
    skill_id = f"skill_{uuid.uuid4().hex[:6]}"
    assert skill_ids(skill_id).status_code == 200

    again = skill_ids(skill_id)

    assert again.status_code == 400
    assert "Invalid skill ID" not in again.json()["detail"]


# ---------------------------------------------------------------- read-only notes


def _shared_note(owner, *grants: dict) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/notes/create", json={"title": "shared", "data": None})
        assert created.status_code == 200, created.text
        note_id = created.json()["id"]
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": list(grants)}
        )
        assert shared.status_code == 200, shared.text
    return note_id


def _read_only_note_ids(viewer) -> set[str]:
    with viewer.client() as client:
        listed = client.get("/api/v1/notes/search", params={"permission": "read_only"})
    assert listed.status_code == 200, listed.text
    return {note["id"] for note in listed.json()["items"]}


def test_a_publicly_shared_note_is_listed_as_read_only(admin, make_user):
    note_id = _shared_note(admin, EVERYONE_READS)

    assert note_id in _read_only_note_ids(make_user()), (
        "a note shared publicly for reading is reachable by link but listed nowhere"
    )


def test_a_note_shared_for_reading_with_one_person_is_listed(admin, make_user):
    viewer = make_user()
    note_id = _shared_note(admin, _grant(viewer.id, "read"))

    assert note_id in _read_only_note_ids(viewer)


def test_writable_and_own_notes_stay_out_of_the_read_only_list(admin, make_user):
    viewer = make_user()
    writable = _shared_note(admin, _grant(viewer.id, "read"), _grant(viewer.id, "write"))
    own = _shared_note(viewer, EVERYONE_READS)

    listed = _read_only_note_ids(viewer)

    assert writable not in listed
    assert own not in listed


# ---------------------------------------------------------------- shared folder order


def test_shared_child_folders_are_listed_most_recently_updated_first(make_user):
    owner, viewer = make_user(), make_user()
    with owner.client() as client:
        parent = _create_folder(client, "Shared").json()
        first = _create_folder(client, "First", parent_id=parent["id"]).json()
        second = _create_folder(client, "Second", parent_id=parent["id"]).json()
        client.post(
            f"/api/v1/folders/{parent['id']}/access/update",
            json={"access_grants": [_grant(viewer.id, "read")]},
        ).raise_for_status()
        time.sleep(1.1)  # updated_at has one-second resolution
        client.post(
            f"/api/v1/folders/{second['id']}/update", json={"name": "Second, renamed"}
        ).raise_for_status()

    with viewer.client() as client:
        shared = client.get("/api/v1/folders/shared")
    assert shared.status_code == 200, shared.text
    children = [folder["id"] for folder in shared.json() if folder["parent_id"] == parent["id"]]

    assert children == [second["id"], first["id"]], (
        "the shared child folders came back in storage order, not most recently updated first"
    )


def test_shared_folders_are_listed_most_recently_updated_first(make_user):
    owner, viewer = make_user(), make_user()
    with owner.client() as client:
        folders = [_create_folder(client, name).json() for name in ("One", "Two")]
        for folder in folders:
            client.post(
                f"/api/v1/folders/{folder['id']}/access/update",
                json={"access_grants": [_grant(viewer.id, "read")]},
            ).raise_for_status()
        time.sleep(1.1)
        # the id SQLite's primary-key index lists last, so storage order cannot put it first
        renamed = max(folders, key=lambda folder: folder["id"])
        client.post(
            f"/api/v1/folders/{renamed['id']}/update", json={"name": "Renamed"}
        ).raise_for_status()

    with viewer.client() as client:
        shared = client.get("/api/v1/folders/shared").json()

    assert [folder["id"] for folder in shared][0] == renamed["id"], (
        "the shared folders came back in storage order, not most recently updated first"
    )
