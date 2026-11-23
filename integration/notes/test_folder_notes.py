"""Journey: notes filed in a chat folder, seen through the API.

A note is filed in a folder when it is created there (`folder_id` on `/api/v1/notes/create`) or
moved there (`POST /api/v1/notes/{id}/folder`, `null` moves it out). `GET
/api/v1/folders/{id}/notes` lists the folder's own notes, the most recently updated first, for
anyone who can read the folder. A folder shared for reading shows its notes to the member, whoever
wrote them, read-only; one shared for writing also takes the member's new notes. Nobody files a
note into a folder they cannot write to. Taking the share back takes the notes away again.
Deleting a folder keeps its notes, out of any folder, unless the delete takes its contents too.
Moving a folder keeps its notes in it. The search window finds a folder's notes for the owner and
the member, also when scoped to the folder. Without the Notes permission the list is refused.
Twin of the browser journeys in e2e/notes/test_folder_notes.py.

Discriminates: journey, no fix. Passes on dev d599f7a08, five runs in a row. In backend copies of
it: the folder's note list ignoring the folder turns the listing, move, reader, writer, refusal,
moved folder and permission tests red; the list sorted oldest first turns the listing test red; a
note no longer readable through its folder turns the reader, writer and unshare tests red; the
list without the folder read check turns the stranger and unshare tests red; filing without the
write check turns the refusal test red; a folder delete leaving its notes on the deleted folder
turns the kept contents case red; the search leaving out folder notes turns the search test red;
the list without the Notes check turns the permission test red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor
from harness.workspace_search import found, unique_word

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}
PERMISSIONS = "/api/v1/users/default/permissions"


def _ok(response: httpx.Response) -> dict:
    assert response.status_code == 200, (
        f"{response.request.url}: {response.status_code} {response.text}"
    )
    return response.json()


def _folder(owner: Actor, name: str = "", parent_id: str | None = None) -> str:
    with owner.client() as client:
        return _ok(
            client.post(
                "/api/v1/folders/",
                json={"name": name or f"Project {uuid.uuid4().hex[:6]}", "parent_id": parent_id},
            )
        )["id"]


def _note(author: Actor, title: str, folder_id: str | None = None, text: str = "") -> str:
    body = {"title": title, "data": {"content": {"md": text or title}}, "access_grants": []}
    if folder_id:
        body["folder_id"] = folder_id
    with author.client() as client:
        return _ok(client.post("/api/v1/notes/create", json=body))["id"]


def _folder_notes(actor: Actor, folder_id: str) -> list[dict]:
    with actor.client() as client:
        return _ok(client.get(f"/api/v1/folders/{folder_id}/notes"))["items"]


def _titles(actor: Actor, folder_id: str) -> list[str]:
    return [note["title"] for note in _folder_notes(actor, folder_id)]


def _stored_folder(actor: Actor, note_id: str) -> str | None:
    with actor.client() as client:
        return _ok(client.get(f"/api/v1/notes/{note_id}")).get("folder_id")


def _move(actor: Actor, note_id: str, folder_id: str | None) -> httpx.Response:
    with actor.client() as client:
        return client.post(f"/api/v1/notes/{note_id}/folder", json={"folder_id": folder_id})


def _share(owner: Actor, folder_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        _ok(
            client.post(
                f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
            )
        )


@pytest.fixture
def crew(admin, make_user):
    """An owner allowed to share folders, a reading and a writing member, and a stranger."""
    owner, reader, writer, stranger = make_user(), make_user(), make_user(), make_user()
    make_group(admin, [owner], FOLDER_SHARING)
    return owner, reader, writer, stranger


def _shared_folder(owner: Actor, reader: Actor, writer: Actor) -> str:
    folder_id = _folder(owner)
    _share(
        owner,
        folder_id,
        [
            grant("user", reader.id, "read"),
            grant("user", writer.id, "read"),
            grant("user", writer.id, "write"),
        ],
    )
    return folder_id


def test_a_folder_lists_its_own_notes_newest_first(make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner)
    subfolder_id = _folder(owner, parent_id=folder_id)
    first = _note(owner, f"First {tag}", folder_id)
    _note(owner, f"Second {tag}", folder_id)
    _note(owner, f"Loose {tag}")
    _note(owner, f"Deep {tag}", subfolder_id)

    with owner.client() as client:
        listed = _ok(client.get(f"/api/v1/folders/{folder_id}/notes"))
    assert [note["title"] for note in listed["items"]] == [f"Second {tag}", f"First {tag}"]
    assert listed["total"] == 2, f"the folder's note count reads {listed['total']}"
    assert _stored_folder(owner, first) == folder_id
    assert _titles(owner, subfolder_id) == [f"Deep {tag}"]

    with owner.client() as client:
        _ok(client.post(f"/api/v1/notes/{first}/update", json={"title": f"First edited {tag}"}))
    assert _titles(owner, folder_id) == [f"First edited {tag}", f"Second {tag}"], (
        "the note edited last is not listed first"
    )


def test_a_note_moves_into_a_folder_and_out_again_and_stays_in_notes(make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner)
    note_id = _note(owner, f"Wanderer {tag}")
    assert _titles(owner, folder_id) == []

    _ok(_move(owner, note_id, folder_id))
    assert _titles(owner, folder_id) == [f"Wanderer {tag}"]
    assert _stored_folder(owner, note_id) == folder_id

    _ok(_move(owner, note_id, None))
    assert _titles(owner, folder_id) == []
    assert _stored_folder(owner, note_id) is None
    with owner.client() as client:
        mine = [note["id"] for note in _ok(client.get("/api/v1/notes/"))]
    assert note_id in mine, "a note moved out of its folder is gone from Notes"


def test_a_reading_member_reads_every_note_in_the_folder_and_changes_none(crew):
    owner, reader, writer, _ = crew
    tag = uuid.uuid4().hex[:6]
    folder_id = _shared_folder(owner, reader, writer)
    owners_note = _note(owner, f"Owner's plan {tag}", folder_id, f"owner text {tag}")
    writers_note = _note(writer, f"Writer's draft {tag}", folder_id)

    listed = {note["title"]: note for note in _folder_notes(reader, folder_id)}
    assert set(listed) == {f"Owner's plan {tag}", f"Writer's draft {tag}"}
    assert not any(note["write_access"] for note in listed.values()), (
        "a read-only member is offered to edit a note in the folder"
    )
    with reader.client() as client:
        opened = client.get(f"/api/v1/notes/{owners_note}")
        assert opened.status_code == 200, f"the member cannot open the owner's note: {opened.text}"
        assert opened.json()["data"]["content"]["md"] == f"owner text {tag}"
        assert client.get(f"/api/v1/notes/{writers_note}").status_code == 200

        renamed = client.post(f"/api/v1/notes/{owners_note}/update", json={"title": "taken over"})
        deleted = client.delete(f"/api/v1/notes/{owners_note}/delete")
        moved_out = client.post(f"/api/v1/notes/{owners_note}/folder", json={"folder_id": None})
    assert renamed.status_code != 200, f"a read-only member renamed the note: {renamed.text}"
    assert deleted.status_code != 200, f"a read-only member deleted the note: {deleted.text}"
    assert moved_out.status_code != 200, f"a read-only member moved the note: {moved_out.text}"
    assert _titles(owner, folder_id).count(f"Owner's plan {tag}") == 1


def test_a_writing_member_files_notes_the_owner_then_finds(crew):
    owner, reader, writer, _ = crew
    tag = uuid.uuid4().hex[:6]
    folder_id = _shared_folder(owner, reader, writer)
    created = _note(writer, f"Created there {tag}", folder_id)
    moved = _note(writer, f"Moved there {tag}")
    _ok(_move(writer, moved, folder_id))

    assert set(_titles(owner, folder_id)) == {f"Created there {tag}", f"Moved there {tag}"}
    with owner.client() as client:
        assert client.get(f"/api/v1/notes/{created}").status_code == 200


def test_nobody_files_a_note_into_a_folder_they_cannot_write(crew):
    owner, reader, writer, stranger = crew
    tag = uuid.uuid4().hex[:6]
    folder_id = _shared_folder(owner, reader, writer)
    private_folder = _folder(owner)
    _note(owner, f"Kept {tag}", private_folder)

    for name, actor, target in (
        ("read-only member", reader, folder_id),
        ("stranger", stranger, private_folder),
    ):
        with actor.client() as client:
            created = client.post(
                "/api/v1/notes/create",
                json={
                    "title": f"Pushed {tag}",
                    "data": {"content": {"md": "x"}},
                    "folder_id": target,
                },
            )
        assert created.status_code != 200, f"a {name} created a note in the folder: {created.text}"
        own = _note(actor, f"Own {name} {tag}")
        moved = _move(actor, own, target)
        assert moved.status_code != 200, f"a {name} moved a note into the folder: {moved.text}"
        assert _stored_folder(actor, own) is None
    assert _titles(owner, folder_id) == []
    assert _titles(owner, private_folder) == [f"Kept {tag}"]


def test_a_stranger_neither_lists_nor_opens_a_folders_notes(crew):
    owner, _, _, stranger = crew
    folder_id = _folder(owner)
    note_id = _note(owner, f"Private {uuid.uuid4().hex[:6]}", folder_id)
    with stranger.client() as client:
        listed = client.get(f"/api/v1/folders/{folder_id}/notes")
        opened = client.get(f"/api/v1/notes/{note_id}")
    assert listed.status_code in (401, 403, 404), f"a stranger lists the notes: {listed.text}"
    assert opened.status_code in (401, 403, 404), f"a stranger opens the note: {opened.text}"


def test_taking_the_share_back_takes_the_notes_away(crew):
    owner, reader, writer, _ = crew
    folder_id = _shared_folder(owner, reader, writer)
    note_id = _note(owner, f"Briefing {uuid.uuid4().hex[:6]}", folder_id)
    with reader.client() as client:
        assert client.get(f"/api/v1/notes/{note_id}").status_code == 200

    _share(owner, folder_id, [])
    with reader.client() as client:
        listed = client.get(f"/api/v1/folders/{folder_id}/notes")
        opened = client.get(f"/api/v1/notes/{note_id}")
    assert listed.status_code in (401, 403, 404), (
        f"the former member lists the notes: {listed.text}"
    )
    assert opened.status_code in (401, 403, 404), f"the former member opens the note: {opened.text}"


@pytest.mark.parametrize("delete_contents", [False, True], ids=["keep contents", "delete contents"])
def test_deleting_a_folder_keeps_or_deletes_its_notes(make_user, delete_contents):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner)
    subfolder_id = _folder(owner, parent_id=folder_id)
    top = _note(owner, f"Top {tag}", folder_id)
    deep = _note(owner, f"Deep {tag}", subfolder_id)

    with owner.client() as client:
        _ok(
            client.delete(
                f"/api/v1/folders/{folder_id}", params={"delete_contents": delete_contents}
            )
        )
        for note_id in (top, deep):
            stored = client.get(f"/api/v1/notes/{note_id}")
            if delete_contents:
                assert stored.status_code == 404, (
                    f"a note survived its folder's delete: {stored.text}"
                )
            else:
                assert stored.status_code == 200, f"a note went with its folder: {stored.text}"
                assert stored.json()["folder_id"] is None, (
                    "a kept note still names the deleted folder"
                )


def test_a_moved_folder_keeps_its_notes(make_user):
    owner = make_user()
    tag = uuid.uuid4().hex[:6]
    folder_id = _folder(owner)
    new_parent = _folder(owner)
    note_id = _note(owner, f"Carried {tag}", folder_id)
    with owner.client() as client:
        _ok(
            client.post(
                f"/api/v1/folders/{folder_id}/update/parent", json={"parent_id": new_parent}
            )
        )
    assert _titles(owner, folder_id) == [f"Carried {tag}"]
    assert _stored_folder(owner, note_id) == folder_id


def test_the_search_window_finds_a_folders_notes_for_owner_and_member(crew):
    owner, reader, writer, stranger = crew
    word = unique_word()
    folder_id = _shared_folder(owner, reader, writer)
    note_id = _note(owner, "Field notes", folder_id, f"the {word} migrates at dawn")
    result = f"note:{note_id}"

    assert result in found(owner, word)
    assert result in found(owner, word, scope=f"folder:{folder_id}")
    assert result in found(reader, word), "a member of the shared folder does not find its note"
    assert result in found(reader, word, scope=f"folder:{folder_id}")
    assert result not in found(stranger, word)


def test_without_the_notes_permission_the_folder_notes_are_refused(admin, make_user, preserve):
    owner = make_user()
    folder_id = _folder(owner)
    _note(owner, f"Plan {uuid.uuid4().hex[:6]}", folder_id)
    assert len(_folder_notes(owner, folder_id)) == 1
    preserve("permissions")
    with admin.client() as client:
        permissions = _ok(client.get(PERMISSIONS))
        permissions["features"] = {**permissions["features"], "notes": False}
        _ok(client.post(PERMISSIONS, json=permissions))
    with owner.client() as client:
        listed = client.get(f"/api/v1/folders/{folder_id}/notes")
    assert listed.status_code in (401, 403), f"the notes are listed without Notes: {listed.text}"
