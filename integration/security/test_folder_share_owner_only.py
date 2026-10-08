"""Regression: on a shared folder only its owner or an admin can change its sharing (8145774e3).

The owner shares a folder with a writer (read and write) and a reader, the way the Share dialog
does. The writer's Share dialog request (`POST /folders/{id}/access/update`) is refused with 403
and the grants the owner reads back (`GET /folders/{id}`, as the dialog loads them) are
unchanged; the owner and an admin still change them. Write access keeps what it is for: the
writer renames the folder and edits its system prompt through the folder edit dialog's request.

Discriminates: in a backend copy with the write-grant branch restored in
`update_folder_access_by_id` (a user with write access passes the check) the refusal test goes red
at the writer's 200 and the rewritten grants.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import grant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def shared(make_user):
    """An owner's folder shared with a writer and a reader; yields them and the folder id."""
    owner, writer, reader = make_user(), make_user(), make_user()
    grants = [
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
        grant("user", reader.id, "read"),
    ]
    with owner.client() as client:
        created = client.post(
            "/api/v1/folders/",
            json={
                "name": f"Harbour {uuid.uuid4().hex[:6]}",
                "data": {"system_prompt": "Be brief."},
            },
        )
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        saved = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
        assert saved.status_code == 200, saved.text
    return owner, writer, reader, folder_id


def stored_grants(owner, folder_id: str) -> set[tuple[str, str]]:
    with owner.client() as client:
        folder = client.get(f"/api/v1/folders/{folder_id}")
    assert folder.status_code == 200, folder.text
    return {
        (entry["principal_id"], entry["permission"]) for entry in folder.json()["access_grants"]
    }


def share(actor, folder_id: str, grants: list[dict]):
    with actor.client() as client:
        return client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )


def test_a_writer_cannot_change_the_sharing_of_the_folder(shared, make_user):
    owner, writer, reader, folder_id = shared
    before = stored_grants(owner, folder_id)
    outsider = make_user()
    widened = [
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
        grant("user", outsider.id, "read"),
    ]

    attempt = share(writer, folder_id, widened)

    assert attempt.status_code == 403, attempt.text
    assert stored_grants(owner, folder_id) == before
    with outsider.client() as client:
        assert client.get(f"/api/v1/folders/{folder_id}").status_code == 404

    removal = share(writer, folder_id, [])

    assert removal.status_code == 403, removal.text
    assert stored_grants(owner, folder_id) == before


def test_the_owner_and_an_admin_still_change_the_sharing(shared, admin, make_user):
    owner, writer, reader, folder_id = shared
    newcomer = make_user()

    narrowed = share(owner, folder_id, [grant("user", reader.id, "read")])

    assert narrowed.status_code == 200, narrowed.text
    assert stored_grants(owner, folder_id) == {(reader.id, "read")}

    widened = share(admin, folder_id, [grant("user", newcomer.id, "read")])

    assert widened.status_code == 200, widened.text
    assert stored_grants(owner, folder_id) == {(newcomer.id, "read")}


def test_a_writer_still_renames_the_folder_and_edits_its_prompt(shared):
    owner, writer, reader, folder_id = shared
    renamed = f"Quay {uuid.uuid4().hex[:6]}"

    with writer.client() as client:
        edited = client.post(
            f"/api/v1/folders/{folder_id}/update",
            json={"name": renamed, "data": {"system_prompt": "Answer in rhyme."}},
        )

    assert edited.status_code == 200, edited.text
    with owner.client() as client:
        folder = client.get(f"/api/v1/folders/{folder_id}").json()
    assert folder["name"] == renamed
    assert folder["data"]["system_prompt"] == "Answer in rhyme."
    assert stored_grants(owner, folder_id) == {
        (writer.id, "read"),
        (writer.id, "write"),
        (reader.id, "read"),
    }
