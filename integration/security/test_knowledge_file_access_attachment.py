"""A file taken out of a knowledge base stops being reachable through that knowledge base.

Regression for open-webui/open-webui#29937, fixed in fbc489726. Adding a file to a knowledge base
stamps its id on the file as `meta.collection_name`, and removing the file with
`delete_file=false` leaves that value behind. `has_access_to_file` still honoured it, so whoever
could open the knowledge base kept reading the detached file, and a write grant on the knowledge
base let them delete it. The fix resolves knowledge access from the attachment alone.

Twin of unit/security/test_knowledge_file_access_attachment.py.

Discriminates: passes on bbfa876af, fails with the `collection_name` branch restored in
`utils/access_control/files.py` (the reader still reads the detached file, the writer deletes it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import httpx
import pytest

from harness.actors import Actor

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

READ_ROUTES = [
    "/api/v1/files/{id}",
    "/api/v1/files/{id}/data/content",
    "/api/v1/files/{id}/content",
]


@dataclass
class SharedFile:
    file_id: str
    knowledge_id: str
    reader: Actor
    writer: Actor


def upload_text(client: httpx.Client, name: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def grant(actor: Actor, *permissions: str) -> list[dict]:
    return [
        {"principal_type": "user", "principal_id": actor.id, "permission": permission}
        for permission in permissions
    ]


@pytest.fixture
def shared_file(admin: Actor, make_user) -> Iterator[SharedFile]:
    """The admin's file in the admin's knowledge base, readable by one user, writable by another."""
    reader, writer = make_user(), make_user()
    with admin.client() as client:
        file_id = upload_text(client, "handbook.txt", "The vault code is 4711.")
        created = client.post(
            "/api/v1/knowledge/create",
            json={
                "name": "Handbook",
                "description": "",
                "access_grants": grant(reader, "read") + grant(writer, "read", "write"),
            },
        )
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        added = client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})
        assert added.status_code == 200, added.text
        yield SharedFile(file_id, knowledge_id, reader, writer)
        client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        client.delete(f"/api/v1/files/{file_id}")


def detach(admin: Actor, shared: SharedFile) -> None:
    """Remove the file from the knowledge base but keep it, as the admin can."""
    with admin.client() as client:
        removed = client.post(
            f"/api/v1/knowledge/{shared.knowledge_id}/file/remove",
            params={"delete_file": "false"},
            json={"file_id": shared.file_id},
        )
        assert removed.status_code == 200, removed.text
        kept = client.get(f"/api/v1/files/{shared.file_id}")
    assert kept.status_code == 200, "the file should survive a removal with delete_file=false"
    # the leftover this bug was about; without it the test below would pin nothing
    assert kept.json()["meta"].get("collection_name") == shared.knowledge_id


@pytest.mark.parametrize("route", READ_ROUTES)
def test_a_detached_file_is_no_longer_readable_through_the_knowledge_base(
    admin, shared_file, route
):
    detach(admin, shared_file)
    with shared_file.reader.client() as client:
        response = client.get(route.format(id=shared_file.file_id))
    assert response.status_code == 404, (
        f"{route} still served a file after it was removed from the only knowledge base the "
        f"reader can open: HTTP {response.status_code} (#29937)"
    )


def test_a_write_grant_no_longer_deletes_a_detached_file(admin, shared_file):
    detach(admin, shared_file)
    with shared_file.writer.client() as client:
        deleted = client.delete(f"/api/v1/files/{shared_file.file_id}")
    assert deleted.status_code == 404, (
        "a write grant on a knowledge base deleted a file that is no longer in it: "
        f"HTTP {deleted.status_code} (#29937)"
    )
    with admin.client() as client:
        assert client.get(f"/api/v1/files/{shared_file.file_id}").status_code == 200


@pytest.mark.parametrize("route", READ_ROUTES)
def test_an_attached_file_is_readable_through_the_knowledge_base(shared_file, route):
    with shared_file.reader.client() as client:
        response = client.get(route.format(id=shared_file.file_id))
    assert response.status_code == 200, response.text


def test_a_write_grant_deletes_an_attached_file_its_knowledge_base_owner_owns(admin, shared_file):
    with shared_file.writer.client() as client:
        deleted = client.delete(f"/api/v1/files/{shared_file.file_id}")
    assert deleted.status_code == 200, deleted.text
    with admin.client() as client:
        assert client.get(f"/api/v1/files/{shared_file.file_id}").status_code == 404
