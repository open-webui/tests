"""Deleting an external knowledge base leaves the connection other knowledge bases still use.

Regression for open-webui/open-webui#28113, fixed in dc03e7e59. `DELETE /knowledge/{id}/delete`
on an external knowledge base rewrote the admin-owned connection list without the connection
the knowledge base pointed at. Connections are shared, so deleting one knowledge base broke every
other one on the same connection, and anyone with a write grant on one of them could do it. The
fix clears the connection only when an admin deletes the last knowledge base referencing it.

Twin of unit/security/test_knowledge_connection_deletion.py.

Discriminates: passes on bbfa876af, fails with the delete route's guard reduced to
`if connection_id:` (the connection disappears after the admin deletes one of two knowledge bases
on it, and after a write-grant user deletes the last one).
"""

from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from harness.actors import Actor

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CONNECTIONS = "/api/v1/knowledge/external/connections"


class ExternalKnowledge:
    """Connections and external knowledge bases the admin creates, removed again afterwards."""

    def __init__(self, client: httpx.Client):
        self.client = client
        self.connection_ids: list[str] = []
        self.knowledge_ids: list[str] = []

    def connection(self) -> str:
        created = self.client.post(
            CONNECTIONS,
            json={"name": "Team vectors", "provider": "qdrant", "endpoint": "http://127.0.0.1:9"},
        )
        assert created.status_code == 200, created.text
        self.connection_ids.append(created.json()["id"])
        return self.connection_ids[-1]

    def knowledge(self, connection_id: str, writer: Actor | None = None) -> str:
        grants = [
            {"principal_type": "user", "principal_id": writer.id, "permission": permission}
            for permission in ("read", "write")
            if writer
        ]
        created = self.client.post(
            "/api/v1/knowledge/external/knowledge/create",
            json={
                "name": "Team docs",
                "connection_id": connection_id,
                "source": {"name": "docs", "config": {"content_field": "text"}},
                "access_grants": grants,
            },
        )
        assert created.status_code == 200, created.text
        self.knowledge_ids.append(created.json()["id"])
        return self.knowledge_ids[-1]

    def listed_connection_ids(self) -> list[str]:
        listed = self.client.get(CONNECTIONS)
        assert listed.status_code == 200, listed.text
        return [connection["id"] for connection in listed.json()["items"]]

    def tear_down(self) -> None:
        for knowledge_id in self.knowledge_ids:
            self.client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        for connection_id in self.connection_ids:
            self.client.delete(f"{CONNECTIONS}/{connection_id}")


@pytest.fixture
def external(admin: Actor) -> Iterator[ExternalKnowledge]:
    with admin.client() as client:
        created = ExternalKnowledge(client)
        yield created
        created.tear_down()


def test_deleting_one_of_two_knowledge_bases_keeps_their_shared_connection(admin, external):
    connection_id = external.connection()
    deleted_id = external.knowledge(connection_id)
    remaining_id = external.knowledge(connection_id)

    with admin.client() as client:
        deleted = client.delete(f"/api/v1/knowledge/{deleted_id}/delete")
        assert deleted.status_code == 200, deleted.text
        assert client.get(f"/api/v1/knowledge/{deleted_id}").status_code == 404
        assert client.get(f"/api/v1/knowledge/{remaining_id}").status_code == 200
    assert connection_id in external.listed_connection_ids(), (
        "deleting one of two external knowledge bases removed the connection the other one "
        "still uses (#28113)"
    )


def test_a_write_grant_deleting_the_last_knowledge_base_keeps_the_connection(external, make_user):
    writer = make_user()
    connection_id = external.connection()
    knowledge_id = external.knowledge(connection_id, writer=writer)

    with writer.client() as client:
        deleted = client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        assert deleted.status_code == 200, deleted.text
        assert client.get(f"/api/v1/knowledge/{knowledge_id}").status_code == 404
    assert connection_id in external.listed_connection_ids(), (
        "a user with a write grant on an external knowledge base removed the admin-owned "
        "connection by deleting it (#28113)"
    )


def test_the_admin_deleting_the_last_knowledge_base_removes_only_its_connection(admin, external):
    connection_id, other_connection_id = external.connection(), external.connection()
    knowledge_id = external.knowledge(connection_id)
    external.knowledge(other_connection_id)

    with admin.client() as client:
        deleted = client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
    assert deleted.status_code == 200, deleted.text
    listed = external.listed_connection_ids()
    assert connection_id not in listed
    assert other_connection_id in listed


def test_a_connection_still_in_use_cannot_be_deleted(admin, external):
    connection_id = external.connection()
    external.knowledge(connection_id)

    with admin.client() as client:
        refused = client.delete(f"{CONNECTIONS}/{connection_id}")
    assert refused.status_code == 400, refused.text
    assert connection_id in external.listed_connection_ids()
