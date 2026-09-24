"""Sync cleanup on one knowledge base only removes what belongs to that knowledge base.

Regression for open-webui/open-webui#26722, fixed in 707efeaed. `POST /knowledge/{id}/sync/cleanup`
checked write access to the knowledge base in the URL and then deleted every caller-supplied
directory id and file id without checking it belonged there. A write grant on one knowledge base
was enough to delete another one's folders together with the files filed in them. The fix skips
directories of other knowledge bases and files the knowledge base does not contain.

Twin of unit/security/test_knowledge_cleanup_scope.py.

Discriminates: passes on bbfa876af, fails with both guards of `sync_knowledge_cleanup` removed
(the other knowledge base loses its folder and the file filed in it). Removing only the file guard
stays green: a later check in the same loop keeps any file another knowledge base still holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import httpx
import pytest

from harness.actors import Actor
from harness.mock_embeddings import embed_through

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FOREIGN_TEXT = "Board minutes: the merger closes in March."


@dataclass
class TwoKnowledgeBases:
    user: Actor
    own_knowledge: str  # the user holds a write grant on it
    other_knowledge: str  # the admin's, out of the user's reach
    other_directory: str
    other_file: str  # filed in other_directory


def upload_text(client: httpx.Client, name: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def create_knowledge(client: httpx.Client, name: str, writer: Actor | None = None) -> str:
    grants = [
        {"principal_type": "user", "principal_id": writer.id, "permission": permission}
        for permission in ("read", "write")
        if writer
    ]
    created = client.post(
        "/api/v1/knowledge/create", json={"name": name, "description": "", "access_grants": grants}
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def create_directory(client: httpx.Client, knowledge_id: str, name: str) -> str:
    created = client.post(f"/api/v1/knowledge/{knowledge_id}/dirs/create", json={"name": name})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def add_file(
    client: httpx.Client, knowledge_id: str, file_id: str, directory_id: str | None = None
) -> None:
    added = client.post(
        f"/api/v1/knowledge/{knowledge_id}/file/add",
        json={"file_id": file_id, "directory_id": directory_id},
    )
    assert added.status_code == 200, added.text


def listing(client: httpx.Client, knowledge_id: str, directory_id: str | None = None) -> dict:
    params = {"directory_id": directory_id} if directory_id is not None else {}
    listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files", params=params)
    assert listed.status_code == 200, listed.text
    return listed.json()


def ids(entries: list[dict]) -> list[str]:
    return [entry["id"] for entry in entries]


@pytest.fixture
def mock_embeddings(upstream, admin: Actor, preserve) -> None:
    embed_through(upstream, admin, preserve)


@pytest.fixture
def knowledge_bases(admin: Actor, make_user, mock_embeddings) -> Iterator[TwoKnowledgeBases]:
    user = make_user()
    with admin.client() as client:
        own_knowledge = create_knowledge(client, "Team notes", writer=user)
        other_knowledge = create_knowledge(client, "Board")
        other_directory = create_directory(client, other_knowledge, "Minutes")
        other_file = upload_text(client, "minutes.txt", FOREIGN_TEXT)
        add_file(client, other_knowledge, other_file, other_directory)
        yield TwoKnowledgeBases(user, own_knowledge, other_knowledge, other_directory, other_file)
        for knowledge_id in (own_knowledge, other_knowledge):
            client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")


def assert_other_knowledge_untouched(admin: Actor, scenario: TwoKnowledgeBases) -> None:
    with admin.client() as client:
        root = listing(client, scenario.other_knowledge)
        filed = listing(client, scenario.other_knowledge, scenario.other_directory)
        content = client.get(f"/api/v1/files/{scenario.other_file}/data/content")
    assert scenario.other_directory in ids(root["directories"]), (
        "the other knowledge base lost its folder (#26722)"
    )
    assert scenario.other_file in ids(filed["items"]), (
        "the other knowledge base lost the file filed in its folder (#26722)"
    )
    assert content.status_code == 200 and content.json()["content"] == FOREIGN_TEXT


def test_cleanup_removes_its_own_entries_and_leaves_another_knowledge_base_alone(
    admin, knowledge_bases
):
    scenario = knowledge_bases
    with scenario.user.client() as client:
        own_file = upload_text(client, "draft.txt", "An outdated draft.")
        add_file(client, scenario.own_knowledge, own_file)
        own_directory = create_directory(client, scenario.own_knowledge, "Drafts")

        cleaned = client.post(
            f"/api/v1/knowledge/{scenario.own_knowledge}/sync/cleanup",
            json={
                "file_ids": [own_file, scenario.other_file],
                "dir_ids": [own_directory, scenario.other_directory],
            },
        )
        assert cleaned.status_code == 200, cleaned.text
        remaining = listing(client, scenario.own_knowledge)
        own_file_lookup = client.get(f"/api/v1/files/{own_file}")

    assert own_file not in ids(remaining["items"])
    assert own_directory not in ids(remaining["directories"])
    assert own_file_lookup.status_code == 404, "the user's own stale file should be deleted"
    assert_other_knowledge_untouched(admin, scenario)


def foreign_id_requests(scenario: TwoKnowledgeBases) -> dict[str, tuple]:
    """Per-item routes of the user's knowledge base, each naming the other one's folder or file."""
    base = f"/api/v1/knowledge/{scenario.own_knowledge}"
    folder, file_id = scenario.other_directory, scenario.other_file
    return {
        "folder delete": ("DELETE", f"{base}/dirs/{folder}/delete", None, 404),
        "folder rename": ("POST", f"{base}/dirs/{folder}/update", {"name": "Renamed"}, 404),
        "file remove": ("POST", f"{base}/file/remove", {"file_id": file_id}, 400),
        "file move": ("POST", f"{base}/file/move", {"file_id": file_id, "directory_id": None}, 404),
    }


@pytest.mark.parametrize(
    "request_kind", ["folder delete", "folder rename", "file remove", "file move"]
)
def test_per_item_routes_refuse_another_knowledge_bases_ids(admin, knowledge_bases, request_kind):
    method, path, body, expected_status = foreign_id_requests(knowledge_bases)[request_kind]
    with knowledge_bases.user.client() as client:
        response = client.request(method, path, json=body)
    assert response.status_code == expected_status, response.text
    assert_other_knowledge_untouched(admin, knowledge_bases)


def test_cleanup_of_a_knowledge_base_without_write_access_is_refused(admin, knowledge_bases):
    scenario = knowledge_bases
    with scenario.user.client() as client:
        refused = client.post(
            f"/api/v1/knowledge/{scenario.other_knowledge}/sync/cleanup",
            json={"file_ids": [scenario.other_file], "dir_ids": [scenario.other_directory]},
        )
    assert refused.status_code == 403, refused.text
    assert_other_knowledge_untouched(admin, scenario)
