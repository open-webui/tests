"""Directory ids sent to one knowledge base cannot reach into another one's folder tree.

Regression for open-webui/open-webui#29887, fixed in a9541c18c. The knowledge routes that take a
`directory_id` or `parent_id` used it without checking it belonged to the knowledge base in the
URL, so a write grant on one knowledge base let a user file documents into, create folders under
and move folders below another knowledge base's folders. An upload carrying
`{knowledge_id, directory_id}` metadata was filed into the foreign folder too, and the file
listing's breadcrumbs walked the foreign folder chain. The fix verifies every directory id against
the knowledge base, drops a foreign one on upload and scopes the breadcrumb walk.

Twin of unit/security/test_knowledge_directory_scope.py.

Discriminates: passes on bbfa876af, fails with the fix's new directory checks removed (add, batch
add, folder create, folder re-parent and upload auto-link accept the foreign folder, and the
breadcrumbs name it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx
import pytest

from harness.actors import Actor
from harness.mock_embeddings import embed_through

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@dataclass
class TwoKnowledgeBases:
    user: Actor
    own_knowledge: str  # the user holds a write grant on it
    own_directory: str
    attached_file: str  # the user's, at the root of own_knowledge
    loose_file: str  # the user's, in no knowledge base
    other_knowledge: str  # the admin's, out of the user's reach
    other_directory: str


def upload_text(client: httpx.Client, name: str, text: str, metadata: dict | None = None) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
        data={"metadata": json.dumps(metadata)} if metadata else None,
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
    with admin.client() as admin_client, user.client() as user_client:
        own_knowledge = create_knowledge(admin_client, "Team notes", writer=user)
        other_knowledge = create_knowledge(admin_client, "Board")
        other_directory = create_directory(admin_client, other_knowledge, "Minutes")
        own_directory = create_directory(user_client, own_knowledge, "Drafts")
        attached_file = upload_text(user_client, "agenda.txt", "Agenda for Monday.")
        added = user_client.post(
            f"/api/v1/knowledge/{own_knowledge}/file/add", json={"file_id": attached_file}
        )
        assert added.status_code == 200, added.text
        loose_file = upload_text(user_client, "memo.txt", "A memo on the budget.")
        yield TwoKnowledgeBases(
            user,
            own_knowledge,
            own_directory,
            attached_file,
            loose_file,
            other_knowledge,
            other_directory,
        )
        for knowledge_id in (own_knowledge, other_knowledge):
            admin_client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")


def foreign_directory_requests(scenario: TwoKnowledgeBases) -> dict[str, tuple]:
    base = f"/api/v1/knowledge/{scenario.own_knowledge}"
    foreign = scenario.other_directory
    return {
        "file add": (
            "POST",
            f"{base}/file/add",
            {"file_id": scenario.loose_file, "directory_id": foreign},
        ),
        "batch add": (
            "POST",
            f"{base}/files/batch/add",
            [{"file_id": scenario.loose_file, "directory_id": foreign}],
        ),
        "folder create": ("POST", f"{base}/dirs/create", {"name": "Inside", "parent_id": foreign}),
        "folder re-parent": (
            "POST",
            f"{base}/dirs/{scenario.own_directory}/update",
            {"parent_id": foreign},
        ),
        "file move": (
            "POST",
            f"{base}/file/move",
            {"file_id": scenario.attached_file, "directory_id": foreign},
        ),
        "folder delete": ("DELETE", f"{base}/dirs/{foreign}/delete", None),
    }


@pytest.mark.parametrize(
    "request_kind",
    ["file add", "batch add", "folder create", "folder re-parent", "file move", "folder delete"],
)
def test_a_foreign_folder_is_refused_and_nothing_changes(admin, knowledge_bases, request_kind):
    scenario = knowledge_bases
    method, path, body = foreign_directory_requests(scenario)[request_kind]
    with scenario.user.client() as client:
        response = client.request(method, path, json=body)
        whole_knowledge = listing(client, scenario.own_knowledge)
        inside_foreign = listing(client, scenario.own_knowledge, scenario.other_directory)
    with admin.client() as client:
        other_root = listing(client, scenario.other_knowledge)

    assert response.status_code == 404, (
        f"{request_kind} accepted another knowledge base's folder: HTTP {response.status_code} "
        "(#29887)"
    )
    assert scenario.loose_file not in ids(whole_knowledge["items"])
    assert scenario.own_directory in ids(whole_knowledge["directories"])
    assert inside_foreign["items"] == [] and inside_foreign["directories"] == []
    assert scenario.other_directory in ids(other_root["directories"])


def test_an_upload_naming_a_foreign_folder_lands_at_the_root(knowledge_bases):
    scenario = knowledge_bases
    metadata = {"knowledge_id": scenario.own_knowledge, "directory_id": scenario.other_directory}
    with scenario.user.client() as client:
        uploaded = upload_text(client, "notes.txt", "Uploaded into the team notes.", metadata)
        root = listing(client, scenario.own_knowledge, directory_id="")
    assert uploaded in ids(root["items"]), (
        "an upload auto-linked into another knowledge base's folder instead of this one's "
        "root (#29887)"
    )


def test_the_breadcrumbs_do_not_walk_a_foreign_folder(knowledge_bases):
    scenario = knowledge_bases
    with scenario.user.client() as client:
        listed = listing(client, scenario.own_knowledge, scenario.other_directory)
    assert listed["breadcrumbs"] == [], (
        f"the breadcrumbs named another knowledge base's folder: {listed['breadcrumbs']} (#29887)"
    )


def test_the_knowledge_bases_own_folder_is_accepted_everywhere(knowledge_bases):
    scenario = knowledge_bases
    own_directory = scenario.own_directory
    metadata = {"knowledge_id": scenario.own_knowledge, "directory_id": own_directory}
    with scenario.user.client() as client:
        added = client.post(
            f"/api/v1/knowledge/{scenario.own_knowledge}/file/add",
            json={"file_id": scenario.loose_file, "directory_id": own_directory},
        )
        assert added.status_code == 200, added.text
        uploaded = upload_text(client, "notes.txt", "Uploaded into the drafts.", metadata)
        listed = listing(client, scenario.own_knowledge, own_directory)
        deleted = client.delete(
            f"/api/v1/knowledge/{scenario.own_knowledge}/dirs/{own_directory}/delete"
        )
        assert deleted.status_code == 200, deleted.text
        root = listing(client, scenario.own_knowledge, directory_id="")

    assert set(ids(listed["items"])) == {scenario.loose_file, uploaded}
    assert ids(listed["breadcrumbs"]) == [own_directory]
    assert own_directory not in ids(root["directories"])
    assert {scenario.loose_file, uploaded} <= set(ids(root["items"])), "files move up on delete"


def test_an_unknown_folder_is_refused(knowledge_bases):
    scenario = knowledge_bases
    with scenario.user.client() as client:
        response = client.post(
            f"/api/v1/knowledge/{scenario.own_knowledge}/file/add",
            json={"file_id": scenario.loose_file, "directory_id": "no-such-folder"},
        )
    assert response.status_code == 404, response.text
