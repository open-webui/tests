"""Journey: keeping a knowledge base in step with its files, and getting its content out again.

Editing a file's text in the knowledge base replaces what a search of the base finds, and
re-syncing a file into the base after its own text changed does the same, for its owner or a
writer only. Uploading a folder first asks for a diff of the local manifest against the base:
new files, changed ones (by checksum) with the stale copy to replace, files and folders to remove
and folders to create. The admin exports a base as a zip of its files' text and re-embeds every
base's name and description for the knowledge search; an external source is added and edited in
one step, each time only after a test query against it finds something.

Discriminates: in a backend copy, dropping the removal of the file's old vectors from the
knowledge file update turns the re-sync test red (the search finds the old text too), comparing
the manifest by filename alone turns the diff test red, and skipping the test query of an
external source create turns the empty-source test red (the source is saved).
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile

import httpx
import pytest

from harness.access import make_group
from harness.external_knowledge import (
    CONNECTIONS,
    QDRANT_API_KEY,
    SOURCE_CONFIG,
    point,
    serve_qdrant,
)
from harness.knowledge_bases import add_text_file, knowledge_base

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def author(admin, make_user):
    """An account allowed to create knowledge bases."""
    account = make_user()
    make_group(admin, [account], {"workspace": {"knowledge": True}})
    return account


def found_text(client: httpx.Client, knowledge_id: str) -> str:
    """Everything a search of the base returns; the scripted embedding ranks every chunk alike."""
    queried = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": knowledge_id, "query": "ferry", "k": 20},
    )
    assert queried.status_code == 200, queried.text
    return " | ".join(queried.json()["documents"][0])


def upload_into(
    client: httpx.Client,
    knowledge_id: str,
    filename: str,
    text: str,
    directory_id: str | None = None,
) -> str:
    """Upload a file straight into the base, the way the folder upload does."""
    metadata = {"knowledge_id": knowledge_id, "directory_id": directory_id}
    uploaded = client.post(
        "/api/v1/files/",
        params={"process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
        data={"metadata": json.dumps(metadata)},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def create_directory(client: httpx.Client, knowledge_id: str, name: str) -> str:
    created = client.post(f"/api/v1/knowledge/{knowledge_id}/dirs/create", json={"name": name})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- replacing a file


def test_editing_a_files_text_replaces_what_the_base_finds(author):
    with author.client() as client, knowledge_base(client, "Timetable") as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "ferry.txt", "The ferry leaves at nine.")

        edited = client.post(
            f"/api/v1/files/{file_id}/data/content/update",
            json={"content": "The ferry leaves at ten."},
        )

        assert edited.status_code == 200, edited.text
        found = found_text(client, knowledge_id)
    assert "at ten" in found
    assert "at nine" not in found, "the base still finds the replaced text"


def reprocess(client: httpx.Client, file_id: str, text: str) -> None:
    """Replace the file's own text without touching the bases that hold it."""
    processed = client.post(
        "/api/v1/retrieval/process/file", json={"file_id": file_id, "content": text}
    )
    assert processed.status_code == 200, processed.text


def test_re_syncing_a_file_puts_its_new_text_in_the_base(author):
    with author.client() as client, knowledge_base(client, "Timetable") as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "ferry.txt", "The ferry leaves at nine.")
        reprocess(client, file_id, "The ferry leaves at eleven.")

        synced = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/update", json={"file_id": file_id}
        )

        assert synced.status_code == 200, synced.text
        assert [entry["id"] for entry in synced.json()["files"]] == [file_id]
        found = found_text(client, knowledge_id)
    assert "at eleven" in found
    assert "at nine" not in found, "the re-sync left the old text in the base"


def test_a_stranger_and_a_file_outside_the_base_are_refused_the_re_sync(author, make_user):
    owner, stranger = author, make_user()
    with owner.client() as client, knowledge_base(client, "Timetable") as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "ferry.txt", "The ferry leaves at nine.")
        loose = client.post(
            "/api/v1/files/",
            params={"process_in_background": "false"},
            files={"file": ("loose.txt", b"Not in the base.", "text/plain")},
        ).json()["id"]
        reprocess(client, file_id, "The ferry leaves at noon.")
        with stranger.client() as stranger_client:
            by_stranger = stranger_client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/update", json={"file_id": file_id}
            )
        outside = client.post(
            f"/api/v1/knowledge/{knowledge_id}/file/update", json={"file_id": loose}
        )
        found = found_text(client, knowledge_id)

    assert by_stranger.status_code == 400, by_stranger.text
    assert outside.status_code == 400, outside.text
    assert "at nine" in found and "at noon" not in found and "Not in the base" not in found


# --------------------------------------------------------------------------- folder sync diff


def test_the_sync_diff_sorts_a_manifest_against_the_base(author, make_user):
    owner, stranger = author, make_user()
    with owner.client() as client, knowledge_base(client, "Handbook") as knowledge_id:
        docs = create_directory(client, knowledge_id, "docs")
        old = create_directory(client, knowledge_id, "old")
        upload_into(client, knowledge_id, "readme.txt", "Read me first.")
        changed_id = upload_into(client, knowledge_id, "guide.txt", "Guide, first edition.", docs)
        dropped_id = upload_into(client, knowledge_id, "retired.txt", "No longer shipped.")
        manifest = [
            {
                "filename": "readme.txt",
                "path": "",
                "checksum": checksum("Read me first."),
                "size": 14,
            },
            {
                "filename": "guide.txt",
                "path": "docs",
                "checksum": checksum("second edition"),
                "size": 14,
            },
            {"filename": "api.txt", "path": "docs/api", "checksum": checksum("new"), "size": 3},
        ]

        diff = client.post(
            f"/api/v1/knowledge/{knowledge_id}/sync/diff", json={"manifest": manifest}
        )
        with stranger.client() as stranger_client:
            refused = stranger_client.post(
                f"/api/v1/knowledge/{knowledge_id}/sync/diff", json={"manifest": manifest}
            )

    assert diff.status_code == 200, diff.text
    answer = diff.json()
    assert answer["added"] == [{"filename": "api.txt", "path": "docs/api"}]
    assert answer["modified"] == [
        {"filename": "guide.txt", "path": "docs", "stale_file_id": changed_id}
    ]
    assert answer["deleted"] == [{"file_id": dropped_id, "filename": "retired.txt"}]
    assert answer["unmodified_count"] == 1
    assert answer["mkdir"] == ["docs/api"]
    assert answer["rmdir"] == [old]
    assert answer["directory_map"] == {"docs": docs, "old": old}
    assert refused.status_code == 403


# --------------------------------------------------------------------------- export


def test_the_admin_exports_a_base_as_a_zip_of_its_text(admin, author):
    owner = author
    with owner.client() as client, knowledge_base(client, "Harbour/Notes") as knowledge_id:
        add_text_file(client, knowledge_id, "gates.txt", "Gate code 4471.")
        add_text_file(client, knowledge_id, "visitors.md", "Sign in at the office.")
        by_owner = client.get(f"/api/v1/knowledge/{knowledge_id}/export")
        with admin.client() as admin_client:
            exported = admin_client.get(f"/api/v1/knowledge/{knowledge_id}/export")

    assert by_owner.status_code == 401
    assert exported.status_code == 200, exported.text
    assert "Harbour_Notes.zip" in exported.headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(exported.content))
    contents = {name: archive.read(name).decode() for name in archive.namelist()}
    assert contents == {"gates.txt": "Gate code 4471.", "visitors.md.txt": "Sign in at the office."}


# --------------------------------------------------------------------------- metadata reindex


def test_the_admin_re_embeds_every_bases_name_and_description(admin, author, upstream):
    name = f"Tide tables {uuid.uuid4().hex[:6]}"
    with author.client() as client:
        created = client.post(
            "/api/v1/knowledge/create", json={"name": name, "description": "When the water turns."}
        )
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        refused = client.post("/api/v1/knowledge/metadata/reindex")
        upstream.reset()
        try:
            with admin.client() as admin_client:
                reindexed = admin_client.post("/api/v1/knowledge/metadata/reindex")
        finally:
            client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")

    assert refused.status_code == 401
    assert reindexed.status_code == 200, reindexed.text
    counts = reindexed.json()
    assert counts["total"] >= 1 and counts["success"] == counts["total"]
    embedded = json.dumps([entry.body["input"] for entry in upstream.requests_to("/embeddings")])
    assert f"{name}\\n\\nWhen the water turns." in embedded
    assert len(upstream.requests_to("/embeddings")) >= counts["total"]


# --------------------------------------------------------------------------- external sources

COLLECTION = "pilot_notes"
SOURCE = {"name": COLLECTION, "config": SOURCE_CONFIG}


@pytest.fixture
def cleanup(admin):
    """Deletes the knowledge bases, then the connections, a test created."""
    created: dict[str, list[str]] = {"knowledge": [], "connections": []}
    yield created
    with admin.client() as client:
        for knowledge_id in created["knowledge"]:
            client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")
        for connection_id in created["connections"]:
            client.delete(f"{CONNECTIONS}/{connection_id}")


def connection_ids(client: httpx.Client) -> set[str]:
    return {connection["id"] for connection in client.get(CONNECTIONS).json()["items"]}


def test_an_external_source_is_added_and_edited_in_one_step(admin, listener, cleanup):
    form = serve_qdrant(listener, COLLECTION, [point(1, "Pilots board at buoy 3.", 0.9)])
    with admin.client() as client:
        before = connection_ids(client)
        created = client.post(
            "/api/v1/knowledge/external/source/create",
            json={
                "name": " Pilot notes ",
                "connection": form,
                "source": SOURCE,
                "test_query": "board",
            },
        )
        assert created.status_code == 200, created.text
        knowledge = created.json()
        cleanup["knowledge"].append(knowledge["id"])
        [connection_id] = connection_ids(client) - before
        cleanup["connections"].append(connection_id)

        without_key = {key: value for key, value in form.items() if key != "auth_config"}
        edited = client.patch(
            f"/api/v1/knowledge/external/source/{knowledge['id']}",
            json={
                "name": "Harbour pilots",
                "description": "Boarding points",
                "connection": {**without_key, "name": "Renamed Qdrant"},
                "source": SOURCE,
                "test_query": "board",
            },
        )
        stored = client.get(f"/api/v1/knowledge/{knowledge['id']}").json()
        connection = client.get(f"{CONNECTIONS}/{connection_id}").json()

    assert knowledge["name"] == "Pilot notes"
    assert knowledge["meta"]["source"] == "external" and knowledge["meta"]["read_only"] is True
    assert knowledge["meta"]["external"]["connection_id"] == connection_id
    assert edited.status_code == 200, edited.text
    assert (stored["name"], stored["description"]) == ("Harbour pilots", "Boarding points")
    assert connection["name"] == "Renamed Qdrant"
    queries = listener.requests_to(f"/collections/{COLLECTION}/points/query")
    assert len(queries) == 2, "each save ran one test query"
    assert queries[-1].headers.get("api-key") == QDRANT_API_KEY, "the edit without a key dropped it"


def test_an_external_source_whose_test_finds_nothing_is_not_saved(admin, listener):
    form = serve_qdrant(listener, COLLECTION, [])
    with admin.client() as client:
        before_connections = connection_ids(client)
        before_knowledge = {
            entry["id"] for entry in client.get("/api/v1/knowledge/").json()["items"]
        }
        refused = client.post(
            "/api/v1/knowledge/external/source/create",
            json={"name": "Empty", "connection": form, "source": SOURCE, "test_query": "board"},
        )
        after_knowledge = {
            entry["id"] for entry in client.get("/api/v1/knowledge/").json()["items"]
        }

        assert refused.status_code == 400, refused.text
        assert connection_ids(client) == before_connections
    assert after_knowledge == before_knowledge


def test_an_ordinary_base_is_not_edited_as_an_external_source(admin, listener):
    form = serve_qdrant(listener, COLLECTION, [point(1, "Pilots board at buoy 3.", 0.9)])
    with admin.client() as client, knowledge_base(client, "Local") as knowledge_id:
        refused = client.patch(
            f"/api/v1/knowledge/external/source/{knowledge_id}",
            json={
                "name": "Taken over",
                "connection": form,
                "source": SOURCE,
                "test_query": "board",
            },
        )
        stored = client.get(f"/api/v1/knowledge/{knowledge_id}").json()

    assert refused.status_code == 404
    assert stored["name"] == "Local"
    assert listener.requests_to(f"/collections/{COLLECTION}/points/query") == []
