"""Journey: what the knowledge base's file browser reads from the server, and what it may change.

The knowledge base page lists a base's files folder by folder and, when searched, across every
folder at once; each listed file names the folder it sits in, as a path from the base's top level,
so a search result shows where the file lives, and renaming a folder renames that path. Opening a
file asks for its stored original; when that upload is gone from storage the original answers 404
and the download falls back to the file's indexed text. Saving edited indexed text while the
embedding provider fails answers an error that says the text was not indexed, and the same save
goes through once the provider answers again. A reader of a shared base browses its folders and
opens and downloads its files, but every change the browser offers (a new folder, renaming,
moving or deleting a folder, moving, renaming or removing a file, editing its indexed text) is
refused and leaves the base as it was; a writer may make each of them.

Discriminates: passes on dev 550311b21. In a backend copy, a file listing that leaves out each
file's folder path fails the folder path test, a download that answers 404 when the stored upload
is gone fails the missing upload test, an indexed text save that answers success when indexing
fails fails the embedding failure test and a write check on the knowledge base's folder routes
that lets everyone through fails the reader test.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterator

import httpx
import pytest

from harness.access import grant
from harness.actors import Actor
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import json_answer

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

EMBEDDING_SETTINGS = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
REFUSED = (400, 401, 403, 404)


@dataclass
class SharedBase:
    id: str
    harbour: str  # a top-level folder
    berths: str  # a folder inside harbour
    nested_file: str  # in berths
    top_file: str  # at the base's top level
    reader: Actor
    writer: Actor


def create_folder(
    client: httpx.Client, knowledge_id: str, name: str, parent: str | None = None
) -> str:
    created = client.post(
        f"/api/v1/knowledge/{knowledge_id}/dirs/create", json={"name": name, "parent_id": parent}
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def move_file(
    client: httpx.Client, knowledge_id: str, file_id: str, folder: str | None
) -> httpx.Response:
    return client.post(
        f"/api/v1/knowledge/{knowledge_id}/file/move",
        json={"file_id": file_id, "directory_id": folder},
    )


def listing(client: httpx.Client, knowledge_id: str, **params) -> dict:
    listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files", params=params)
    assert listed.status_code == 200, listed.text
    return listed.json()


def search_paths(client: httpx.Client, knowledge_id: str) -> dict[str, tuple[str | None, str]]:
    """Each file the base's search finds, by name, with its folder id and path."""
    found = listing(client, knowledge_id, query=".txt")["items"]
    return {
        entry["filename"]: (entry.get("directory_id"), entry.get("directory_path"))
        for entry in found
    }


def tree(client: httpx.Client, knowledge_id: str) -> dict:
    """Every folder of the base with its parent and name, every file with its folder."""
    folders: dict[str, tuple[str | None, str]] = {}
    files: dict[str, str | None] = {}
    queue: list[str | None] = [None]
    while queue:
        folder = queue.pop()
        page = listing(client, knowledge_id, directory_id=folder or "")
        for entry in page["directories"]:
            folders[entry["id"]] = (entry["parent_id"], entry["name"])
            queue.append(entry["id"])
        for entry in page["items"]:
            files[entry["id"]] = folder
    return {"folders": folders, "files": files}


@pytest.fixture
def shared_base(admin, make_user) -> Iterator[SharedBase]:
    """An admin's base with a nested folder, shared with a reader and a writer."""
    reader, writer = make_user(), make_user()
    grants = [
        grant("user", reader.id, "read"),
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
    ]
    with admin.client() as client:
        name = f"Port {uuid.uuid4().hex[:6]}"
        with knowledge_base(client, name, access_grants=grants) as knowledge_id:
            harbour = create_folder(client, knowledge_id, "Harbour")
            berths = create_folder(client, knowledge_id, "Berths", harbour)
            nested_file = add_text_file(
                client, knowledge_id, "berth-plan.txt", "Berth four takes the ferry."
            )
            assert move_file(client, knowledge_id, nested_file, berths).status_code == 200
            top_file = add_text_file(client, knowledge_id, "tides.txt", "High tide at noon.")
            yield SharedBase(knowledge_id, harbour, berths, nested_file, top_file, reader, writer)


def test_a_search_names_the_folder_each_file_sits_in(shared_base, admin):
    with shared_base.reader.client() as client:
        found = search_paths(client, shared_base.id)
    assert found == {
        "berth-plan.txt": (shared_base.berths, "Harbour/Berths"),
        "tides.txt": (None, ""),
    }, f"a search result does not name the folder its file sits in: {found}"

    with admin.client() as client:
        renamed = client.post(
            f"/api/v1/knowledge/{shared_base.id}/dirs/{shared_base.harbour}/update",
            json={"name": "Port"},
        )
        assert renamed.status_code == 200, renamed.text
        assert search_paths(client, shared_base.id)["berth-plan.txt"] == (
            shared_base.berths,
            "Port/Berths",
        )


def test_a_folder_listing_holds_only_that_folders_files(shared_base):
    with shared_base.reader.client() as client:
        top = listing(client, shared_base.id, directory_id="")
        inside = listing(client, shared_base.id, directory_id=shared_base.berths)
    assert [entry["id"] for entry in top["items"]] == [shared_base.top_file]
    assert [entry["name"] for entry in top["directories"]] == ["Harbour"]
    assert [entry["id"] for entry in inside["items"]] == [shared_base.nested_file]
    assert inside["items"][0]["directory_path"] == "Harbour/Berths"


def stored_upload(instance, file_id: str):
    uploads = list((instance.data_dir / "uploads").glob(f"{file_id}_*"))
    assert len(uploads) == 1, f"expected one stored upload for {file_id}, found {uploads}"
    return uploads[0]


def test_a_file_whose_stored_upload_is_gone_downloads_its_indexed_text(shared_base, instance):
    stored_upload(instance, shared_base.top_file).unlink()

    with shared_base.reader.client() as client:
        original = client.get(f"/api/v1/files/{shared_base.top_file}/content")
        assert original.status_code == 404, original.text

        downloaded = client.get(f"/api/v1/files/{shared_base.top_file}/content/tides.txt")
    assert downloaded.status_code == 200, (
        f"a download whose upload is gone was refused instead of falling back to its text: "
        f"{downloaded.status_code} {downloaded.text}"
    )
    assert downloaded.text == "High tide at noon."


def test_a_file_whose_stored_upload_is_present_downloads_the_upload(shared_base):
    with shared_base.reader.client() as client:
        original = client.get(f"/api/v1/files/{shared_base.top_file}/content")
        downloaded = client.get(f"/api/v1/files/{shared_base.top_file}/content/tides.txt")
    assert original.status_code == 200 and original.text == "High tide at noon."
    assert downloaded.status_code == 200 and downloaded.text == "High tide at noon."


def save_embedding(client: httpx.Client, url: str) -> None:
    current = client.get(EMBEDDING_SETTINGS[0]).json()
    saved = client.post(
        EMBEDDING_SETTINGS[1],
        json={
            "RAG_EMBEDDING_ENGINE": "openai",
            "RAG_EMBEDDING_MODEL": current["RAG_EMBEDDING_MODEL"],
            "openai_config": {**current["openai_config"], "url": url},
        },
    )
    assert saved.status_code == 200, saved.text


@pytest.fixture
def failing_embeddings(admin, preserve, listener):
    """The embedding provider answers 500 until the test calls the returned function."""
    preserve(EMBEDDING_SETTINGS)
    listener.route(
        "POST", "/v1/embeddings", json_answer({"error": {"message": "down"}}, status=500)
    )
    with admin.client() as client:
        working_url = client.get(EMBEDDING_SETTINGS[0]).json()["openai_config"]["url"]
        save_embedding(client, f"{listener.base_url}/v1")
        yield lambda: save_embedding(client, working_url)
        save_embedding(client, working_url)


def indexed_text(client: httpx.Client, file_id: str) -> str:
    read = client.get(f"/api/v1/files/{file_id}/data/content")
    assert read.status_code == 200, read.text
    return read.json()["content"]


def test_saving_indexed_text_while_embedding_fails_reports_the_failure(
    shared_base, failing_embeddings
):
    edit = {"content": "High tide moved to one."}
    with shared_base.writer.client() as client:
        saved = client.post(f"/api/v1/files/{shared_base.top_file}/data/content/update", json=edit)
        assert saved.status_code == 500, (
            "saving indexed text answered success although it could not be indexed: "
            f"{saved.status_code} {saved.text}"
        )
        assert "not fully indexed" in saved.json()["detail"]

        failing_embeddings()
        retried = client.post(
            f"/api/v1/files/{shared_base.top_file}/data/content/update", json=edit
        )
        assert retried.status_code == 200, retried.text
        assert indexed_text(client, shared_base.top_file) == edit["content"]


def reader_changes(base: SharedBase) -> list[tuple[str, str, str, dict | None]]:
    """Every change the browser offers on this base, as method, label, path and body."""
    prefix = f"/api/v1/knowledge/{base.id}"
    return [
        ("POST", "new folder", f"{prefix}/dirs/create", {"name": "Moorings", "parent_id": None}),
        ("POST", "rename folder", f"{prefix}/dirs/{base.harbour}/update", {"name": "Quay"}),
        ("POST", "move folder", f"{prefix}/dirs/{base.berths}/update", {"parent_id": None}),
        (
            "POST",
            "move file",
            f"{prefix}/file/move",
            {"file_id": base.top_file, "directory_id": base.berths},
        ),
        (
            "POST",
            "rename file",
            f"/api/v1/files/{base.top_file}/rename",
            {"filename": "renamed.txt"},
        ),
        (
            "POST",
            "edit text",
            f"/api/v1/files/{base.top_file}/data/content/update",
            {"content": "rewritten"},
        ),
        ("POST", "remove file", f"{prefix}/file/remove", {"file_id": base.nested_file}),
        ("DELETE", "delete folder", f"{prefix}/dirs/{base.harbour}/delete?move_files=false", None),
    ]


def snapshot(client: httpx.Client, base: SharedBase) -> dict:
    names = {
        file_id: client.get(f"/api/v1/files/{file_id}").json()["filename"]
        for file_id in (base.top_file, base.nested_file)
    }
    return {**tree(client, base.id), "names": names, "text": indexed_text(client, base.top_file)}


def test_a_reader_browses_the_folders_but_every_change_is_refused(shared_base, admin):
    with admin.client() as client:
        before = snapshot(client, shared_base)

    with shared_base.reader.client() as client:
        assert tree(client, shared_base.id) == {
            "folders": before["folders"],
            "files": before["files"],
        }
        assert indexed_text(client, shared_base.nested_file) == "Berth four takes the ferry."
        let_through = []
        for method, label, path, body in reader_changes(shared_base):
            answered = client.request(method, path, json=body)
            if answered.status_code not in REFUSED:
                let_through.append(f"{label}: {answered.status_code}")
    assert not let_through, f"a reader of the base was allowed to change it: {let_through}"

    with admin.client() as client:
        assert snapshot(client, shared_base) == before, "a refused change still altered the base"


def test_a_writer_may_make_every_change_the_browser_offers(shared_base, admin):
    with shared_base.writer.client() as client:
        for method, label, path, body in reader_changes(shared_base):
            answered = client.request(method, path, json=body)
            assert answered.status_code == 200, f"{label}: {answered.status_code} {answered.text}"

    with admin.client() as client:
        after = tree(client, shared_base.id)
    # Berths moved to the top level before Harbour was deleted, so it survives with the moved file
    assert sorted(after["folders"].values()) == [(None, "Berths"), (None, "Moorings")]
    assert after["files"] == {shared_base.top_file: shared_base.berths}


def test_a_stranger_cannot_list_the_base(shared_base, make_user):
    with make_user().client() as client:
        listed = client.get(f"/api/v1/knowledge/{shared_base.id}/files", params={"query": ".txt"})
    assert listed.status_code in REFUSED, listed.text
