"""Journey: querying several collections at once, and the admin's vector store maintenance.

`/query/collection` searches several collections and merges the hits into one list of at most
`k`; with hybrid search and enriched texts, BM25 also reads each chunk's file name, so a file
named for a word outranks a file that only mentions it once. The admin removes one file's
entries from a collection, empties the upload folder and resets the vector store, which also
drops every knowledge base.
Each maintenance route is refused to users. The resets run on an instance of this module's own.

Discriminates: in a backend copy, `merge_and_sort_query_results` ignoring `k` turned the merge
test red; `get_enriched_texts` returning the bare chunk text turned the file name test red; the
delete route filtering on another hash turned the delete test red; and the vector store reset
leaving knowledge bases in place turned the reset test red.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def own(instance_with):
    return instance_with({"RAG_TOP_K": "5"})


@pytest.fixture
def admin_client(own):
    with admin_of(own).client() as client:
        yield client


def upload(client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def query_collections(client, names: list[str], query: str, **options) -> dict:
    answered = client.post(
        "/api/v1/retrieval/query/collection",
        json={"collection_names": names, "query": query, **options},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()


def file_ids_of(result: dict) -> set[str]:
    return {entry["file_id"] for entry in result["metadatas"][0]}


def test_a_query_over_several_collections_merges_them(own):
    reader = create_user(own)
    with reader.client() as client:
        herons = upload(client, "herons.txt", "Herons wait.\n\nHerons stalk.")
        grebes = upload(client, "grebes.txt", "Grebes dive.")
        names = [f"file-{herons}", f"file-{grebes}"]

        merged = query_collections(client, names, "birds", k=10, hybrid=False)
        capped = query_collections(client, names, "birds", k=1, hybrid=False)

    assert file_ids_of(merged) == {herons, grebes}, merged
    assert len(capped["documents"][0]) == 1, f"k=1 returned more than one chunk: {capped}"


def test_enriched_texts_let_a_file_be_found_by_its_name(own, admin_client, preserve):
    preserve(RETRIEVAL_CONFIG, on=own)
    switched = admin_client.post(RETRIEVAL_CONFIG[1], json={"ENABLE_RAG_HYBRID_SEARCH": True})
    assert switched.status_code == 200, switched.text
    with knowledge_base(admin_client, "Mixed") as knowledge_id:
        add_text_file(admin_client, knowledge_id, "birds.txt", "A ledger of herons by the reeds.")
        for index, bird in enumerate(("Grebes dive.", "Coots squabble.", "Otters slide.")):
            add_text_file(
                admin_client, knowledge_id, f"filler-{index}.txt", bird
            )  # BM25 needs a corpus
        ledger = add_text_file(admin_client, knowledge_id, "ledger-accounts.txt", "Paid 40.")

        found = query_collections(
            admin_client,
            [knowledge_id],
            "ledger",
            k=1,
            k_reranker=1,
            hybrid=True,
            hybrid_bm25_weight=1,
            enable_enriched_texts=True,
        )

    assert file_ids_of(found) == {ledger}, f"the file name did not outweigh the text: {found}"


def test_an_admin_deletes_one_files_entries_from_a_collection(own, admin_client):
    user = create_user(own)
    with knowledge_base(admin_client, "Two files") as knowledge_id:
        kept = add_text_file(admin_client, knowledge_id, "kept.txt", "Herons stay.")
        dropped = add_text_file(admin_client, knowledge_id, "dropped.txt", "Grebes leave.")
        form = {"collection_name": knowledge_id, "file_id": dropped}

        with user.client() as client:
            refused = client.post("/api/v1/retrieval/delete", json=form)
        deleted = admin_client.post("/api/v1/retrieval/delete", json=form)
        unknown = admin_client.post(
            "/api/v1/retrieval/delete", json={"collection_name": knowledge_id, "file_id": "nope"}
        )
        missing = admin_client.post(
            "/api/v1/retrieval/delete", json={"collection_name": "no-such", "file_id": kept}
        )
        left = query_collections(admin_client, [knowledge_id], "birds", k=10, hybrid=False)

    assert refused.status_code in (401, 403), refused.text
    assert deleted.json() == {"status": True}, deleted.text
    assert file_ids_of(left) == {kept}, f"the dropped file's chunks are still there: {left}"
    assert unknown.status_code == 404
    assert missing.json() == {"status": False}


def test_resetting_the_uploads_removes_the_stored_files(own, admin_client):
    user = create_user(own)
    with user.client() as client:
        file_id = upload(client, "keep.txt", "Herons wait.")
        assert client.get(f"/api/v1/files/{file_id}/content").status_code == 200
        refused = client.post("/api/v1/retrieval/reset/uploads")

    reset = admin_client.post("/api/v1/retrieval/reset/uploads")

    assert refused.status_code in (401, 403), refused.text
    assert reset.status_code == 200 and reset.json() is True, reset.text
    assert not [path for path in (own.data_dir / "uploads").iterdir()], "uploads remain"
    with user.client() as client:
        assert client.get(f"/api/v1/files/{file_id}/content").status_code == 404


def test_resetting_the_vector_store_drops_every_knowledge_base(own, admin_client):
    user = create_user(own)
    created = admin_client.post(
        "/api/v1/knowledge/create", json={"name": "Doomed", "description": ""}
    )
    assert created.status_code == 200, created.text
    with user.client() as client:
        refused = client.post("/api/v1/retrieval/reset/db")

    reset = admin_client.post("/api/v1/retrieval/reset/db")

    assert refused.status_code in (401, 403), refused.text
    assert reset.status_code == 200, reset.text
    listed = admin_client.get("/api/v1/knowledge/")
    assert listed.status_code == 200
    items = listed.json()
    items = items.get("items", items) if isinstance(items, dict) else items
    assert items == [], f"knowledge bases survived the reset: {items}"
