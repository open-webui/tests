"""Regression: a file is stored in Milvus through `MilvusClient`, even where a scalar index fails.

- open-webui PR #26911 (`f4a6ea930`): the Milvus stores moved from pymilvus's ORM API
  (`Collection`, `connections`, `utility`) to `MilvusClient`. Uploading a file and reading it
  back through retrieval covers the new calls end to end, in both store layouts.
- PR #27521 (`a15e44a5f`, issue #26978): the multitenant store's shared collections carry a
  `resource_id` scalar index. It was created without a type, which Milvus Lite and older servers
  refuse, and the refusal escaped collection creation, so no file could be stored at all. The
  store now retries with an explicit INVERTED index and never fails creation over it.

Milvus is played by `harness.milvus_server`, which can refuse an untyped scalar index the way
those servers do.

Twin of the Milvus part of unit/retrieval/test_document_ingestion.py, which keeps the audit that
neither store imports the ORM API again.

Discriminates: passes on dev ef67cc3fa; without the INVERTED retry (a15e44a5f reverted) the
refused-index case fails (the file is marked failed and nothing is stored); the other cases pass
on both.

Fix ac8bc4552 (PR open-webui/open-webui#33745, issue #33744): adding a file to a knowledge base
looks up the file's stored chunks, and on Milvus that lookup failed with no limit given, so the
file was processed again and its chunks stored a second time. The knowledge base test passes on
dev e8c0ad85e and fails on a backend copy with ac8bc4552 reverted (the file's own collection holds
its chunk twice).
"""

from __future__ import annotations

import uuid

import pytest

from harness.actors import admin_of
from harness.knowledge_bases import add_text_file, knowledge_base

pytest.importorskip("pymilvus", reason="pymilvus not installed in this env")

from harness.milvus_server import milvus_env, serving_milvus  # noqa: E402

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RESOURCE_ID = "resource_id"


@pytest.fixture(scope="module")
def milvus():
    with serving_milvus() as fake:
        yield fake


@pytest.fixture
def fresh_milvus(milvus):
    """The fake with no collections, accepting every index unless a test says otherwise."""
    milvus.collections.clear()
    milvus.index_requests.clear()
    milvus.refuse_untyped_scalar_index = False
    return milvus


@pytest.fixture(scope="module")
def multitenant(instance_with, milvus):
    return admin_of(instance_with(milvus_env(milvus, multitenancy=True)))


@pytest.fixture(scope="module")
def per_collection(instance_with, milvus):
    return admin_of(instance_with(milvus_env(milvus, multitenancy=False)))


def store_and_read_back(owner, text: str) -> list[str]:
    """Upload `text` as a file, then read its own collection back through retrieval."""
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/?process_in_background=false",
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        processed = client.get(f"/api/v1/files/{file_id}/process/status").json()
        assert processed == {"status": "completed"}, f"the file was not stored: {processed}"
        queried = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": f"file-{file_id}", "query": text},
        )
    assert queried.status_code == 200, queried.text
    return queried.json()["documents"][0]


def scalar_index_types(milvus) -> list[str]:
    return [index_type for _, field, index_type in milvus.index_requests if field == RESOURCE_ID]


def test_a_file_is_stored_in_a_shared_collection(multitenant, fresh_milvus):
    assert store_and_read_back(multitenant, "herons nest in colonies") == [
        "herons nest in colonies"
    ]
    assert "open_webui_files" in fresh_milvus.collections
    assert scalar_index_types(fresh_milvus) == [""], "the server no longer picks the index type"


def test_a_refused_scalar_index_falls_back_to_inverted(multitenant, fresh_milvus):
    fresh_milvus.refuse_untyped_scalar_index = True

    stored = store_and_read_back(multitenant, "ospreys dive for fish")

    assert stored == ["ospreys dive for fish"], "a refused scalar index stopped the upload (#26978)"
    assert scalar_index_types(fresh_milvus) == ["", "INVERTED"]


def test_a_file_is_stored_in_a_collection_of_its_own(per_collection, fresh_milvus):
    assert store_and_read_back(per_collection, "kingfishers perch low") == ["kingfishers perch low"]
    assert any(name.startswith("open_webui_file_") for name in fresh_milvus.collections)


def rows_of(milvus, name: str) -> list[dict]:
    collection = milvus.collections.get(name)
    return list(collection.rows.values()) if collection else []


def test_a_file_added_to_a_knowledge_base_is_stored_once(per_collection, fresh_milvus):
    # the add re-processed the file and stored its chunks a second time (issue #33744)
    with per_collection.client() as client, knowledge_base(client, "Birds") as knowledge_id:
        file_id = add_text_file(client, knowledge_id, "waders.txt", "avocets sweep the mud")
        file_rows = rows_of(fresh_milvus, f"open_webui_file_{file_id.replace('-', '_')}")
        knowledge_rows = rows_of(fresh_milvus, f"open_webui_{knowledge_id.replace('-', '_')}")

    assert [row["text"] for row in file_rows] == ["avocets sweep the mud"], (
        f"the file's own collection holds its chunks {len(file_rows)} times"
    )
    assert [row["text"] for row in knowledge_rows] == ["avocets sweep the mud"], (
        f"the knowledge base holds the file's chunks {len(knowledge_rows)} times"
    )
