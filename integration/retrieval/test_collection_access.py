"""Regression: who may read a vector collection, and a web search collection reaching the model.

- Collection access (`filter_accessible_collections`, the gate behind `/query/doc` and every
  chat retrieval): a malformed name is refused before the admin bypass, `user-memory-*` belongs
  to its user, `file-*` follows the file's access, `knowledge-bases` is never exposed and any
  other name must be a knowledge base the user can read. The web-search namespace is pinned by
  integration/security/test_websearch_collection_owner_scope.py.
- open-webui#25585 (v0.9.6): the middleware adds a web search's results as a `web_search` item
  carrying the server-minted `collection_name`, and `get_sources_from_items` had no branch for
  that type. It fell through to the untyped `collection_name` gate, which drops the name unless
  `BYPASS_RETRIEVAL_ACCESS_CONTROL` is on, so the pages were fetched and embedded and then never
  queried; the model answered without them. An untyped item must still be dropped.

The escape hatches `ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS` and `BYPASS_RETRIEVAL_ACCESS_CONTROL`
run on one instance booted with them, shared with test_knowledge_lifecycle.

Twin of unit/retrieval/test_collection_access.py.

Discriminates: passes on dev bbfa876af; dropping the `web_search` dispatch branch fails the web
search answer, admitting admins before the name check fails the malformed names, and admitting
any `user-memory-`, `file-` or knowledge base name fails that case of the refusal matrix.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# shares one boot with test_knowledge_lifecycle
ESCAPE_HATCHES = {
    "ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS": "true",
    "BYPASS_RETRIEVAL_ACCESS_CONTROL": "true",
    "ENABLE_KNOWLEDGE_FILE_RETENTION": "true",
}
LEGACY_WEB_SEARCH = {"features": {"web_search": True}, "params": {"function_calling": "legacy"}}


def query(actor, collection_name: str):
    with actor.client() as client:
        return client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": collection_name, "query": "what is stored here"},
        )


def upload(actor, text: str, process: bool = True) -> str:
    with actor.client() as client:
        uploaded = client.post(
            f"/api/v1/files/?process={str(process).lower()}&process_in_background=false",
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def knowledge_base(owner, file_id: str | None = None, reader=None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/knowledge/create", json={"name": "kb", "description": ""})
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        if file_id:
            added = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
            )
            assert added.status_code == 200, added.text
        if reader:
            grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
            granted = client.post(
                f"/api/v1/knowledge/{knowledge_id}/access/update", json={"access_grants": [grant]}
            )
            assert granted.status_code == 200, granted.text
    return knowledge_id


def documents_of(response) -> list[str]:
    assert response.status_code == 200, response.text
    return response.json()["documents"][0]


@pytest.fixture
def strangers_collections(make_user, admin) -> dict[str, str]:
    """Collection names that belong to someone other than the asking user, by kind."""
    stranger = make_user()
    return {
        "another user's memory": f"user-memory-{stranger.id}",
        "another user's file": f"file-{upload(stranger, 'not yours', process=False)}",
        "an unshared knowledge base": knowledge_base(admin),
        "the knowledge base index": "knowledge-bases",
        "a name no knowledge base has": f"legacy-{uuid.uuid4().hex[:12]}",
    }


@pytest.mark.parametrize(
    "kind",
    [
        "another user's memory",
        "another user's file",
        "an unshared knowledge base",
        "the knowledge base index",
        "a name no knowledge base has",
    ],
)
def test_a_user_is_refused_a_collection_that_is_not_theirs(make_user, strangers_collections, kind):
    name = strangers_collections[kind]

    assert query(make_user(), name).status_code == 403, f"{kind} ({name}) was readable"


def test_a_user_reads_their_own_file_and_memory(make_user):
    reader = make_user()
    file_id = upload(reader, "my own heron count")
    with reader.client() as client:
        remembered = client.post("/api/v1/memories/add", json={"content": "I keep bees"})
    assert remembered.status_code == 200, remembered.text

    assert documents_of(query(reader, f"file-{file_id}")) == ["my own heron count"]
    assert "I keep bees" in documents_of(query(reader, f"user-memory-{reader.id}"))[0]


def test_a_knowledge_base_shared_with_a_user_is_readable(make_user, admin):
    reader = make_user()
    knowledge_id = knowledge_base(admin, upload(admin, "shared osprey notes"), reader=reader)

    assert documents_of(query(reader, knowledge_id)) == ["shared osprey notes"]


def test_an_admin_reads_another_users_file(make_user, admin):
    file_id = upload(make_user(), "a user's heron count")

    assert documents_of(query(admin, f"file-{file_id}")) == ["a user's heron count"]


@pytest.mark.parametrize("name", ["../etc/passwd", "a/b", "a b", "a.b", "a'b"])
def test_an_admin_is_refused_a_malformed_name(admin, name):
    assert query(admin, name).status_code == 403, f"{name!r} reached the vector store"


@pytest.fixture
def searching(admin, preserve, listener):
    """Web search through a local engine whose result snippets are embedded, pages not loaded."""
    preserve(RETRIEVAL_CONFIG)
    pages = [f"{listener.base_url}/kestrel-migration"]
    with admin.client() as client:
        save_web_settings(
            client,
            **serve_search_results(listener, pages),
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=False,
        )
    return f"snippet of {pages[0]}"


def sent_text(upstream) -> str:
    return str(upstream.chat_requests()[-1]["messages"])


def test_a_web_search_result_reaches_the_model(searching, make_user, upstream):
    upstream.queue(reply.text("they fly south"))
    with make_user().client() as client:
        _, answer = ask(client, "where do kestrels migrate", **LEGACY_WEB_SEARCH)

    assert searching in sent_text(upstream), (
        "the web search collection was never queried, so the model answered without the "
        "results (#25585)"
    )
    assert answer["sources"], "the stored reply lists no web source"


def test_an_untyped_collection_item_is_not_queried(searching, make_user, upstream):
    searcher = make_user()
    with searcher.client() as client:
        searched = client.post(
            "/api/v1/retrieval/process/web/search", json={"queries": ["kestrel"]}
        )
        assert searched.status_code == 200, searched.text
        untyped = {"collection_name": searched.json()["collection_names"][0], "name": "kestrel"}
        ask(client, "where do kestrels migrate", files=[untyped])

    assert searching not in sent_text(upstream), "a bare client-supplied collection was queried"


@pytest.fixture(scope="module")
def unguarded(instance_with):
    return instance_with(ESCAPE_HATCHES)


@pytest.mark.slow
def test_unscoped_collections_are_readable_when_enabled(unguarded):
    reader = create_user(unguarded)
    name = f"legacy-{uuid.uuid4().hex[:12]}"
    with unguarded.client() as client:
        stored = client.post(
            "/api/v1/retrieval/process/text",
            json={"name": "legacy", "content": "legacy text", "collection_name": name},
        )
    assert stored.status_code == 200, stored.text

    assert documents_of(query(reader, name)) == ["legacy text"]


@pytest.mark.slow
def test_an_untyped_collection_item_is_queried_when_access_control_is_bypassed(unguarded):
    searcher = create_user(unguarded)
    with unguarded.client() as client:
        stored = client.post(
            "/api/v1/retrieval/process/text",
            json={"name": "bypass", "content": "bypass payload", "collection_name": "bypass-coll"},
        )
    assert stored.status_code == 200, stored.text
    with searcher.client() as client:
        ask(client, "what is in it", files=[{"collection_name": "bypass-coll", "name": "bypass"}])

    assert "bypass payload" in sent_text(unguarded.upstream)
