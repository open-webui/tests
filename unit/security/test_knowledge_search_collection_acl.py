"""Knowledge-base search must not reach past what the caller may read.

`query_knowledge_bases` narrows its vector search to the knowledge bases the
caller can actually open, by handing the vector store a metadata filter of the
form ``{'knowledge_base_id': {'$in': [...accessible ids...]}}``. Several of the
bundled vector backends accepted that `filter` argument and then dropped it on
the floor, so the store answered with every neighbour it had, including chunks
belonging to knowledge bases the caller cannot open.

The tests below drive the backends against fake stores that *enforce* whatever
restriction they are given, so a backend that forgets to pass the filter down
gets caught by the forbidden row coming back, not merely by a missing kwarg.

The caller side was never the bug: `query_knowledge_bases` already sent that
filter before the fix. Commit `1d6d4e6e6` is entirely in the backends, which
gained `iter_filter_conditions`/`normalize_filter` from
`retrieval/vector/utils.py` to translate the filter into store syntax.

Discriminates: passes on v0.11.1, fails on v0.11.0, where qdrant calls
`query_points` with no `query_filter` at all, pinecone sends
`filter={'collection_name': ...}` and nothing else, and elasticsearch builds
`bool.filter` from the collection term alone. On each the forbidden knowledge
base's chunk comes back.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

ALLOWED_KB = "kb-allowed"
FORBIDDEN_KB = "kb-forbidden"
KB_COLLECTION = "knowledge-bases"
ACCESS_FILTER = {"knowledge_base_id": {"$in": [ALLOWED_KB]}}


# --------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------


def _qdrant_condition_matches(condition, payload: dict) -> bool:
    value = payload
    for part in condition.key.split("."):
        value = (value or {}).get(part)
    match = condition.match
    if hasattr(match, "any"):
        return value in match.any
    return value == match.value


class FakeQdrantBackend:
    """Qdrant stand-in that honours `query_filter` the way the real server does."""

    def __init__(self, points):
        self.points = points
        self.query_points_calls = []
        self.scroll_calls = []

    def collection_exists(self, collection_name):
        return True

    def query_points(self, collection_name, query, limit, query_filter=None, **kwargs):
        self.query_points_calls.append(
            {"collection_name": collection_name, "limit": limit, "query_filter": query_filter}
        )
        kept = [
            point
            for point in self.points
            if query_filter is None
            or all(
                _qdrant_condition_matches(condition, point.payload)
                for condition in query_filter.must
            )
        ]
        return SimpleNamespace(points=kept[:limit])

    def scroll(self, collection_name, limit, scroll_filter=None, **kwargs):
        self.scroll_calls.append(
            {"collection_name": collection_name, "scroll_filter": scroll_filter}
        )
        if scroll_filter is None:
            return (list(self.points), None)
        conditions = scroll_filter.should or scroll_filter.must or []
        kept = [
            point
            for point in self.points
            if any(_qdrant_condition_matches(condition, point.payload) for condition in conditions)
        ]
        return (kept, None)


def _qdrant_point(point_id: str, knowledge_base_id: str, text: str, score: float):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={"text": text, "metadata": {"knowledge_base_id": knowledge_base_id, "name": text}},
    )


def _qdrant_points():
    return [
        _qdrant_point("point-allowed", ALLOWED_KB, "public handbook", 0.9),
        _qdrant_point("point-forbidden", FORBIDDEN_KB, "salary review notes", 0.95),
    ]


@pytest.fixture
def qdrant_client(owui_module):
    """A QdrantClient wired to a fake backend, built without touching a server."""
    module = owui_module("open_webui.retrieval.vector.dbs.qdrant")
    client = object.__new__(module.QdrantClient)
    client.collection_prefix = "open_webui"
    client.client = FakeQdrantBackend(_qdrant_points())
    return client


# --------------------------------------------------------------------------
# Pinecone
# --------------------------------------------------------------------------


def _pinecone_metadata_matches(metadata: dict, filter: dict | None) -> bool:
    for key, expected in (filter or {}).items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


class FakePineconeIndex:
    """Pinecone stand-in that applies the metadata filter it is handed."""

    def __init__(self, matches):
        self.matches = matches
        self.query_calls = []

    def query(self, vector, top_k, include_metadata=False, filter=None, **kwargs):
        self.query_calls.append({"top_k": top_k, "filter": filter})
        kept = [
            match for match in self.matches if _pinecone_metadata_matches(match.metadata, filter)
        ]
        return SimpleNamespace(matches=kept[:top_k])


def _pinecone_match(match_id: str, knowledge_base_id: str, text: str, score: float):
    return SimpleNamespace(
        id=match_id,
        score=score,
        metadata={
            "collection_name": f"open-webui_{KB_COLLECTION}",
            "knowledge_base_id": knowledge_base_id,
            "text": text,
        },
    )


@pytest.fixture
def pinecone_client(owui_module):
    module = owui_module("open_webui.retrieval.vector.dbs.pinecone")
    client = object.__new__(module.PineconeClient)
    client.collection_prefix = "open-webui"
    client.dimension = 3
    client.metric = "cosine"
    client.index = FakePineconeIndex(
        [
            _pinecone_match("match-forbidden", FORBIDDEN_KB, "salary review notes", 0.95),
            _pinecone_match("match-allowed", ALLOWED_KB, "public handbook", 0.9),
        ]
    )
    return client


# --------------------------------------------------------------------------
# Elasticsearch
# --------------------------------------------------------------------------


def _es_field(source: dict, path: str):
    value = source
    for part in path.split("."):
        value = (value or {}).get(part)
    return value


def _es_clause_matches(clause: dict, source: dict) -> bool:
    if "term" in clause:
        ((path, expected),) = clause["term"].items()
        return _es_field(source, path) == expected
    if "terms" in clause:
        ((path, expected),) = clause["terms"].items()
        return _es_field(source, path) in expected
    raise AssertionError(f"unexpected filter clause: {clause}")


class FakeElasticsearchBackend:
    """Elasticsearch stand-in that evaluates the bool filter in the query body."""

    def __init__(self, documents):
        self.documents = documents
        self.search_calls = []

    def search(self, index, body):
        self.search_calls.append({"index": index, "body": body})
        inner = body["query"]["script_score"]["query"]
        clauses = inner.get("bool", {}).get("filter", []) if "bool" in inner else []
        hits = [
            {"_id": document["_id"], "_score": document["_score"], "_source": document["_source"]}
            for document in self.documents
            if all(_es_clause_matches(clause, document["_source"]) for clause in clauses)
        ]
        return {"hits": {"hits": hits[: body["size"]]}}


def _es_document(document_id: str, knowledge_base_id: str, text: str, score: float):
    return {
        "_id": document_id,
        "_score": score,
        "_source": {
            "collection": KB_COLLECTION,
            "text": text,
            "metadata": {"knowledge_base_id": knowledge_base_id, "name": text},
        },
    }


@pytest.fixture
def elasticsearch_client(owui_module):
    module = owui_module("open_webui.retrieval.vector.dbs.elasticsearch")
    client = object.__new__(module.ElasticsearchClient)
    client.index_prefix = "open_webui"
    client.client = FakeElasticsearchBackend(
        [
            _es_document("doc-forbidden", FORBIDDEN_KB, "salary review notes", 1.95),
            _es_document("doc-allowed", ALLOWED_KB, "public handbook", 1.9),
        ]
    )
    return client


# ==========================================================================
# Narrow: the restriction the caller asks for actually reaches the store
# ==========================================================================


def test_qdrant_search_drops_chunks_from_inaccessible_knowledge_bases(qdrant_client):
    result = qdrant_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    assert result.ids == [["point-allowed"]]
    assert result.documents == [["public handbook"]]
    assert FORBIDDEN_KB not in [metadata["knowledge_base_id"] for metadata in result.metadatas[0]]


def test_qdrant_search_sends_the_allowed_ids_to_the_server(qdrant_client):
    qdrant_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    query_filter = qdrant_client.client.query_points_calls[0]["query_filter"]
    assert query_filter is not None, "the accessible-id restriction never reached qdrant"
    conditions = query_filter.must
    assert [condition.key for condition in conditions] == ["metadata.knowledge_base_id"]
    assert conditions[0].match.any == [ALLOWED_KB]


def test_pinecone_search_drops_chunks_from_inaccessible_knowledge_bases(pinecone_client):
    result = pinecone_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    assert result.ids == [["match-allowed"]]
    assert result.documents == [["public handbook"]]

    sent_filter = pinecone_client.index.query_calls[0]["filter"]
    assert sent_filter.get("knowledge_base_id") == {"$in": [ALLOWED_KB]}


def test_elasticsearch_search_drops_chunks_from_inaccessible_knowledge_bases(elasticsearch_client):
    result = elasticsearch_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    assert result.ids == [["doc-allowed"]]
    assert result.documents == [["public handbook"]]

    clauses = elasticsearch_client.client.search_calls[0]["body"]["query"]["script_score"]["query"][
        "bool"
    ]["filter"]
    assert {"terms": {"metadata.knowledge_base_id": [ALLOWED_KB]}} in clauses


def test_query_knowledge_bases_tool_hides_knowledge_bases_the_user_cannot_open(
    owui_module, monkeypatch
):
    """End to end: the tool's answer must not name a knowledge base the caller lacks."""
    builtin = owui_module("open_webui.tools.builtin")
    knowledge_models = owui_module("open_webui.models.knowledge")
    async_client_module = owui_module("open_webui.retrieval.vector.async_client")
    qdrant = owui_module("open_webui.retrieval.vector.dbs.qdrant")

    backend_client = object.__new__(qdrant.QdrantClient)
    backend_client.collection_prefix = "open_webui"
    # the knowledge-bases collection keys each point by the knowledge base's own id
    backend_client.client = FakeQdrantBackend(
        [
            _qdrant_point(ALLOWED_KB, ALLOWED_KB, "public handbook", 0.9),
            _qdrant_point(FORBIDDEN_KB, FORBIDDEN_KB, "salary review notes", 0.95),
        ]
    )
    monkeypatch.setattr(
        async_client_module,
        "ASYNC_VECTOR_DB_CLIENT",
        async_client_module.AsyncVectorDBClient(backend_client),
    )

    accessible = SimpleNamespace(items=[SimpleNamespace(id=ALLOWED_KB)])
    knowledge_bases = {
        ALLOWED_KB: SimpleNamespace(id=ALLOWED_KB, name="Handbook", description="public"),
        FORBIDDEN_KB: SimpleNamespace(id=FORBIDDEN_KB, name="HR Private", description="salaries"),
    }

    async def fake_search_knowledge_bases(user_id, filter, skip=0, limit=30, **kwargs):
        return accessible

    async def fake_get_knowledge_by_id(knowledge_base_id, *args, **kwargs):
        return knowledge_bases.get(knowledge_base_id)

    monkeypatch.setattr(
        knowledge_models.Knowledges, "search_knowledge_bases", fake_search_knowledge_bases
    )
    monkeypatch.setattr(
        knowledge_models.Knowledges, "get_knowledge_by_id", fake_get_knowledge_by_id
    )

    async def fake_get_groups_by_member_id(user_id, *args, **kwargs):
        return []

    monkeypatch.setattr(builtin.Groups, "get_groups_by_member_id", fake_get_groups_by_member_id)

    async def fake_embedding_function(query, prefix=None, user=None):
        return [0.1, 0.2, 0.3]

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(EMBEDDING_FUNCTION=fake_embedding_function))
    )
    user = {
        "id": "user-1",
        "email": "user@example.com",
        "name": "User One",
        "role": "user",
        "last_active_at": 0,
        "updated_at": 0,
        "created_at": 0,
    }

    answer = json.loads(
        asyncio.run(
            builtin.query_knowledge_bases(query="pay", count=5, __request__=request, __user__=user)
        )
    )

    assert isinstance(answer, list), answer
    assert [entry["id"] for entry in answer] == [ALLOWED_KB]


# ==========================================================================
# Broad: the rest of the search contract still holds
# ==========================================================================


def test_qdrant_search_without_a_filter_still_returns_the_whole_collection(qdrant_client):
    result = qdrant_client.search(
        collection_name=KB_COLLECTION, vectors=[[0.1, 0.2, 0.3]], limit=10
    )

    assert sorted(result.ids[0]) == ["point-allowed", "point-forbidden"]
    assert (
        qdrant_client.client.query_points_calls[0]["collection_name"]
        == f"open_webui_{KB_COLLECTION}"
    )
    # qdrant scores live in [-1, 1] and are reported normalized to [0, 1]
    assert result.distances == [[0.95, 0.975]]


def test_pinecone_search_still_pins_the_collection_name(pinecone_client):
    pinecone_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    sent_filter = pinecone_client.index.query_calls[0]["filter"]
    assert sent_filter["collection_name"] == f"open-webui_{KB_COLLECTION}"


def test_elasticsearch_search_still_pins_the_collection_name(elasticsearch_client):
    elasticsearch_client.search(
        collection_name=KB_COLLECTION,
        vectors=[[0.1, 0.2, 0.3]],
        filter=ACCESS_FILTER,
        limit=10,
    )

    call = elasticsearch_client.client.search_calls[0]
    assert call["index"] == "open_webui_d3"
    clauses = call["body"]["query"]["script_score"]["query"]["bool"]["filter"]
    assert {"term": {"collection": KB_COLLECTION}} in clauses


# ==========================================================================
# Nearby: adjacent metadata-filter paths that were already correct
# ==========================================================================


def test_qdrant_query_by_metadata_filter_is_unaffected(qdrant_client):
    result = qdrant_client.query(
        collection_name=KB_COLLECTION, filter={"knowledge_base_id": ALLOWED_KB}
    )

    assert result.ids == [["point-allowed"]]
    assert qdrant_client.client.scroll_calls[0]["collection_name"] == f"open_webui_{KB_COLLECTION}"


def test_pinecone_query_merges_caller_filter_with_collection_name(pinecone_client):
    result = pinecone_client.query(
        collection_name=KB_COLLECTION, filter={"knowledge_base_id": ALLOWED_KB}
    )

    assert result.ids == [["match-allowed"]]
    sent_filter = pinecone_client.index.query_calls[0]["filter"]
    assert sent_filter == {
        "collection_name": f"open-webui_{KB_COLLECTION}",
        "knowledge_base_id": ALLOWED_KB,
    }
