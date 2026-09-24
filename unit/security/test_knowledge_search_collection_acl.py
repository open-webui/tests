"""Knowledge-base search must not return chunks of knowledge bases the caller cannot open.

`query_knowledge_bases` hands the vector store `filter={'knowledge_base_id': {'$in': [...]}}`,
the knowledge bases the caller may read. Before commit 1d6d4e6e6 (v0.11.1) eleven bundled
backends took that `filter` argument of `search()` and dropped it, so the store answered with
every neighbour, forbidden knowledge bases included. Qdrant sent no `query_filter`, pinecone
only the collection name and elasticsearch only the collection term.

This needs a real store, which the integration suite does not have, so it stays a unit test.
Qdrant runs for real in its local in-memory mode; pinecone and elasticsearch get specced clients
that apply the filter they are sent, so a dropped filter shows as the forbidden row coming back.
Every store is filled through the backend's own `insert`. The default Chroma store always applied
the filter; integration/security/test_knowledge_search_collection_acl.py guards the caller side.

Discriminates: passes on bbfa876af, fails with 1d6d4e6e6 reverted (every `search` test returns
the forbidden knowledge base, the audit lists the eleven backends whose `search` ignores `filter`).
"""

from __future__ import annotations

import ast
import uuid
from unittest.mock import create_autospec

import pytest
import qdrant_client
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from pinecone.core.openapi.db_data.models import QueryResponse, ScoredVector
from pinecone.grpc import GRPCIndex, PineconeGRPC

pytestmark = pytest.mark.regression

KB_COLLECTION = "knowledge-bases"
OTHER_COLLECTION = "other-collection"
EMBEDDING = [0.1, 0.2, 0.3]
ALLOWED_KB, FORBIDDEN_KB, OTHER_ROW = (str(uuid.uuid4()) for _ in range(3))
ACCESS_FILTER = {"knowledge_base_id": {"$in": [ALLOWED_KB]}}

# (collection, row id, knowledge base id), shaped like the knowledge-base metadata embeddings
STORED_ROWS = [
    (KB_COLLECTION, ALLOWED_KB, ALLOWED_KB),
    (KB_COLLECTION, FORBIDDEN_KB, FORBIDDEN_KB),
    (OTHER_COLLECTION, OTHER_ROW, ALLOWED_KB),
]


def qdrant_store(module, monkeypatch):
    monkeypatch.setattr(module, "QDRANT_URI", "http://qdrant.invalid:6333")
    monkeypatch.setattr(module, "Qclient", lambda **_: qdrant_client.QdrantClient(":memory:"))
    return module.QdrantClient()


def pinecone_accepts(metadata: dict, conditions: dict) -> bool:
    for key, condition in conditions.items():
        if isinstance(condition, dict):
            ((operator, operand),) = condition.items()
            accepted = {"$in": operand, "$eq": [operand]}[operator]
        else:
            accepted = [condition]
        if metadata.get(key) not in accepted:
            return False
    return True


def pinecone_store(module, monkeypatch):
    stored: list[dict] = []

    def query(vector=None, top_k=10, filter=None, **_):
        matches = [
            ScoredVector(id=entry["id"], score=1.0, metadata=entry["metadata"])
            for entry in stored
            if pinecone_accepts(entry["metadata"], filter or {})
        ]
        return QueryResponse(matches=matches[:top_k], namespace="")

    index = create_autospec(GRPCIndex, instance=True)
    index.upsert.side_effect = lambda vectors, **_: stored.extend(vectors)
    index.query.side_effect = query
    client_class = create_autospec(PineconeGRPC)
    client_class.return_value.Index.return_value = index
    for client_class_name in ("Pinecone", "PineconeGRPC"):
        monkeypatch.setattr(module, client_class_name, client_class, raising=False)
    monkeypatch.setattr(module, "PINECONE_API_KEY", "pinecone-key")
    monkeypatch.setattr(module, "PINECONE_ENVIRONMENT", "us-east-1")
    return module.PineconeClient()


def elasticsearch_accepts(clause: dict, source: dict) -> bool:
    ((kind, condition),) = clause.items()
    ((path, expected),) = condition.items()
    value = source
    for part in path.split("."):
        value = (value or {}).get(part)
    return value in {"term": [expected], "terms": expected}[kind]


def elasticsearch_store(module, monkeypatch):
    documents: list[dict] = []

    def search(index=None, body=None, **_):
        clauses = body["query"]["script_score"]["query"]["bool"]["filter"]
        hits = [
            {"_id": document["_id"], "_score": 1.0, "_source": document["_source"]}
            for document in documents
            if all(elasticsearch_accepts(clause, document["_source"]) for clause in clauses)
        ]
        return {"hits": {"hits": hits[: body["size"]]}}

    client_class = create_autospec(Elasticsearch)
    client_class.return_value.search.side_effect = search
    client_class.return_value.indices = create_autospec(IndicesClient, instance=True)
    monkeypatch.setattr(module, "Elasticsearch", client_class)
    monkeypatch.setattr(module, "bulk", lambda client, actions, **_: documents.extend(actions))
    return module.ElasticsearchClient()


STORE_BUILDERS = {
    "qdrant": qdrant_store,
    "qdrant_multitenancy": qdrant_store,
    "pinecone": pinecone_store,
    "elasticsearch": elasticsearch_store,
}


@pytest.fixture(params=list(STORE_BUILDERS))
def store(request, owui_module, monkeypatch):
    module = owui_module(f"open_webui.retrieval.vector.dbs.{request.param}")
    vector_store = STORE_BUILDERS[request.param](module, monkeypatch)
    for collection, row_id, knowledge_base_id in STORED_ROWS:
        row = {
            "id": row_id,
            "text": f"knowledge base {knowledge_base_id}",
            "vector": EMBEDDING,
            "metadata": {"knowledge_base_id": knowledge_base_id},
        }
        vector_store.insert(collection_name=collection, items=[row])
    return vector_store


def test_search_returns_only_the_knowledge_bases_the_filter_allows(store):
    result = store.search(
        collection_name=KB_COLLECTION, vectors=[EMBEDDING], filter=ACCESS_FILTER, limit=10
    )
    assert result.ids == [[ALLOWED_KB]], "search() dropped the accessible-knowledge-base filter"


def test_search_without_a_filter_returns_the_whole_collection(store):
    result = store.search(collection_name=KB_COLLECTION, vectors=[EMBEDDING], limit=10)
    assert sorted(result.ids[0]) == sorted([ALLOWED_KB, FORBIDDEN_KB])


@pytest.mark.parametrize("store", ["qdrant", "qdrant_multitenancy", "pinecone"], indirect=True)
def test_query_by_metadata_still_filters(store):
    result = store.query(collection_name=KB_COLLECTION, filter={"knowledge_base_id": ALLOWED_KB})
    assert result.ids == [[ALLOWED_KB]]


def filter_parameters_never_read(source: str) -> list[str]:
    unread = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        if "filter" not in {argument.arg for argument in arguments}:
            continue
        loaded = {
            name.id
            for name in ast.walk(node)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
        }
        if "filter" not in loaded:
            unread.append(node.name)
    return unread


def test_no_vector_backend_ignores_the_filter_it_is_given(open_webui_backend):
    backends = sorted((open_webui_backend / "open_webui/retrieval/vector/dbs").glob("*.py"))
    sources = {path.stem: path.read_text(encoding="utf-8") for path in backends}
    assert any("def search(" in source for source in sources.values()), (
        "no vector backend defines search() any more; retarget this audit"
    )
    unread = {
        f"{name}.{function}"
        for name, source in sources.items()
        for function in filter_parameters_never_read(source)
    }
    assert not unread, f"these vector backend methods drop their `filter` argument: {unread}"
