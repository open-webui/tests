"""A Qdrant, Milvus or vector store server played by a local service, and the knowledge it serves.

`serve_qdrant(listener, collection, points)` routes a listener as a Qdrant REST server whose
`collection` answers a vector query with the first `limit` of `points` (built with `point(...)`)
and returns the admin's connection form for it. `queries_to(listener, collection)` is every query
body it got.
`serve_milvus(fake, collection, rows)` does the same for a `FakeMilvus` (harness/milvus_server.py)
holding `collection` with `rows` (built with `milvus_row(...)`), ranked by cosine against the
embedded question, and `searches_of(fake, collection)` is every vector search it got.
`external_connection(client, form)` saves that connection and `external_knowledge_base(...)` a
read-only knowledge base on one of its collections, each removed again afterwards.
`VectorStores(listener, flavour)` routes a listener as the OpenAI-compatible vector store API of
OpenAI, Azure OpenAI (under `/openai/v1`) or LiteLLM (which lists through its own registry), with
stores added and removed while a test runs; `form()` is the admin's connection form for it.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

CONNECTIONS = "/api/v1/knowledge/external/connections"
QDRANT_API_KEY = "qdrant-key-0123456789"
MILVUS_TOKEN = "milvus-token-0123456789"
# where the payload keeps what `point` writes
SOURCE_CONFIG = {"content_field": "payload.text", "metadata_field": "payload.metadata"}
# the columns `milvus_row` fills, as the form offers them for Milvus
MILVUS_SOURCE_CONFIG = {
    "content_field": "data.text",
    "vector_field": "vector",
    "metadata_field": "metadata",
    "document_id_field": "id",
}
# the mock provider embeds every text as this
EMBEDDING = [0.1, 0.2, 0.3]


def point(point_id: int, text: str, score: float, **metadata) -> dict:
    """A scored point as Qdrant's query API returns it."""
    payload = {"text": text, "metadata": metadata}
    return {"id": point_id, "version": 1, "score": score, "payload": payload}


def serve_qdrant(listener: Listener, collection: str, points: list[dict]) -> dict:
    """Answer as Qdrant holding `collection`; returns the connection form that uses it."""
    listener.route("GET", "/", json_answer({"title": "qdrant", "version": "1.15.0"}))

    def query(request: ReceivedRequest) -> Answer:
        nearest = points[: request.json().get("limit", 10)]
        return json_answer({"result": {"points": nearest}, "status": "ok", "time": 0.001})

    listener.route("POST", f"/collections/{collection}/points/query", query)
    return {
        "name": "Test Qdrant",
        "provider": "qdrant",
        "endpoint": listener.base_url,
        "auth_config": {"api_key": QDRANT_API_KEY},
    }


def milvus_row(row_id: str, text: str, vector: list[float], **metadata) -> dict:
    """A row of a Milvus collection: the text sits in the JSON column `data`."""
    return {"id": row_id, "vector": vector, "data": {"text": text}, "metadata": metadata}


def serve_milvus(fake, collection: str, rows: list[dict], db_name: str = "") -> dict:
    """Hold `collection` in `fake`; returns the connection form that reaches it."""
    fake.hold_collection(collection, rows)
    return {
        "name": "Test Milvus",
        "provider": "milvus",
        "endpoint": fake.address,
        "auth_config": {"type": "bearer", "api_key": MILVUS_TOKEN},
        "config": {"db_name": db_name} if db_name else {},
    }


def searches_of(fake, collection: str) -> list[tuple[str, str, int, list[str]]]:
    return [search for search in fake.searches if search[0] == collection]


def queries_to(listener: Listener, collection: str) -> list[dict]:
    return [call.json() for call in listener.requests_to(f"/collections/{collection}/points/query")]


@contextlib.contextmanager
def external_connection(client: httpx.Client, form: dict) -> Iterator[str]:
    created = client.post(CONNECTIONS, json=form)
    assert created.status_code == 200, f"saving the external connection failed: {created.text}"
    connection_id = created.json()["id"]
    try:
        yield connection_id
    finally:
        client.delete(f"{CONNECTIONS}/{connection_id}")


@contextlib.contextmanager
def external_knowledge_base(
    client: httpx.Client,
    connection_id: str,
    collection: str,
    name: str = "Team vectors",
    config: dict = SOURCE_CONFIG,
) -> Iterator[str]:
    created = client.post(
        "/api/v1/knowledge/external/knowledge/create",
        json={
            "name": name,
            "connection_id": connection_id,
            "source": {"name": collection, "config": config},
        },
    )
    assert created.status_code == 200, f"creating the external knowledge failed: {created.text}"
    knowledge_id = created.json()["id"]
    try:
        yield knowledge_id
    finally:
        client.delete(f"/api/v1/knowledge/{knowledge_id}/delete")


VECTOR_STORE_API_KEY = "sk-vector-stores-0123456789"
# where each flavour serves the OpenAI vector store API, as the admin enters it as the Endpoint
VECTOR_STORE_PREFIXES = {"openai": "/v1", "azure": "/openai/v1", "litellm": "/v1"}
# a real server may return fewer stores than asked for; two per page makes the listing page
STORE_PAGE_SIZE = 2


def search_hit(text: str, filename: str, score: float, **attributes) -> dict:
    """A result of an OpenAI vector store search."""
    return {
        "file_id": f"file-{filename}",
        "filename": filename,
        "score": score,
        "attributes": attributes,
        "content": [{"type": "text", "text": text}],
    }


@dataclass
class VectorStore:
    store_id: str
    name: str
    description: str = ""
    hits: list[dict] = field(default_factory=list)


class VectorStores:
    """An OpenAI-compatible vector store endpoint holding `stores`, which a test may change."""

    def __init__(self, listener: Listener, flavour: str, stores: list[VectorStore]):
        self.listener = listener
        self.flavour = flavour
        self.prefix = VECTOR_STORE_PREFIXES[flavour]
        self.stores = {store.store_id: store for store in stores}
        self.lock = threading.Lock()
        if flavour == "litellm":
            listener.route("GET", f"{self.prefix}/vector_store/list", self._registry)
        else:
            listener.route("GET", f"{self.prefix}/vector_stores", self._list)
        listener.route("POST", f"{self.prefix}/vector_stores/*", self._search)

    @property
    def endpoint(self) -> str:
        return f"{self.listener.base_url}{self.prefix}"

    def form(self, name: str = "Test vector stores", **config) -> dict:
        return {
            "name": name,
            "provider": "openai",
            "endpoint": self.endpoint,
            "auth_config": {"type": "bearer", "api_key": VECTOR_STORE_API_KEY},
            "config": config,
        }

    def add(self, store: VectorStore) -> None:
        with self.lock:
            self.stores[store.store_id] = store

    def remove(self, store_id: str) -> None:
        with self.lock:
            del self.stores[store_id]

    def searches(self) -> dict[str, list[ReceivedRequest]]:
        """Every search request, by the store it went to."""
        with self.listener.lock:
            received = list(self.listener.received)
        found: dict[str, list[ReceivedRequest]] = {}
        for request in received:
            path = request.path.split("?")[0]
            if request.method == "POST" and path.endswith("/search"):
                store_id = path.removeprefix(f"{self.prefix}/vector_stores/").removesuffix(
                    "/search"
                )
                found.setdefault(store_id, []).append(request)
        return found

    def _authorised(self, request: ReceivedRequest) -> bool:
        return request.headers.get("Authorization") == f"Bearer {VECTOR_STORE_API_KEY}"

    def _refused(self) -> Answer:
        return json_answer({"error": {"message": "Incorrect API key provided"}}, status=401)

    def _held(self) -> list[VectorStore]:
        with self.lock:
            return list(self.stores.values())

    def _list(self, request: ReceivedRequest) -> Answer:
        if not self._authorised(request):
            return self._refused()
        query = parse_qs(urlsplit(request.path).query)
        limit = min(int(query.get("limit", ["20"])[0]), STORE_PAGE_SIZE)
        held = self._held()
        ids = [store.store_id for store in held]
        after = query.get("after", [None])[0]
        start = ids.index(after) + 1 if after in ids else 0
        page = held[start : start + limit]
        data = [
            {
                "id": store.store_id,
                "object": "vector_store",
                "name": store.name,
                "status": "completed",
                "metadata": {"description": store.description} if store.description else {},
            }
            for store in page
        ]
        return json_answer(
            {
                "object": "list",
                "data": data,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
                "has_more": start + limit < len(held),
            }
        )

    def _registry(self, request: ReceivedRequest) -> Answer:
        if not self._authorised(request):
            return self._refused()
        data = [
            {
                "vector_store_id": store.store_id,
                "custom_llm_provider": "openai",
                "vector_store_name": store.name,
                "vector_store_description": store.description or None,
            }
            for store in self._held()
        ]
        return json_answer(
            {
                "object": "list",
                "data": data,
                "total_count": len(data),
                "current_page": 1,
                "total_pages": 1,
            }
        )

    def _search(self, request: ReceivedRequest) -> Answer:
        if not self._authorised(request):
            return self._refused()
        path = request.path.split("?")[0]
        store_id = path.removeprefix(f"{self.prefix}/vector_stores/").removesuffix("/search")
        with self.lock:
            store = self.stores.get(store_id)
        if not store or not path.endswith("/search"):
            return json_answer(
                {"error": {"message": f"No vector store found with id '{store_id}'"}}, status=404
            )
        body = request.json()
        hits = store.hits[: body.get("max_num_results", 10)]
        return json_answer(
            {
                "object": "vector_store.search_results.page",
                "search_query": [body["query"]]
                if isinstance(body["query"], str)
                else body["query"],
                "data": hits,
                "has_more": False,
                "next_page": None,
            }
        )
