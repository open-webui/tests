"""External knowledge: a Qdrant collection the admin connects, searched from a test and a chat.

The admin saves a Qdrant connection, tries a query against a collection and adds that collection
as a read-only knowledge base; a chat with the knowledge base attached searches it with the
embedded question and hands the points it found to the model. A local service plays Qdrant, so
the tests read the vector query it got and the context the model was sent.

Discriminates: fails with the Qdrant branch of `retrieve_external_knowledge_for_connection`
querying `limit=1` (the connection test and the chat each see one point of two).
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.external_knowledge import (
    CONNECTIONS,
    QDRANT_API_KEY,
    SOURCE_CONFIG,
    external_connection,
    external_knowledge_base,
    point,
    queries_to,
    serve_qdrant,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

COLLECTION = "team_docs"
POINTS = [
    point(11, "The harbour gate code is 4471.", 0.92, source="gates.md", page=2),
    point(12, "Visitors sign in at the harbour office.", 0.81, source="visitors.md"),
]


@pytest.fixture
def connection_id(admin, listener):
    form = serve_qdrant(listener, COLLECTION, POINTS)
    with admin.client() as client, external_connection(client, form) as created:
        yield created


def test_the_admin_retrieval_test_searches_the_collection(admin, listener, upstream, connection_id):
    with admin.client() as client:
        tried = client.post(
            f"{CONNECTIONS}/{connection_id}/retrieve-test",
            json={
                "query": "harbour gate code",
                "count": 3,
                "source": {"name": COLLECTION, "config": SOURCE_CONFIG},
            },
        )

    assert tried.status_code == 200, tried.text
    found = tried.json()
    assert found["documents"] == [
        "The harbour gate code is 4471.",
        "Visitors sign in at the harbour office.",
    ]
    assert found["distances"] == [0.92, 0.81]
    assert found["metadatas"][0]["source"] == "gates.md" and found["metadatas"][0]["page"] == 2
    [query] = queries_to(listener, COLLECTION)
    assert query["query"] == {"nearest": [0.1, 0.2, 0.3]}  # the mock provider's embedding
    assert query["limit"] == 3 and query["with_payload"] is True
    [call] = listener.requests_to(f"/collections/{COLLECTION}/points/query")
    assert call.headers.get("api-key") == QDRANT_API_KEY
    embedded = [entry.body["input"] for entry in upstream.requests_to("/embeddings")]
    assert "harbour gate code" in json.dumps(embedded)


def test_a_chat_with_the_knowledge_base_attached_gets_its_points(
    admin, listener, upstream, connection_id
):
    upstream.queue(reply.text("The code is 4471.", match=reply.answering("gate code")))
    with admin.client() as client, external_knowledge_base(client, connection_id, COLLECTION) as kb:
        _, answer = ask(
            client,
            "what is the harbour gate code?",
            files=[{"type": "collection", "id": kb, "name": "Team vectors"}],
        )

    assert answer["content"] == "The code is 4471."
    [query] = queries_to(listener, COLLECTION)
    assert query["query"] == {"nearest": [0.1, 0.2, 0.3]}
    [chat_request] = [
        body for body in upstream.chat_requests() if reply.answering("gate code")(body)
    ]
    sent = json.dumps(chat_request["messages"])
    assert "The harbour gate code is 4471." in sent
    assert "Visitors sign in at the harbour office." in sent
