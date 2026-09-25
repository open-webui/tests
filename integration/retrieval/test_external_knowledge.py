"""External knowledge: a Qdrant collection the admin connects, searched from a test and a chat.

The admin saves a Qdrant connection, tries a query against a collection and adds that collection
as a read-only knowledge base; a chat with the knowledge base attached searches it with the
embedded question and hands the points it found to the model. A local service plays Qdrant, so
the tests read the vector query it got and the context the model was sent. The admin also reads,
renames, health-checks and deletes a connection, whose key never comes back and survives an
update that leaves it out; a connection a knowledge base uses is kept. A user is refused every
external knowledge route, and Qdrant is never queried for them.

Discriminates: fails with the Qdrant branch of `retrieve_external_knowledge_for_connection`
querying `limit=1` (the connection test and the chat each see one point of two). In a backend
copy, `_get_external_auth_config` taking an omitted key as none turns the CRUD test red,
dropping the in-use check from the connection delete turns its test red, and switching
`test_external_knowledge_source` to `get_verified_user` turns the user test red (Qdrant is
queried for the user).
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


def _connections(client) -> dict[str, dict]:
    listed = client.get(CONNECTIONS)
    assert listed.status_code == 200, listed.text
    return {connection["id"]: connection for connection in listed.json()["items"]}


def _retrieve_test(client, connection_id: str):
    return client.post(
        f"{CONNECTIONS}/{connection_id}/retrieve-test",
        json={"query": "gate", "count": 1, "source": {"name": COLLECTION, "config": SOURCE_CONFIG}},
    )


def test_the_admin_saves_reads_updates_and_deletes_a_connection(admin, listener):
    form = serve_qdrant(listener, COLLECTION, POINTS)
    with admin.client() as client:
        created = client.post(CONNECTIONS, json=form)
        assert created.status_code == 200, created.text
        connection_id = created.json()["id"]
        listed = _connections(client)[connection_id]
        read = client.get(f"{CONNECTIONS}/{connection_id}")
        renamed = {key: value for key, value in form.items() if key != "auth_config"}
        updated = client.patch(
            f"{CONNECTIONS}/{connection_id}", json={**renamed, "name": "Harbour vectors"}
        )
        tried = _retrieve_test(client, connection_id)
        deleted = client.delete(f"{CONNECTIONS}/{connection_id}")
        gone = client.get(f"{CONNECTIONS}/{connection_id}")
        remaining = _connections(client)

    assert listed["name"] == "Test Qdrant" and listed["auth_configured"] is True, listed
    assert QDRANT_API_KEY not in created.text + json.dumps(listed) + read.text
    assert read.status_code == 200 and read.json()["endpoint"] == listener.base_url, read.text
    assert updated.status_code == 200 and updated.json()["name"] == "Harbour vectors", updated.text
    assert tried.status_code == 200, tried.text
    [call] = listener.requests_to(f"/collections/{COLLECTION}/points/query")
    assert call.headers.get("api-key") == QDRANT_API_KEY, "an update without a key dropped it"
    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    assert gone.status_code == 404
    assert connection_id not in remaining


def test_a_connection_a_knowledge_base_uses_is_kept(admin, connection_id):
    with admin.client() as client:
        with external_knowledge_base(client, connection_id, COLLECTION):
            refused = client.delete(f"{CONNECTIONS}/{connection_id}")
            still_listed = connection_id in _connections(client)

    assert refused.status_code == 400, refused.text
    assert still_listed


def test_the_connection_test_reports_health_and_saves_it(admin, listener, connection_id):
    form = serve_qdrant(listener, COLLECTION, POINTS)
    with admin.client() as client:
        healthy = client.post(f"{CONNECTIONS}/{connection_id}/test")
        saved_health = client.get(f"{CONNECTIONS}/{connection_id}").json()["health"]
        client.patch(
            f"{CONNECTIONS}/{connection_id}", json={**form, "enabled": False}
        ).raise_for_status()
        disabled = client.post(f"{CONNECTIONS}/{connection_id}/test")

    assert healthy.status_code == 200, healthy.text
    assert healthy.json()["ok"] is True and healthy.json()["provider"] == "qdrant"
    assert saved_health == healthy.json()
    assert disabled.json()["ok"] is False


def test_an_unsaved_connection_is_tried_against_qdrant(admin, listener):
    form = serve_qdrant(listener, COLLECTION, POINTS)
    with admin.client() as client:
        before = _connections(client)
        tried = client.post(
            "/api/v1/knowledge/external/source/test",
            json={
                "connection": form,
                "source": {"name": COLLECTION, "config": SOURCE_CONFIG},
                "query": "harbour gate code",
                "count": 1,
            },
        )
        after = _connections(client)

    assert tried.status_code == 200, tried.text
    assert tried.json()["documents"] == ["The harbour gate code is 4471."]
    assert [query["limit"] for query in queries_to(listener, COLLECTION)] == [1]
    assert after.keys() == before.keys(), "trying a connection saved it"


SOURCE = {"name": COLLECTION, "config": SOURCE_CONFIG}


def _user_routes(form: dict, connection_id: str, knowledge_id: str) -> list[tuple]:
    """Every external knowledge route as (method, path, body), with bodies an admin could send."""
    source_form = {"name": "Mine", "connection": form, "source": SOURCE, "test_query": "gate"}
    return [
        ("GET", CONNECTIONS, None),
        ("POST", CONNECTIONS, form),
        ("GET", f"{CONNECTIONS}/{connection_id}", None),
        ("PATCH", f"{CONNECTIONS}/{connection_id}", {**form, "name": "taken over"}),
        ("DELETE", f"{CONNECTIONS}/{connection_id}", None),
        ("POST", f"{CONNECTIONS}/{connection_id}/test", None),
        (
            "POST",
            f"{CONNECTIONS}/{connection_id}/retrieve-test",
            {"query": "gate", "source": SOURCE},
        ),
        (
            "POST",
            "/api/v1/knowledge/external/source/test",
            {"connection": form, "source": SOURCE, "query": "gate"},
        ),
        (
            "POST",
            "/api/v1/knowledge/external/knowledge/create",
            {"name": "Mine", "connection_id": connection_id, "source": SOURCE},
        ),
        ("POST", "/api/v1/knowledge/external/source/create", source_form),
        ("PATCH", f"/api/v1/knowledge/external/source/{knowledge_id}", source_form),
    ]


def test_a_user_is_refused_every_external_knowledge_route(admin, make_user, listener):
    form = serve_qdrant(listener, COLLECTION, POINTS)
    with (
        admin.client() as admin_client,
        external_connection(admin_client, form) as created,
        external_knowledge_base(admin_client, created, COLLECTION) as knowledge_id,
    ):
        before = _connections(admin_client)
        listener.received.clear()
        with make_user().client() as client:
            answered = {
                f"{method} {path}": client.request(method, path, json=body).status_code
                for method, path, body in _user_routes(form, created, knowledge_id)
            }
        after = _connections(admin_client)
        knowledge = admin_client.get(f"/api/v1/knowledge/{knowledge_id}").json()

    reached = {route: status for route, status in answered.items() if status != 401}
    assert not reached, f"external knowledge routes a user was not refused on: {reached}"
    assert listener.received == [], "Qdrant was queried for a user"
    assert after == before
    assert knowledge["name"] == "Team vectors"
