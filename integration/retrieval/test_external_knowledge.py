"""External knowledge: a Qdrant or Milvus collection the admin connects, searched by test and chat.

The admin saves a Qdrant connection, tries a query against a collection and adds that collection
as a read-only knowledge base; a chat with the knowledge base attached searches it with the
embedded question and hands the points it found to the model. A local service plays Qdrant, so
the tests read the vector query it got and the context the model was sent. A Milvus collection
is searched the same way through a local gRPC service, from the column names the source names, with
the connection's token and database; a Milvus source without a vector column is refused. The admin
also reads, renames, health-checks and deletes a connection, whose key never comes back and
survives an update that leaves it out; a connection a knowledge base uses is kept. A user is
refused every external knowledge route, and Qdrant is never queried for them. The points a
chat cites, through attached knowledge or the knowledge search tool, carry no file id, so no
citation links to an Open WebUI file that does not exist (open-webui#32231).

Discriminates: fails with the Qdrant branch of `retrieve_external_knowledge_for_connection`
querying `limit=1` (the connection test and the chat each see one point of two). In a backend
copy, `_get_external_auth_config` taking an omitted key as none turns the CRUD test red,
dropping the in-use check from the connection delete turns its test red, and switching
`test_external_knowledge_source` to `get_verified_user` turns the user test red (Qdrant is
queried for the user). With the Milvus search leaving out the database name the Milvus retrieval
test goes red, with its output fields emptied the Milvus chat test does, and with the vector field
no longer required for Milvus the refusal test does, and with the source's column names ignored
the custom-columns test does. Passes on dev 5032e834e; with 1369027db (open-webui#32233)
reverted both chat tests cite a file id (the point id, or `external-<knowledge id>` from the
tool), and with only its tool half reverted the tool test does.
"""

from __future__ import annotations

import base64
import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.external_knowledge import (
    CONNECTIONS,
    EMBEDDING,
    MILVUS_SOURCE_CONFIG,
    MILVUS_TOKEN,
    QDRANT_API_KEY,
    SOURCE_CONFIG,
    external_connection,
    external_knowledge_base,
    milvus_row,
    point,
    queries_to,
    searches_of,
    serve_milvus,
    serve_qdrant,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

COLLECTION = "team_docs"
POINTS = [
    point(11, "The harbour gate code is 4471.", 0.92, source="gates.md", page=2, file_id="gates"),
    point(12, "Visitors sign in at the harbour office.", 0.81, source="visitors.md"),
]

MILVUS_COLLECTION = "dock_docs"
MILVUS_ROWS = [
    milvus_row("far", "Cranes are serviced on Mondays.", [0.3, -0.2, 0.1], source="cranes.md"),
    milvus_row("near", "The dock master sits in cabin 7.", EMBEDDING, source="dock.md", page=4),
    milvus_row("mid", "Cabin 7 has a red door.", [0.3, 0.2, 0.1], source="cabins.md"),
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


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_the_knowledge_base_attached_gets_its_points(
    admin, listener, upstream, connection_id
):
    """Red on dev e24cd406e with Postgres: integer point ids make the file name lookup fail and
    the chat gets no knowledge (open-webui/open-webui#32206)."""
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
    cited = [entry for source in answer["sources"] for entry in source["metadata"]]
    assert [entry["source"] for entry in cited] == ["gates.md", "visitors.md"]
    # external points are not Open WebUI files, so no citation may link to one
    assert not [entry for entry in cited if entry.get("file_id")], cited


@pytest.mark.regression
def test_the_knowledge_search_tool_cites_external_points_without_a_file(
    admin, upstream, connection_id
):
    prompt = "search the knowledge for the gate code"
    with admin.client() as client, external_knowledge_base(client, connection_id, COLLECTION) as kb:
        upstream.queue(
            reply.tool_call(
                "query_knowledge_files",
                {"query": "gate code", "knowledge_ids": [kb]},
                match=reply.answering(prompt),
            ),
            reply.text("The code is 4471.", match=reply.answering(prompt)),
        )
        _, answer = ask(client, prompt)

    cited = [entry for source in answer["sources"] for entry in source["metadata"]]
    assert [entry["source"] for entry in cited] == ["gates.md", "visitors.md"]
    assert not [entry for entry in cited if entry.get("file_id")], cited


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
    assert still_listed, "a connection a knowledge base uses was deleted"


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


def test_the_admin_retrieval_test_searches_a_milvus_collection(admin, milvus_service):
    form = serve_milvus(milvus_service, MILVUS_COLLECTION, MILVUS_ROWS, db_name="harbour")
    with admin.client() as client, external_connection(client, form) as connection:
        tried = client.post(
            f"{CONNECTIONS}/{connection}/retrieve-test",
            json={
                "query": "who sits in the cabin?",
                "count": 2,
                "source": {"name": MILVUS_COLLECTION, "config": MILVUS_SOURCE_CONFIG},
            },
        )

    assert tried.status_code == 200, tried.text
    found = tried.json()
    assert found["documents"] == ["The dock master sits in cabin 7.", "Cabin 7 has a red door."]
    assert found["distances"][0] == pytest.approx(1.0, abs=1e-4)
    assert found["distances"][0] > found["distances"][1]
    assert found["metadatas"][0]["source"] == "dock.md" and found["metadatas"][0]["page"] == 4
    [(_, vector_field, limit, output_fields)] = searches_of(milvus_service, MILVUS_COLLECTION)
    assert (vector_field, limit) == ("vector", 2)
    assert set(output_fields) == {"data", "metadata", "id"}
    calls = milvus_service.call_metadata
    assert any(call.get("dbname") == "harbour" for call in calls), f"no call named the db: {calls}"
    token = base64.b64encode(MILVUS_TOKEN.encode()).decode()
    assert any(call.get("authorization") == token for call in calls), (
        f"no call sent the token: {calls}"
    )


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_a_milvus_knowledge_base_attached_gets_its_rows(
    admin, milvus_service, upstream
):
    form = serve_milvus(milvus_service, MILVUS_COLLECTION, MILVUS_ROWS)
    upstream.queue(reply.text("Cabin 7.", match=reply.answering("dock master")))
    with (
        admin.client() as client,
        external_connection(client, form) as connection,
        external_knowledge_base(
            client, connection, MILVUS_COLLECTION, "Dock vectors", MILVUS_SOURCE_CONFIG
        ) as kb,
    ):
        _, answer = ask(
            client,
            "where does the dock master sit?",
            files=[{"type": "collection", "id": kb, "name": "Dock vectors"}],
        )

    assert answer["content"] == "Cabin 7."
    assert searches_of(milvus_service, MILVUS_COLLECTION), "Milvus was never searched for the chat"
    [chat_request] = [
        body for body in upstream.chat_requests() if reply.answering("dock master")(body)
    ]
    sent = json.dumps(chat_request["messages"])
    assert "The dock master sits in cabin 7." in sent
    assert "Cabin 7 has a red door." in sent


def test_a_milvus_source_without_a_vector_field_is_refused(admin, milvus_service):
    form = serve_milvus(milvus_service, MILVUS_COLLECTION, MILVUS_ROWS)
    config = {key: value for key, value in MILVUS_SOURCE_CONFIG.items() if key != "vector_field"}
    with admin.client() as client:
        tried = client.post(
            "/api/v1/knowledge/external/source/test",
            json={
                "connection": form,
                "source": {"name": MILVUS_COLLECTION, "config": config},
                "query": "dock master",
                "count": 1,
            },
        )

    assert tried.status_code == 400, tried.text
    assert "Vector field is required" in tried.text
    assert searches_of(milvus_service, MILVUS_COLLECTION) == []


def test_a_milvus_source_is_searched_on_the_columns_it_names(admin, milvus_service):
    rows = [
        {"id": "a", "embedding": EMBEDDING, "body": {"text": "Berth 9 is free."}, "tags": {"n": 1}},
        {
            "id": "b",
            "embedding": [0.3, -0.2, 0.1],
            "body": {"text": "Berth 4 is full."},
            "tags": {},
        },
    ]
    config = {
        "content_field": "body.text",
        "vector_field": "embedding",
        "metadata_field": "tags",
        "document_id_field": "id",
    }
    form = serve_milvus(milvus_service, "berths", rows)
    with admin.client() as client, external_connection(client, form) as connection:
        tried = client.post(
            f"{CONNECTIONS}/{connection}/retrieve-test",
            json={
                "query": "which berth?",
                "count": 1,
                "source": {"name": "berths", "config": config},
            },
        )

    assert tried.status_code == 200, tried.text
    found = tried.json()
    assert found["documents"] == ["Berth 9 is free."]
    assert found["metadatas"][0]["n"] == 1
    [(_, vector_field, _, output_fields)] = searches_of(milvus_service, "berths")
    assert vector_field == "embedding"
    assert set(output_fields) == {"body", "tags", "id"}
