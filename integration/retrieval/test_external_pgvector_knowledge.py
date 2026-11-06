"""External knowledge: a pgvector table the admin connects, searched from a test and a chat.

The admin tries a query against a table in a Postgres database and adds one collection of it as a
read-only knowledge base; a chat with the knowledge base attached searches that collection with
the embedded question and hands the nearest rows to the model. The database is a real Postgres
with the vector extension, filled with rows in Open WebUI's own table layout, so the tests read
what the search returned rather than what a fake was asked. A pgvector connection keeps no
credentials of its own (they belong in the database URL) and the source needs a vector field.

Discriminates: passes on dev 015dbc861; the search tests fail on dev a5bc78300, where the query
embedding reaches Postgres as a float array and every search fails (open-webui/open-webui#26663),
until PR #31112 sends it as a vector. In a backend copy, filtering on the other collections turns
the unsaved-connection and chat tests red (the other rows come back), ordering by descending
distance turns the two route tests red, keeping the key of a pgvector connection turns the
credentials test red, dropping the vector field check turns the refusal test red and reading a
schema-qualified name as one identifier turns the schema test red.
"""

from __future__ import annotations

import json

import psycopg
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.external_knowledge import (
    CONNECTIONS,
    external_connection,
    external_knowledge_base,
)
from harness.pgvector_source import COLLECTION, SOURCE_CONFIG, chunk, serving_pgvector

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.requires_postgres,
]

CHUNKS = [
    chunk("far", "Moorings are inspected every May.", [0.3, 0.2, 0.1], source="moorings.md"),
    chunk(
        "near",
        "The tide tables hang in the harbour office.",
        [0.1, 0.2, 0.3],
        source="tides.md",
        page=3,
    ),
    chunk("elsewhere", "The canteen closes at four.", [0.1, 0.2, 0.3], collection="canteen_docs"),
]
SOURCE = {"name": COLLECTION, "config": SOURCE_CONFIG}
NEAREST_FIRST = [
    "The tide tables hang in the harbour office.",
    "Moorings are inspected every May.",
]


@pytest.fixture
def source_form():
    with serving_pgvector(CHUNKS) as form:
        yield form


@pytest.fixture
def schema_copy(source_form):
    """The rows again in another schema."""
    with psycopg.connect(source_form["endpoint"], autocommit=True) as conn:
        conn.execute("CREATE SCHEMA harbour")
        conn.execute("CREATE TABLE harbour.document_chunk AS TABLE document_chunk")


@pytest.fixture
def connection_id(admin, source_form):
    with admin.client() as client, external_connection(client, source_form) as created:
        yield created


def test_an_unsaved_connection_is_tried_against_the_table(admin, source_form):
    with admin.client() as client:
        tried = client.post(
            "/api/v1/knowledge/external/source/test",
            json={
                "connection": source_form,
                "source": SOURCE,
                "query": "where are the tide tables",
                "count": 5,
            },
        )

    assert tried.status_code == 200, tried.text
    found = tried.json()
    assert found["documents"] == NEAREST_FIRST
    assert found["metadatas"][0]["source"] == "tides.md" and found["metadatas"][0]["page"] == 3
    assert found["distances"] == sorted(found["distances"])


def test_the_retrieval_test_of_a_saved_connection_returns_only_as_many_rows_as_asked(
    admin, connection_id
):
    with admin.client() as client:
        tried = client.post(
            f"{CONNECTIONS}/{connection_id}/retrieve-test",
            json={"query": "tide tables", "count": 1, "source": SOURCE},
        )

    assert tried.status_code == 200, tried.text
    assert tried.json()["documents"] == NEAREST_FIRST[:1]


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_the_knowledge_base_attached_gets_its_nearest_rows(
    admin, upstream, connection_id
):
    upstream.queue(reply.text("In the harbour office.", match=reply.answering("tide tables")))
    with (
        admin.client() as client,
        external_knowledge_base(client, connection_id, COLLECTION, config=SOURCE_CONFIG) as kb,
    ):
        _, answer = ask(
            client,
            "where are the tide tables?",
            files=[{"type": "collection", "id": kb, "name": "Team vectors"}],
        )

    assert answer["content"] == "In the harbour office."
    [chat_request] = [
        body for body in upstream.chat_requests() if reply.answering("tide tables")(body)
    ]
    sent = json.dumps(chat_request["messages"])
    assert NEAREST_FIRST[0] in sent and NEAREST_FIRST[1] in sent
    assert "canteen" not in sent, "a row of another collection was sent"


def test_a_pgvector_connection_keeps_no_credentials_of_its_own(admin, source_form):
    with admin.client() as client:
        created = client.post(
            CONNECTIONS, json={**source_form, "auth_config": {"type": "bearer", "api_key": "k"}}
        )
        assert created.status_code == 200, created.text
        connection_id = created.json()["id"]
        try:
            listed = client.get(f"{CONNECTIONS}/{connection_id}")
        finally:
            client.delete(f"{CONNECTIONS}/{connection_id}")

    assert listed.json()["auth_configured"] is False
    assert '"api_key"' not in listed.text


def test_a_source_without_a_vector_field_is_refused(admin, connection_id):
    config = {key: value for key, value in SOURCE_CONFIG.items() if key != "vector_field"}
    with admin.client() as client:
        refused = client.post(
            "/api/v1/knowledge/external/knowledge/create",
            json={
                "name": "No vectors",
                "connection_id": connection_id,
                "source": {"name": COLLECTION, "config": config},
            },
        )

    assert refused.status_code == 400, refused.text
    assert "Vector field" in refused.text


def _try_table(admin, source_form, table_name):
    config = {**SOURCE_CONFIG, "table_name": table_name}
    with admin.client() as client:
        return client.post(
            "/api/v1/knowledge/external/source/test",
            json={
                "connection": source_form,
                "source": {"name": COLLECTION, "config": config},
                "query": "where are the tide tables",
                "count": 5,
            },
        )


def test_a_table_in_another_schema_is_searched(admin, source_form, schema_copy):
    tried = _try_table(admin, source_form, "harbour.document_chunk")

    assert tried.status_code == 200, tried.text
    assert tried.json()["documents"] == NEAREST_FIRST
