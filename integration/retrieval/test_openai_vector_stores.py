"""External knowledge on OpenAI-compatible vector stores (OpenAI, Azure OpenAI or LiteLLM).

The admin enters the endpoint and key of an OpenAI-compatible vector store API, which lists its
stores, and builds a knowledge base on the stores they pick once a test query returned documents.
The server runs the search itself, so a chat with that knowledge base attached sends the question
as text to every picked store and hands the model what came back, with no embedding of its own.
With 'List stores as knowledge bases' on, every store of the endpoint, or only the Store IDs
named, shows up as a read-only knowledge base for whoever the connection's access allows, follows
the stores the endpoint holds, and answers a chat that attaches it. A local service plays the
vector store API of each provider, so the tests read what it was asked and what the model was sent.
An admin's Delete on a listed store removed the whole connection (open-webui/open-webui#35042).

Discriminates: passes on dev f80a4f782 except the delete test (#35042), which passes with its fix
branch. In a backend copy, with the search sending the store id as the query the picked-store chat
test fails; with the listing reading only the first page the verify test fails for OpenAI and
Azure (two stores of three); with the Store IDs filter ignored the listing test fails (all three
stores listed); with the connection access check always true the access tests fail (a stranger
sees and searches the stores); with the read-only guard dropped from the file add route the
picked-store chat test fails (the file is added).
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.chat import ask
from harness.external_knowledge import (
    CONNECTIONS,
    VECTOR_STORE_API_KEY,
    VectorStore,
    VectorStores,
    external_connection,
    search_hit,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FLAVOURS = ["openai", "azure", "litellm"]
SOURCE_CREATE = "/api/v1/knowledge/external/source/create"


def _stores() -> list[VectorStore]:
    return [
        VectorStore(
            "vs_handbook",
            "Harbour handbook",
            "Rules for the harbour",
            [search_hit("The harbour gate code is 4471.", "gates.md", 0.91)],
        ),
        VectorStore(
            "vs_crew",
            "Crew notes",
            "",
            [search_hit("The night watch starts at 22:00.", "watch.md", 0.83)],
        ),
        VectorStore(
            "vs_archive",
            "Old archive",
            "Retired documents",
            [search_hit("The old gate code was 1234.", "archive.md", 0.99)],
        ),
    ]


def _cited_names(answer: dict) -> list[str]:
    return sorted(entry.get("name") for source in answer["sources"] for entry in source["metadata"])


def _sent_to_model(upstream, prompt: str) -> str:
    [request] = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    return json.dumps(request["messages"])


def _listed_for(client, connection_id: str) -> dict[str, dict]:
    """What an account sees in Workspace > Knowledge that comes from the connection."""
    listed = client.get("/api/v1/knowledge/search", params={"source": "external"})
    assert listed.status_code == 200, listed.text
    return {
        item["name"]: item
        for item in listed.json()["items"]
        if (item.get("meta") or {}).get("external", {}).get("connection_id") == connection_id
    }


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_verify_connection_lists_every_store_of_the_endpoint(admin, listener, flavour):
    endpoint = VectorStores(listener, flavour, _stores())
    with admin.client() as client:
        verified = client.post(
            "/api/v1/knowledge/external/source/stores", json={"connection": endpoint.form()}
        )
        wrong_key = {**endpoint.form(), "auth_config": {"type": "bearer", "api_key": "sk-wrong"}}
        refused = client.post(
            "/api/v1/knowledge/external/source/stores", json={"connection": wrong_key}
        )

    assert verified.status_code == 200, verified.text
    assert sorted(verified.json()["items"], key=lambda store: store["id"]) == [
        {"id": "vs_archive", "name": "Old archive", "description": "Retired documents"},
        {"id": "vs_crew", "name": "Crew notes", "description": ""},
        {"id": "vs_handbook", "name": "Harbour handbook", "description": "Rules for the harbour"},
    ], f"Verify Connection did not load the endpoint's three stores: {verified.text}"
    assert refused.status_code >= 400, (
        f"a wrong API key was taken as a working connection: {refused.status_code} {refused.text}"
    )
    assert VECTOR_STORE_API_KEY not in refused.text + verified.text


@pytest.mark.usefixtures("file_context_on")
@pytest.mark.parametrize("flavour", FLAVOURS)
def test_a_knowledge_base_on_picked_stores_answers_a_chat_from_them(
    admin, listener, upstream, flavour
):
    endpoint = VectorStores(listener, flavour, [*_stores(), VectorStore("vs_empty", "Empty")])
    picked = {
        "type": "collection",
        "name": "vs_handbook, vs_crew",
        "config": {"vector_store_ids": ["vs_handbook", "vs_crew"]},
    }
    name = f"Harbour stores {uuid.uuid4().hex[:6]}"
    prompt = f"what is the gate code and when does the watch start? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("4471, and at 22:00.", match=reply.answering(prompt)))
    with admin.client() as client:
        empty = client.post(
            SOURCE_CREATE,
            json={
                "name": f"{name} empty",
                "connection": endpoint.form(),
                "source": {"type": "collection", "name": "vs_empty"},
                "test_query": "gate code",
            },
        )
        created = client.post(
            SOURCE_CREATE,
            json={
                "name": name,
                "connection": endpoint.form(),
                "source": picked,
                "test_query": "gate code",
            },
        )
        assert created.status_code == 200, f"creating the knowledge base failed: {created.text}"
        knowledge = created.json()
        connection_id = knowledge["meta"]["external"]["connection_id"]
        try:
            listed = _listed_for(client, connection_id)
            uploaded = client.post(
                "/api/v1/files/?process_in_background=false",
                files={"file": ("extra.txt", b"Boats leave at dawn.", "text/plain")},
            )
            assert uploaded.status_code == 200, uploaded.text
            refused_upload = client.post(
                f"/api/v1/knowledge/{knowledge['id']}/file/add",
                json={"file_id": uploaded.json()["id"]},
            )
            files_after = client.get(f"/api/v1/knowledge/{knowledge['id']}/files")
            embeddings_before = len(upstream.requests_to("/embeddings"))
            _, answer = ask(
                client, prompt, files=[{"type": "collection", "id": knowledge["id"], "name": name}]
            )
            embedded_after = [
                entry.body for entry in upstream.requests_to("/embeddings")[embeddings_before:]
            ]
        finally:
            client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")
            client.delete(f"{CONNECTIONS}/{connection_id}")

    assert empty.status_code == 400, (
        f"a knowledge base was created though its test query found nothing: {empty.text}"
    )
    assert list(listed) == [name], f"the new knowledge base is not listed: {listed}"
    assert knowledge["meta"].get("read_only") is True, knowledge["meta"]
    assert refused_upload.status_code >= 400, "a file was added to a read-only knowledge base"
    assert "extra.txt" not in files_after.text, files_after.text
    assert answer["content"] == "4471, and at 22:00."
    searches = endpoint.searches()
    assert set(searches) >= {"vs_handbook", "vs_crew"} and "vs_archive" not in searches, (
        f"the chat did not search exactly the picked stores: {sorted(searches)}"
    )
    chat_queries = [call.json()["query"] for call in searches["vs_crew"]]
    assert any(prompt in json.dumps(query) for query in chat_queries), (
        f"the store was not sent the question as text: {chat_queries}"
    )
    sent = _sent_to_model(upstream, prompt)
    assert (
        "The harbour gate code is 4471." in sent and "The night watch starts at 22:00." in sent
    ), "the model was not sent the results of both picked stores"
    assert "1234" not in sent, "the model was sent a store that was not picked"
    assert prompt not in json.dumps(embedded_after), (
        "Open WebUI embedded the question itself, though the server runs the search"
    )
    assert _cited_names(answer) == ["gates.md", "watch.md"], answer["sources"]


@pytest.fixture
def listing(admin, listener, make_user):
    """An endpoint whose connection lists its stores, read-shared with one account."""
    endpoint = VectorStores(listener, "openai", _stores())
    reader, stranger = make_user(), make_user()
    form = endpoint.form(store_ids=[], access_grants=[grant("user", reader.id, "read")])
    with admin.client() as client, external_connection(client, form) as connection_id:
        yield endpoint, form, connection_id, reader, stranger


def test_listed_stores_show_as_read_only_knowledge_for_whoever_the_connection_allows(
    admin, listing
):
    endpoint, form, connection_id, reader, stranger = listing
    with admin.client() as client:
        for_admin = _listed_for(client, connection_id)
    with reader.client() as client:
        for_reader = _listed_for(client, connection_id)
        handbook_id = for_reader["Harbour handbook"]["id"]
        opened = client.get(f"/api/v1/knowledge/{handbook_id}")
        writes = {
            "rename": client.post(
                f"/api/v1/knowledge/{handbook_id}/update", json={"name": "Mine", "description": ""}
            ),
            "add a file": client.post(
                f"/api/v1/knowledge/{handbook_id}/file/add", json={"file_id": "any"}
            ),
            "delete": client.delete(f"/api/v1/knowledge/{handbook_id}/delete"),
        }
    with stranger.client() as client:
        for_stranger = _listed_for(client, connection_id)
        stranger_opened = client.get(f"/api/v1/knowledge/{handbook_id}")

    every_store = ["Crew notes", "Harbour handbook", "Old archive"]
    assert sorted(for_admin) == every_store, (
        f"the admin does not see every store: {sorted(for_admin)}"
    )
    assert sorted(for_reader) == every_store, (
        f"the reader does not see every store: {sorted(for_reader)}"
    )
    assert for_reader["Harbour handbook"]["description"] == "Rules for the harbour"
    assert all(item["meta"].get("read_only") for item in for_reader.values()), for_reader
    assert not any(item.get("write_access") for item in for_reader.values()), (
        "a store listed as knowledge reads as writable for the reader"
    )
    assert opened.status_code == 200 and opened.json()["name"] == "Harbour handbook", opened.text
    changed = {
        action: answer.status_code for action, answer in writes.items() if answer.status_code < 400
    }
    assert not changed, f"the reader changed a read-only store knowledge base: {changed}"
    assert for_stranger == {}, (
        f"an account the connection does not allow sees: {sorted(for_stranger)}"
    )
    assert stranger_opened.status_code in (401, 403, 404), stranger_opened.text

    with admin.client() as client:
        narrowed = client.patch(
            f"{CONNECTIONS}/{connection_id}",
            json={
                **form,
                "auth_config": None,
                "config": {**form["config"], "store_ids": ["vs_crew"]},
            },
        )
        assert narrowed.status_code == 200, narrowed.text
    with reader.client() as client:
        after_narrowing = _listed_for(client, connection_id)
    assert sorted(after_narrowing) == ["Crew notes"], (
        f"with one Store ID the reader sees {sorted(after_narrowing)}"
    )


def test_deleting_a_listed_store_in_the_workspace_keeps_the_connection_and_its_stores(
    admin, listing
):
    """The admin's Workspace > Knowledge offers Delete on a listed store. The store stays on the
    endpoint, so it stays listed, the connection with every other store stays, and a model that
    has it attached keeps it.

    Red on dev f80a4f782: the delete removes the whole connection (open-webui/open-webui#35042,
    introduced by c68b8a632)."""
    _endpoint, _form, connection_id, reader, _stranger = listing
    model_id = f"harbour-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        handbook_id = _listed_for(client, connection_id)["Harbour handbook"]["id"]
        attached = [{"type": "collection", "id": handbook_id, "name": "Harbour handbook"}]
        added = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {"knowledge": attached},
                "params": {},
                "access_grants": [],
            },
        )
        assert added.status_code == 200, added.text
        try:
            client.delete(f"/api/v1/knowledge/{handbook_id}/delete")
            connection = client.get(f"{CONNECTIONS}/{connection_id}")
            model = client.get("/api/v1/models/model", params={"id": model_id})
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
    with reader.client() as client:
        still_listed = _listed_for(client, connection_id)

    assert connection.status_code == 200, (
        f"deleting one listed store removed the whole connection: {connection.status_code}"
    )
    assert sorted(still_listed) == ["Crew notes", "Harbour handbook", "Old archive"], (
        f"after the admin deleted one listed store the reader sees {sorted(still_listed)}"
    )
    assert model.status_code == 200, model.text
    assert model.json()["meta"].get("knowledge") == attached, (
        f"the model lost the listed store it had attached: {model.json()['meta'].get('knowledge')}"
    )


@pytest.mark.usefixtures("file_context_on")
def test_a_chat_with_a_listed_store_gets_its_results_only_for_allowed_accounts(listing, upstream):
    endpoint, _form, connection_id, reader, stranger = listing
    with reader.client() as client:
        handbook_id = _listed_for(client, connection_id)["Harbour handbook"]["id"]
    attached = [{"type": "collection", "id": handbook_id, "name": "Harbour handbook"}]
    allowed_prompt = f"what is the gate code? {uuid.uuid4().hex[:6]}"
    stranger_prompt = f"what is the gate code, please? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("4471.", match=reply.answering(allowed_prompt)),
        reply.text("I do not know.", match=reply.answering(stranger_prompt)),
    )
    with reader.client() as client:
        _, answer = ask(client, allowed_prompt, files=attached)
    searched_for_reader = len(endpoint.searches().get("vs_handbook", []))
    with stranger.client() as client:
        ask(client, stranger_prompt, files=attached)

    assert answer["content"] == "4471."
    assert "The harbour gate code is 4471." in _sent_to_model(upstream, allowed_prompt), (
        "a chat with the listed store attached was not sent its results"
    )
    assert _cited_names(answer) == ["gates.md"], answer["sources"]
    assert "4471" not in _sent_to_model(upstream, stranger_prompt), (
        "an account the connection does not allow got the store's results"
    )
    assert len(endpoint.searches().get("vs_handbook", [])) == searched_for_reader, (
        "the store was searched for an account the connection does not allow"
    )


@pytest.mark.slow
def test_listed_stores_follow_the_stores_the_endpoint_holds(admin, listing):
    """Bounded at 90 seconds: the listing may be cached briefly; the docs say it is read live."""
    endpoint, _form, connection_id, _reader, _stranger = listing
    with admin.client() as client:
        before = sorted(_listed_for(client, connection_id))
        archive_id = _listed_for(client, connection_id)["Old archive"]["id"]
        endpoint.remove("vs_archive")
        endpoint.add(
            VectorStore(
                "vs_tides", "Tide tables", "", [search_hit("High tide at 6.", "tides.md", 0.9)]
            )
        )
        deadline = time.monotonic() + 90
        after = before
        while time.monotonic() < deadline:
            after = sorted(_listed_for(client, connection_id))
            if after == ["Crew notes", "Harbour handbook", "Tide tables"]:
                break
            time.sleep(1)
        archive_opened = client.get(f"/api/v1/knowledge/{archive_id}")

    assert before == ["Crew notes", "Harbour handbook", "Old archive"], before
    assert after == ["Crew notes", "Harbour handbook", "Tide tables"], (
        f"the listed knowledge bases did not follow the endpoint within 90 seconds: {after}"
    )
    assert archive_opened.status_code == 404, (
        f"a store removed from the endpoint still opens: {archive_opened.status_code}"
    )
