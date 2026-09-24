"""Knowledge base lifecycle and embedding settings regressions fixed in v0.11.1.

- Reindexing left stale per-file vectors (2a6e671f54 + 89922cc9d5, #28106): the reindex rebuilt
  the knowledge base collection only, so each file's own `file-{id}` collection kept its chunks
  from the old embedding model and a query against it failed. The reindex now drops `file-{id}`
  and adding the file back rebuilds it from the stored text.
- A chat shared with you attached nothing (5cd9a39534): the `chat` item only admitted the owner
  or an admin, so a chat shared by grant or through a shared folder added no context.
- Saving embedding settings blanked the other providers (87d9b7e84e): every provider block the
  form carried was written whatever the engine, and each block's fields were required.
- Emptying a knowledge base left its files behind (363ad352fe, #27988): the reset deleted the
  collection only; file rows, blobs and `file-{id}` collections stayed. It deletes the files the
  caller owns now, unless `ENABLE_KNOWLEDGE_FILE_RETENTION` is set.
- An empty embedding key still sent `Authorization: Bearer ` (97466deea1, PR #28684, #28683),
  which a password-protected embedding server rejects.

Twin of unit/retrieval/test_knowledge_lifecycle.py, which keeps the audit that the file status
stream holds no request-scoped database session (PR #28183): no route shows a pinned pool
connection.

Discriminates: passes on dev bbfa876af; reverting the reindex drop or the `file-{id}` rebuild
fails the reindex case, dropping the grant and folder checks fails both shared-chat cases,
writing every provider block fails all three engines and required form fields fail the partial
block, a reset that deletes no files fails the reset case and an unconditional bearer header
fails both empty-key cases.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EMBEDDING_SETTINGS = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
# shares one boot with test_collection_access
ESCAPE_HATCHES = {
    "ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS": "true",
    "BYPASS_RETRIEVAL_ACCESS_CONTROL": "true",
    "ENABLE_KNOWLEDGE_FILE_RETENTION": "true",
}


def upload(actor, text: str) -> str:
    with actor.client() as client:
        uploaded = client.post(
            "/api/v1/files/?process_in_background=false",
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def knowledge_base_with(owner, *file_ids: str) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/knowledge/create", json={"name": "kb", "description": ""})
        assert created.status_code == 200, created.text
        knowledge_id = created.json()["id"]
        for file_id in file_ids:
            added = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
            )
            assert added.status_code == 200, added.text
    return knowledge_id


def file_status(actor, file_id: str) -> int:
    with actor.client() as client:
        return client.get(f"/api/v1/files/{file_id}").status_code


def reset(actor, knowledge_id: str) -> None:
    with actor.client() as client:
        emptied = client.post(f"/api/v1/knowledge/{knowledge_id}/reset")
    assert emptied.status_code == 200, emptied.text


def query(actor, collection_name: str) -> list[str]:
    with actor.client() as client:
        queried = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": collection_name, "query": "heron"},
        )
    assert queried.status_code == 200, f"{collection_name}: {queried.text}"
    return queried.json()["documents"][0]


# --- embedding provider settings ---------------------------------------------------------------

PROVIDER_BLOCKS = {
    "openai": ("openai_config", {"url": "http://127.0.0.1:9/openai", "key": "openai-key"}),
    "ollama": ("ollama_config", {"url": "http://127.0.0.1:9/ollama", "key": "ollama-key"}),
    "azure_openai": (
        "azure_openai_config",
        {"url": "http://127.0.0.1:9/azure", "key": "azure-key", "version": "2024-02-01"},
    ),
}
BLANK_BLOCKS = {
    "openai_config": {"url": "", "key": ""},
    "ollama_config": {"url": "", "key": ""},
    "azure_openai_config": {"url": "", "key": "", "version": ""},
}


def save_embedding(client, engine: str, **blocks) -> dict:
    current = client.get(EMBEDDING_SETTINGS[0]).json()
    saved = client.post(
        EMBEDDING_SETTINGS[1],
        json={
            "RAG_EMBEDDING_ENGINE": engine,
            "RAG_EMBEDDING_MODEL": current["RAG_EMBEDDING_MODEL"],
            **blocks,
        },
    )
    assert saved.status_code == 200, saved.text
    return saved.json()


@pytest.fixture
def embedding_admin(admin, preserve):
    """The admin's embedding settings, with every provider's own settings put back afterwards."""
    preserve(EMBEDDING_SETTINGS)
    with admin.client() as client:
        before = client.get(EMBEDDING_SETTINGS[0]).json()
        yield client
        for engine, (block, _) in PROVIDER_BLOCKS.items():
            save_embedding(client, engine, **{block: before[block]})


@pytest.mark.parametrize("engine", PROVIDER_BLOCKS)
def test_saving_one_embedding_provider_keeps_the_others(embedding_admin, engine):
    for other_engine, (block, values) in PROVIDER_BLOCKS.items():
        save_embedding(embedding_admin, other_engine, **{block: values})
    edited_block, _ = PROVIDER_BLOCKS[engine]
    edited = {"url": "http://127.0.0.1:9/edited", "key": "edited-key", "version": "2025-01-01"}

    # what the settings form sent before the fix: every block, the others blank
    saved = save_embedding(embedding_admin, engine, **{**BLANK_BLOCKS, edited_block: edited})

    for other_engine, (block, values) in PROVIDER_BLOCKS.items():
        if other_engine != engine:
            assert saved[block] == values, f"saving {engine} overwrote the {other_engine} settings"
    assert saved[edited_block]["url"] == edited["url"]


def test_a_provider_block_may_leave_fields_out(embedding_admin):
    saved = save_embedding(embedding_admin, "ollama", ollama_config={"url": "http://127.0.0.1:9"})

    assert saved["ollama_config"] == {"url": "http://127.0.0.1:9", "key": ""}


# --- an empty embedding key --------------------------------------------------------------------


@pytest.fixture
def embedding_servers(upstream, listener):
    """engine: (its URL, the requests it received), for the scripted provider and a local Ollama."""
    listener.route("POST", "/api/embed", json_answer({"embeddings": [[0.1, 0.2, 0.3]]}))
    return {
        "openai": (upstream.base_url, lambda: upstream.requests_to("/embeddings")),
        "ollama": (listener.base_url, lambda: listener.requests_to("/api/embed")),
    }


def embedding_headers(client, engine: str, key: str, embedding_servers) -> dict:
    url, received = embedding_servers[engine]
    save_embedding(client, engine, **{PROVIDER_BLOCKS[engine][0]: {"url": url, "key": key}})
    # fresh text, since an existing collection is not embedded again
    content = f"embed me {uuid.uuid4().hex}"
    stored = client.post(
        "/api/v1/retrieval/process/text", json={"name": "probe", "content": content}
    )
    assert stored.status_code == 200, stored.text
    assert received(), f"nothing reached the {engine} embedding endpoint"
    return {name.lower(): value for name, value in received()[-1].headers.items()}


@pytest.mark.parametrize("engine", ["openai", "ollama"])
def test_an_empty_embedding_key_sends_no_authorization(embedding_admin, embedding_servers, engine):
    headers = embedding_headers(embedding_admin, engine, "", embedding_servers)

    assert "authorization" not in headers, (
        f"{engine} got {headers.get('authorization')!r} for an empty key (#28683)"
    )


@pytest.mark.parametrize("engine", ["openai", "ollama"])
def test_an_embedding_key_is_still_sent(embedding_admin, embedding_servers, engine):
    headers = embedding_headers(embedding_admin, engine, "sk-embed", embedding_servers)

    assert headers["authorization"] == "Bearer sk-embed"


# --- emptying a knowledge base -----------------------------------------------------------------


def test_emptying_a_knowledge_base_deletes_its_files(admin):
    file_id = upload(admin, "heron colony count")
    knowledge_id = knowledge_base_with(admin, file_id)

    reset(admin, knowledge_id)

    assert file_status(admin, file_id) == 404, "the emptied knowledge base left its file (#27988)"
    with admin.client() as client:
        assert client.get(f"/api/v1/knowledge/{knowledge_id}").status_code == 200


def test_a_writer_emptying_a_knowledge_base_keeps_files_it_does_not_own(admin, make_user):
    writer = make_user()
    admins_file = upload(admin, "the admin's own notes")
    knowledge_id = knowledge_base_with(admin, admins_file)
    grant = {"principal_type": "user", "principal_id": writer.id, "permission": "write"}
    with admin.client() as client:
        granted = client.post(
            f"/api/v1/knowledge/{knowledge_id}/access/update", json={"access_grants": [grant]}
        )
    assert granted.status_code == 200, granted.text

    reset(writer, knowledge_id)

    assert file_status(admin, admins_file) == 200, "a writer's reset deleted someone else's file"


@pytest.fixture(scope="module")
def unguarded(instance_with):
    return instance_with(ESCAPE_HATCHES)


@pytest.mark.slow
def test_file_retention_keeps_the_files_of_an_emptied_knowledge_base(unguarded):
    owner = admin_of(unguarded)
    file_id = upload(owner, "kept on purpose")

    reset(owner, knowledge_base_with(owner, file_id))

    assert file_status(owner, file_id) == 200


# --- reindexing --------------------------------------------------------------------------------


def four_dimensional(request):
    vectors = [{"embedding": [0.1, 0.2, 0.3, 0.4]} for _ in request.json()["input"]]
    return json_answer({"object": "list", "data": vectors})


@pytest.mark.slow
def test_reindexing_rebuilds_each_files_own_collection(unguarded, preserve, listener):
    owner = admin_of(unguarded)
    file_id = upload(owner, "heron colony count")
    knowledge_id = knowledge_base_with(owner, file_id)
    preserve(EMBEDDING_SETTINGS, on=unguarded)
    listener.route("POST", "/v1/embeddings", four_dimensional)
    with owner.client() as client:
        save_embedding(
            client, "openai", openai_config={"url": f"{listener.base_url}/v1", "key": ""}
        )
        reindexed = client.post("/api/v1/knowledge/reindex")
    assert reindexed.status_code == 200, reindexed.text

    assert query(owner, f"file-{file_id}") == ["heron colony count"], (
        "the file's own collection kept vectors from the old embedding model (#28106)"
    )
    assert query(owner, knowledge_id) == ["heron colony count"]


# --- file processing status --------------------------------------------------------------------


def processing_status(actor, file_id: str):
    with actor.client() as client:
        return client.get(f"/api/v1/files/{file_id}/process/status")


def test_the_processing_status_is_read_for_its_owner_only(make_user):
    owner, stranger = make_user(), make_user()
    file_id = upload(owner, "status probe")

    assert processing_status(owner, file_id).json() == {"status": "completed"}
    assert processing_status(stranger, file_id).status_code == 404
    assert processing_status(owner, str(uuid.uuid4())).status_code == 404


# --- attaching a shared chat -------------------------------------------------------------------


def chat_with_secret(owner, secret: str) -> str:
    with owner.client() as client:
        turn, _ = ask(client, f"the harbour code is {secret}")
    return turn.chat_id


def share_by_grant(owner, chat_id: str, reader) -> None:
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        assert client.post(f"/api/v1/chats/{chat_id}/share").status_code == 200
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )
    assert granted.status_code == 200, granted.text


def share_through_folder(owner, chat_id: str, reader) -> None:
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"shared {uuid.uuid4().hex[:6]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
        granted = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": [grant]}
        )
    assert granted.status_code == 200, granted.text


def attach_and_ask(reader, chat_id: str, upstream) -> str:
    upstream.queue(reply.text("noted"))
    with reader.client() as client:
        ask(client, "what is the harbour code?", files=[{"type": "chat", "id": chat_id}])
    return str(upstream.chat_requests()[-1]["messages"])


@pytest.mark.parametrize("share", [share_by_grant, share_through_folder], ids=["grant", "folder"])
def test_a_chat_shared_with_a_user_can_be_attached(make_user, upstream, share):
    owner, reader = make_user(), make_user()
    secret = f"TEAL-{uuid.uuid4().hex[:6]}"
    chat_id = chat_with_secret(owner, secret)
    share(owner, chat_id, reader)

    assert secret in attach_and_ask(reader, chat_id, upstream), (
        "a chat shared with the reader added nothing to the model's context"
    )


def test_an_unshared_chat_adds_nothing_and_its_owner_still_attaches_it(make_user, upstream):
    owner, stranger = make_user(), make_user()
    secret = f"TEAL-{uuid.uuid4().hex[:6]}"
    chat_id = chat_with_secret(owner, secret)

    assert secret not in attach_and_ask(stranger, chat_id, upstream)
    assert secret in attach_and_ask(owner, chat_id, upstream)


def test_an_admin_attaches_any_chat(make_user, admin, upstream):
    secret = f"TEAL-{uuid.uuid4().hex[:6]}"
    chat_id = chat_with_secret(make_user(), secret)

    assert secret in attach_and_ask(admin, chat_id, upstream)
