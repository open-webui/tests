"""Knowledge base lifecycle, file limits and embedding settings.

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
- Processing-status streams pinned the connection pool (ba0c4b393, PR #28183): the file status
  and knowledge pending-files streams took a request-scoped session, which FastAPI releases only
  when the body ends, and these bodies run for hours while a file is stuck. With session sharing
  on, a handful of watchers used up the pool and every other request failed. Here a file stays
  stuck behind an embedding service that never answers, on an instance whose pool holds eleven
  connections, and more watchers than that must leave the instance answering.
- Knowledge base file limits (v0.13.0, 7618df3f5 + 5c4fb61d8, PR #33318):
  `KNOWLEDGE_MAX_FILE_COUNT`, `KNOWLEDGE_MAX_FILE_SIZE` and `KNOWLEDGE_MAX_UPLOAD_COUNT` cap every
  knowledge base, and each base may set its own count and size limit; the stricter one applies.
  A file past a limit is refused with the reason, by every way in (add, batch add, upload into the
  base, the preflight check), and the base keeps exactly the files it had.

Twin of unit/retrieval/test_knowledge_lifecycle.py.

Discriminates: passes on dev bbfa876af; reverting the reindex drop or the `file-{id}` rebuild
fails the reindex case, dropping the grant and folder checks fails both shared-chat cases,
writing every provider block fails all three engines and required form fields fail the partial
block, a reset that deletes no files fails the reset case and an unconditional bearer header
fails both empty-key cases. On dev ef67cc3fa, taking `Depends(get_async_session)` in either
stream handler again (ba0c4b393 reverted) fails its watcher case: the next request waits out
the pool and fails. Every file limit case fails on v0.12.5 (no limits, the extra file is linked),
and on v0.13.0 letting a base's own limit replace the global one fails both stricter-global cases.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.listener import json_answer, listening

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


# --- knowledge base file limits ----------------------------------------------------------------

KNOWLEDGE_LIMITS = {
    "KNOWLEDGE_MAX_UPLOAD_COUNT": "2",
    "KNOWLEDGE_MAX_FILE_COUNT": "3",
    "KNOWLEDGE_MAX_FILE_SIZE": "2",
}
MB = 1024 * 1024


@pytest.fixture(scope="module")
def limited(instance_with):
    """The admin of an instance allowing 2 files per upload, 3 per knowledge base, 2 MB each."""
    return admin_of(instance_with(KNOWLEDGE_LIMITS))


def new_knowledge_base(owner, **limits) -> str:
    body = {"name": f"kb {uuid.uuid4().hex[:6]}", "description": ""}
    if limits:
        body["meta"] = limits
    with owner.client() as client:
        created = client.post("/api/v1/knowledge/create", json=body)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def sized_text(size: int) -> bytes:
    # padded with blanks, so a large file has one short chunk to embed
    words = f"heron colony {uuid.uuid4().hex}".encode()
    return words + b" " * (size - len(words))


def stored_file(owner, size: int) -> str:
    """A processed upload of `size` bytes, ready to join a knowledge base."""
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/?process_in_background=false",
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", sized_text(size), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def add_file(owner, knowledge_id: str, file_id: str) -> httpx.Response:
    with owner.client() as client:
        return client.post(f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id})


def add_batch(owner, knowledge_id: str, file_ids: list[str]) -> httpx.Response:
    with owner.client() as client:
        return client.post(
            f"/api/v1/knowledge/{knowledge_id}/files/batch/add",
            json=[{"file_id": file_id} for file_id in file_ids],
        )


def upload_into(owner, knowledge_id: str, content: bytes) -> httpx.Response:
    """An upload straight into a knowledge base, the way its Add Files button sends one."""
    with owner.client() as client:
        return client.post(
            "/api/v1/files/?process_in_background=false",
            files={"file": (f"{uuid.uuid4().hex[:8]}.txt", content, "text/plain")},
            data={"metadata": json.dumps({"knowledge_id": knowledge_id})},
        )


def linked_files(owner, knowledge_id: str) -> list[str]:
    with owner.client() as client:
        listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
    assert listed.status_code == 200, listed.text
    return sorted(item["id"] for item in listed.json()["items"])


def filled(owner, knowledge_id: str, count: int) -> list[str]:
    file_ids = [upload(owner, f"limit probe {uuid.uuid4().hex}") for _ in range(count)]
    for file_id in file_ids:
        added = add_file(owner, knowledge_id, file_id)
        assert added.status_code == 200, f"a file within the limits was refused: {added.text}"
    assert linked_files(owner, knowledge_id) == sorted(file_ids)
    return sorted(file_ids)


def refused(response: httpx.Response, status: int, reason: str) -> None:
    assert response.status_code == status, (
        f"expected a refusal naming {reason!r}, got HTTP {response.status_code}: {response.text}"
    )
    assert reason in response.json()["detail"], f"the refusal does not say why: {response.text}"


@pytest.mark.slow
def test_a_full_knowledge_base_refuses_one_more_file(limited):
    knowledge_id = new_knowledge_base(limited)
    kept = filled(limited, knowledge_id, 3)

    refused(
        add_file(limited, knowledge_id, upload(limited, "one too many")), 400, "3 files maximum"
    )
    assert linked_files(limited, knowledge_id) == kept, "a refused file joined the knowledge base"


@pytest.mark.slow
def test_uploading_straight_into_a_full_knowledge_base_is_refused(limited):
    knowledge_id = new_knowledge_base(limited)
    for _ in range(3):
        uploaded = upload_into(limited, knowledge_id, f"direct {uuid.uuid4().hex}".encode())
        assert uploaded.status_code == 200, (
            f"an upload within the limit was refused: {uploaded.text}"
        )
    kept = linked_files(limited, knowledge_id)
    assert len(kept) == 3, f"the uploads within the limit were not all linked: {kept}"

    refused(upload_into(limited, knowledge_id, b"one too many"), 400, "3 files maximum")
    assert linked_files(limited, knowledge_id) == kept


@pytest.mark.slow
def test_a_batch_past_the_file_count_is_refused_whole(limited):
    knowledge_id = new_knowledge_base(limited)
    kept = filled(limited, knowledge_id, 2)
    two_more = [upload(limited, f"batch {uuid.uuid4().hex}") for _ in range(2)]

    refused(add_batch(limited, knowledge_id, two_more), 400, "3 files maximum")
    assert linked_files(limited, knowledge_id) == kept, "part of a refused batch was linked"

    added = add_batch(limited, knowledge_id, two_more[:1])
    assert added.status_code == 200, f"a batch that fits was refused: {added.text}"
    assert linked_files(limited, knowledge_id) == sorted([*kept, two_more[0]])


@pytest.mark.slow
def test_more_files_in_one_upload_than_allowed_is_refused(limited):
    knowledge_id = new_knowledge_base(limited)
    three = [upload(limited, f"batch {uuid.uuid4().hex}") for _ in range(3)]

    refused(add_batch(limited, knowledge_id, three), 400, "maximum of 2 file(s) per batch")
    assert linked_files(limited, knowledge_id) == []
    with limited.client() as client:
        preflight = client.post(
            f"/api/v1/knowledge/{knowledge_id}/files/validate", json={"count": 3}
        )
    refused(preflight, 400, "maximum of 2 file(s) per batch")

    added = add_batch(limited, knowledge_id, three[:2])
    assert added.status_code == 200, f"an upload of the allowed size was refused: {added.text}"
    assert linked_files(limited, knowledge_id) == sorted(three[:2])


@pytest.mark.slow
def test_a_file_over_the_size_limit_is_refused(limited):
    knowledge_id = new_knowledge_base(limited)
    too_large = int(2.5 * MB)

    refused(add_file(limited, knowledge_id, stored_file(limited, too_large)), 413, "less than 2 MB")
    refused(upload_into(limited, knowledge_id, sized_text(too_large)), 413, "less than 2 MB")
    assert linked_files(limited, knowledge_id) == [], "a file over the size limit was linked"
    filled(limited, knowledge_id, 1)


@pytest.mark.slow
def test_a_knowledge_bases_own_count_limit_applies_below_the_global_one(limited):
    knowledge_id = new_knowledge_base(limited)
    with limited.client() as client:
        updated = client.post(
            f"/api/v1/knowledge/{knowledge_id}/update",
            json={"name": "capped", "description": "", "meta": {"max_file_count": 1}},
        )
    assert updated.status_code == 200, updated.text
    kept = filled(limited, knowledge_id, 1)

    refused(
        add_file(limited, knowledge_id, upload(limited, "a second file")), 400, "1 files maximum"
    )
    assert linked_files(limited, knowledge_id) == kept


@pytest.mark.slow
def test_a_looser_own_count_limit_does_not_lift_the_global_one(limited):
    knowledge_id = new_knowledge_base(limited, max_file_count=10)
    kept = filled(limited, knowledge_id, 3)

    refused(
        add_file(limited, knowledge_id, upload(limited, "a fourth file")), 400, "3 files maximum"
    )
    assert linked_files(limited, knowledge_id) == kept


@pytest.mark.slow
def test_the_stricter_of_the_own_and_global_size_limits_applies(limited):
    strict = new_knowledge_base(limited, max_file_size=1)
    loose = new_knowledge_base(limited, max_file_size=5)

    refused(add_file(limited, strict, stored_file(limited, int(1.5 * MB))), 413, "less than 1 MB")
    refused(add_file(limited, loose, stored_file(limited, int(2.5 * MB))), 413, "less than 2 MB")
    assert linked_files(limited, strict) == []
    assert linked_files(limited, loose) == []


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


# --- watching a stuck file does not hold the connection pool ----------------------------------

POOL_OF_ELEVEN = {
    "DATABASE_ENABLE_SESSION_SHARING": "true",
    "DATABASE_POOL_SIZE": "1",
    "DATABASE_POOL_MAX_OVERFLOW": "10",
    "DATABASE_POOL_TIMEOUT": "3",
}
WATCHERS = 20
STUCK_TEXT = "a file the embedder never finishes"


@pytest.fixture(scope="module")
def stalled_embedding():
    """An embedding service that holds the stuck file's text until the test lets it go."""
    released = threading.Event()

    def held(request):
        if STUCK_TEXT not in request.body.decode():
            return four_dimensional(request)
        released.wait(timeout=120)
        return json_answer({"error": "released"}, status=500)

    with listening() as service:
        service.route("POST", "/embeddings", held)
        yield service, released
        released.set()


def stuck_embeddings(service) -> list:
    return [call for call in service.requests_to("/embeddings") if STUCK_TEXT in call.body.decode()]


@pytest.fixture
def stuck_file(instance_with, stalled_embedding):
    """(instance, owner, file id, knowledge id): a file still processing into a knowledge base."""
    service, released = stalled_embedding
    released.clear()
    small_pool = instance_with({**POOL_OF_ELEVEN, "RAG_OPENAI_API_BASE_URL": service.base_url})
    owner = admin_of(small_pool)
    knowledge_id = knowledge_base_with(owner)
    stuck_before = len(stuck_embeddings(service))
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            files={"file": ("stuck.txt", STUCK_TEXT.encode(), "text/plain")},
            data={"metadata": json.dumps({"knowledge_id": knowledge_id})},
        )
    assert uploaded.status_code == 200, uploaded.text
    deadline = time.monotonic() + 30
    while len(stuck_embeddings(service)) == stuck_before:
        assert time.monotonic() < deadline, "the upload never reached the embedding service"
        time.sleep(0.2)
    try:
        yield small_pool, owner, uploaded.json()["id"], knowledge_id
    finally:
        released.set()


# pending until processing starts, then processing until it finishes
STILL_PROCESSING = ({"status": "pending"}, {"status": "processing"})


def first_event(stream) -> object:
    line = next(line for line in stream.iter_lines() if line.startswith("data: "))
    return json.loads(line.removeprefix("data: "))


@contextlib.contextmanager
def watching(owner, path: str):
    """More open status streams on `path` than the pool has connections, opened at once.

    Yields the streams the instance answered.
    """
    with contextlib.ExitStack() as watchers, owner.client() as client:
        lock = threading.Lock()

        def watch():
            stream = client.stream("GET", path, params={"stream": "true"}, timeout=15)
            with contextlib.suppress(httpx.HTTPError):
                response = stream.__enter__()
                with lock:
                    watchers.push(stream)
                return response

        with ThreadPoolExecutor(max_workers=WATCHERS) as pool:
            opened = list(pool.map(lambda _: watch(), range(WATCHERS)))
        yield [response for response in opened if response and response.status_code == 200]


def still_answering(owner, file_id: str) -> httpx.Response:
    with owner.client() as client:
        return client.get(f"/api/v1/files/{file_id}", timeout=15)


@pytest.mark.slow
def test_watching_a_stuck_files_status_leaves_the_pool_free(stuck_file):
    _, owner, file_id, _ = stuck_file
    with watching(owner, f"/api/v1/files/{file_id}/process/status") as streams:
        answered = still_answering(owner, file_id)
        events = [first_event(stream) for stream in streams]

    assert len(streams) == WATCHERS, (
        f"only {len(streams)} of {WATCHERS} status watchers were answered: the ones before them "
        "held the connection pool (PR #28183)"
    )
    assert all(event in STILL_PROCESSING for event in events), (
        f"a watcher was told the stuck file had finished: {events}"
    )
    assert answered.status_code == 200, answered.text


@pytest.mark.slow
def test_watching_a_knowledge_bases_pending_files_leaves_the_pool_free(stuck_file):
    _, owner, file_id, knowledge_id = stuck_file
    with watching(owner, f"/api/v1/knowledge/{knowledge_id}/files/pending") as streams:
        answered = still_answering(owner, file_id)
        listed = [[entry["id"] for entry in first_event(stream)] for stream in streams]

    assert len(streams) == WATCHERS, (
        f"only {len(streams)} of {WATCHERS} pending-file watchers were answered: the ones before "
        "them held the connection pool (PR #28183)"
    )
    assert listed == [[file_id]] * WATCHERS, "a watcher was told the stuck file had finished"
    assert answered.status_code == 200, answered.text


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
