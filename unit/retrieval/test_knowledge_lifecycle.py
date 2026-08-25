"""Knowledge-base lifecycle and embedding configuration regressions fixed in v0.11.1.

Six upstream fixes, all in the knowledge/file/retrieval lifecycle:

  Reindexing left stale per-file vectors (2a6e671f54 + 89922cc9d5, #28106).
    `reindex_knowledge_files` rebuilt only the knowledge-base collection. Each file's own
    `file-{id}` collection kept its old chunks, so attaching that file to a chat afterwards
    returned nothing. The fix drops `file-{id}` first, and `process_file` now rebuilds it
    from the stored SQL content while adding the file to the knowledge collection.

  A chat shared with you attached nothing (5cd9a39534, `get_sources_from_items`).
    The 'chat' branch only accepted owner or admin, so a chat shared by grant or through a
    shared folder was silently dropped. The fix consults AccessGrants and the folder ACL.

  Watching file processing pinned a pooled DB connection (ba0c4b393, PR #28183).
    `get_file_process_status` and `get_pending_knowledge_files` took
    `Depends(get_async_session)`, which FastAPI only releases when the SSE body finishes, up
    to two hours later. The fix drops the dependency; each query opens its own session.

  Saving embedding settings blanked the other providers (87d9b7e84e).
    `update_embedding_config` wrote all three provider configs whatever the engine, so the
    address and key of the two you were not editing were overwritten with whatever the form
    carried. The fix writes only the selected provider, and the form fields are optional.

  Emptying a knowledge base left its files behind (363ad352fe, #27988).
    `reset_knowledge_by_id` deleted the knowledge collection only. File rows, storage blobs
    and `file-{id}` collections stayed, unreachable from the UI. The fix deletes all three
    unless the new ENABLE_KNOWLEDGE_FILE_RETENTION flag is set.

  Empty embedding key still sent an Authorization header (97466deea1, PR #28684, #28683).
    Headers were built inline, so an unset key produced `Authorization: Bearer `, which a
    username/password-protected embedding server rejects. `get_json_bearer_headers` omits
    the header when the key is empty.

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix the file collection is never
dropped or rebuilt, shared chats yield no sources, both status endpoints pass a
request-scoped session into their queries, saving one provider blanks the other two, a
knowledge reset deletes no files, and an empty key still sends a bearer header).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.regression


# -----------------------------------------------------------------------------
# Module fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def knowledge_router_module(owui_module):
    """`open_webui.routers.knowledge`."""
    return owui_module("open_webui.routers.knowledge")


@pytest.fixture(scope="session")
def retrieval_router_module(owui_module):
    """`open_webui.routers.retrieval`."""
    return owui_module("open_webui.routers.retrieval")


@pytest.fixture(scope="session")
def files_router_module(owui_module):
    """`open_webui.routers.files`."""
    return owui_module("open_webui.routers.files")


@pytest.fixture(scope="session")
def headers_module(owui_module):
    """`open_webui.utils.headers`."""
    return owui_module("open_webui.utils.headers")


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


def _user(uid: str = "u1", role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=uid, role=role)


def _request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def _file(fid: str = "f1", owner: str = "u1", content: str = "stored text") -> SimpleNamespace:
    return SimpleNamespace(
        id=fid,
        filename=f"{fid}.txt",
        user_id=owner,
        path=f"uploads/{fid}.txt",
        hash="h",
        meta={},
        data={"content": content, "status": "completed"},
    )


def _vector_result(ids, documents=None, metadatas=None) -> SimpleNamespace:
    return SimpleNamespace(
        ids=[ids],
        documents=[documents if documents is not None else []],
        metadatas=[metadatas if metadatas is not None else []],
    )


class _FakeAsyncDB:
    """Stands in for `get_async_db()`'s async context manager."""

    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, *exc):
        return False


def _files_store(file=None) -> MagicMock:
    store = MagicMock()
    store.get_file_by_id = AsyncMock(return_value=file)
    store.get_file_by_id_and_user_id = AsyncMock(return_value=file)
    store.update_file_data_by_id = AsyncMock(return_value=file)
    store.update_file_metadata_by_id = AsyncMock(return_value=file)
    store.update_file_hash_by_id = AsyncMock(return_value=file)
    store.delete_file_by_id = AsyncMock(return_value=True)
    return store


def _vector_client(query_result=None, has_collection=True) -> MagicMock:
    client = MagicMock()
    client.query = AsyncMock(return_value=query_result)
    client.has_collection = AsyncMock(return_value=has_collection)
    client.delete_collection = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    return client


def _deleted_collections(client: MagicMock) -> set[str]:
    return {call.kwargs.get("collection_name") for call in client.delete_collection.await_args_list}


# =============================================================================
# 66 — reindexing must drop the stale per-file collection and rebuild it
# =============================================================================


async def _run_reindex(module, files, vector_client, process_file):
    kb = SimpleNamespace(id="kb-1", user_id="u1", meta={})
    knowledges = MagicMock()
    knowledges.get_knowledge_bases = AsyncMock(return_value=[kb])
    knowledges.get_files_by_id = AsyncMock(return_value=files)

    with (
        patch.object(module, "Knowledges", knowledges),
        patch.object(module, "ASYNC_VECTOR_DB_CLIENT", vector_client),
        patch.object(module, "process_file", process_file),
    ):
        await module.reindex_knowledge_files(
            _request(), user=_user(role="admin"), db=AsyncMock()
        )


@pytest.mark.asyncio
async def test_reindex_drops_stale_per_file_collection(knowledge_router_module) -> None:
    """Regression for open-webui#28106: the file's own collection must be deleted."""
    vector_client = _vector_client()
    process_file = AsyncMock()
    await _run_reindex(knowledge_router_module, [_file("f1")], vector_client, process_file)

    deleted = _deleted_collections(vector_client)
    assert "file-f1" in deleted, (
        "Regression of #28106: stale per-file collection not dropped on reindex; "
        f"deleted={deleted!r}"
    )
    process_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_reindex_still_rebuilds_the_knowledge_collection(knowledge_router_module) -> None:
    """Nearby: the knowledge-base collection is still dropped and every file reprocessed."""
    vector_client = _vector_client()
    process_file = AsyncMock()
    files = [_file("f1"), _file("f2")]
    await _run_reindex(knowledge_router_module, files, vector_client, process_file)

    deleted = _deleted_collections(vector_client)
    assert "kb-1" in deleted
    assert process_file.await_count == 2


@pytest.mark.asyncio
async def test_reindex_skips_delete_when_no_per_file_collection(knowledge_router_module) -> None:
    """Nearby: nothing is deleted for a file that has no vector collection."""
    vector_client = _vector_client(has_collection=False)
    await _run_reindex(knowledge_router_module, [_file("f1")], vector_client, AsyncMock())

    deleted = _deleted_collections(vector_client)
    assert "file-f1" not in deleted


# =============================================================================
# 66 — process_file rebuilds file-{id} when its chunks are gone
# =============================================================================


async def _run_process_file(module, file, query_result):
    """Drive the real process_file with only its I/O boundaries stubbed.

    Returns the save_docs_to_vector_db mock so the caller can inspect which
    collections were written.
    """
    save_docs = MagicMock(return_value=True)
    config = SimpleNamespace(BYPASS_EMBEDDING_AND_RETRIEVAL=False)

    with (
        patch.object(module, "get_retrieval_config", AsyncMock(return_value=config)),
        patch.object(module, "Files", _files_store(file)),
        patch.object(module, "_validate_collection_access", AsyncMock(return_value=None)),
        patch.object(module, "ASYNC_VECTOR_DB_CLIENT", _vector_client(query_result)),
        patch.object(module, "save_docs_to_vector_db", save_docs),
        patch.object(module, "get_async_db", MagicMock(return_value=_FakeAsyncDB())),
        patch.object(module, "publish_event", AsyncMock()),
    ):
        await module.process_file(
            _request(),
            module.ProcessFileForm(file_id=file.id, collection_name="kb-1"),
            user=_user(role="admin"),
            db=AsyncMock(),
        )
    return save_docs


def _saved_collections(save_docs: MagicMock) -> set[str]:
    return {call.kwargs.get("collection_name") for call in save_docs.call_args_list}


@pytest.mark.asyncio
async def test_kb_add_restores_missing_per_file_collection(retrieval_router_module) -> None:
    """Regression for open-webui#28106: with no per-file chunks left, adding the file to a
    knowledge base must also repopulate `file-{id}` from the stored content."""
    save_docs = await _run_process_file(
        retrieval_router_module, _file("f1"), _vector_result(ids=[])
    )
    saved = _saved_collections(save_docs)
    assert saved == {"kb-1", "file-f1"}, (
        f"Regression of #28106: per-file collection not rebuilt; written collections={saved!r}"
    )


@pytest.mark.asyncio
async def test_kb_add_reuses_existing_per_file_chunks(retrieval_router_module) -> None:
    """Nearby: when the per-file chunks exist they are reused and only the knowledge
    collection is written."""
    result = _vector_result(
        ids=["c1"],
        documents=["chunk one"],
        metadatas=[{"file_id": "f1", "name": "f1.txt"}],
    )
    save_docs = await _run_process_file(retrieval_router_module, _file("f1"), result)

    assert _saved_collections(save_docs) == {"kb-1"}
    docs = save_docs.call_args_list[0].kwargs["docs"]
    assert [doc.page_content for doc in docs] == ["chunk one"]


# =============================================================================
# 67 — a chat shared with you must attach
# =============================================================================


def _chat(cid: str = "c1", owner: str = "owner", folder_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        user_id=owner,
        folder_id=folder_id,
        title="Shared chat",
        chat={
            "history": {
                "messages": {"m1": {"id": "m1", "role": "user", "content": "shared secret sauce"}},
                "currentId": "m1",
            }
        },
    )


async def _sources_for_chat(module, chat, user, *, granted=False, folder=None, folder_access=False):
    chats = MagicMock()
    chats.get_chat_by_id = AsyncMock(return_value=chat)
    grants = MagicMock()
    grants.has_access = AsyncMock(return_value=granted)
    folders = MagicMock()
    folders.get_folder_by_id = AsyncMock(return_value=folder)

    with (
        patch.object(module, "Chats", chats),
        patch.object(module, "AccessGrants", grants),
        patch.object(module, "Folders", folders),
        patch.object(module, "has_folder_access", AsyncMock(return_value=folder_access)),
    ):
        return await module.get_sources_from_items(
            request=_request(),
            items=[{"type": "chat", "id": chat.id}],
            queries=["what did we decide"],
            embedding_function=AsyncMock(return_value=[[0.0]]),
            k=3,
            reranking_function=None,
            k_reranker=None,
            r=None,
            hybrid_bm25_weight=0.5,
            hybrid_search=False,
            full_context=False,
            user=user,
        )


@pytest.mark.asyncio
async def test_chat_shared_by_grant_produces_a_source(retrieval_utils_module) -> None:
    """Narrow: a shared_chat read grant makes the attached chat contribute context."""
    sources = await _sources_for_chat(
        retrieval_utils_module, _chat(), _user("reader"), granted=True
    )
    assert sources, "A chat shared by grant attached nothing"
    assert "shared secret sauce" in sources[0]["document"][0]


@pytest.mark.asyncio
async def test_chat_shared_through_folder_produces_a_source(retrieval_utils_module) -> None:
    """Narrow: a chat inside a folder shared with the user must attach too."""
    chat = _chat(folder_id="fold-1")
    folder = SimpleNamespace(id="fold-1", user_id="owner")
    sources = await _sources_for_chat(
        retrieval_utils_module,
        chat,
        _user("reader"),
        granted=False,
        folder=folder,
        folder_access=True,
    )
    assert sources, "A chat in a shared folder attached nothing"
    assert sources[0]["metadata"][0]["file_id"] == "c1"


@pytest.mark.asyncio
async def test_own_chat_still_attaches(retrieval_utils_module) -> None:
    """Nearby: the owner path is untouched."""
    sources = await _sources_for_chat(
        retrieval_utils_module, _chat(owner="me"), _user("me"), granted=False
    )
    assert sources


@pytest.mark.asyncio
async def test_admin_chat_still_attaches(retrieval_utils_module) -> None:
    """Nearby: admins keep reading any chat."""
    sources = await _sources_for_chat(
        retrieval_utils_module, _chat(), _user("admin-1", role="admin"), granted=False
    )
    assert sources


@pytest.mark.asyncio
async def test_unshared_chat_still_attaches_nothing(retrieval_utils_module) -> None:
    """Nearby: the fix must not widen access. No grant, no folder, no source."""
    sources = await _sources_for_chat(
        retrieval_utils_module, _chat(), _user("stranger"), granted=False
    )
    assert sources == []


@pytest.mark.asyncio
async def test_folder_shared_chat_denied_when_folder_access_denied(retrieval_utils_module) -> None:
    """Nearby: a folder the user cannot read grants nothing."""
    chat = _chat(folder_id="fold-1")
    folder = SimpleNamespace(id="fold-1", user_id="owner")
    sources = await _sources_for_chat(
        retrieval_utils_module,
        chat,
        _user("stranger"),
        granted=False,
        folder=folder,
        folder_access=False,
    )
    assert sources == []


# =============================================================================
# 125 — status endpoints must not hold a request-scoped DB session
# =============================================================================


def _session_dependency_params(handler, get_async_session) -> list[str]:
    """Parameter names whose default is `Depends(get_async_session)`."""
    return [
        name
        for name, param in inspect.signature(handler).parameters.items()
        if getattr(param.default, "dependency", None) is get_async_session
    ]


@pytest.mark.asyncio
async def test_file_status_query_opens_its_own_session(files_router_module) -> None:
    """Regression for PR #28183: the handler must not pass a request-scoped session into
    the lookup, or that connection is pinned for the whole two-hour stream."""
    files = _files_store(_file("f1", owner="u1"))
    with (
        patch.object(files_router_module, "Files", files),
        patch.object(files_router_module, "has_access_to_file", AsyncMock(return_value=True)),
    ):
        result = await files_router_module.get_file_process_status(
            id="f1", stream=False, user=_user("u1")
        )

    assert result == {"status": "completed"}
    args, kwargs = files.get_file_by_id.await_args
    assert "db" not in kwargs and len(args) == 1, (
        "Regression of PR #28183: a request-scoped session reached the lookup, "
        f"args={args!r} kwargs={kwargs!r}"
    )


def test_streaming_status_endpoints_declare_no_session_dependency(
    files_router_module, knowledge_router_module
) -> None:
    """Broad: no long-lived SSE endpoint may take a request-scoped session, because
    FastAPI holds it until the response body finishes."""
    get_async_session = files_router_module.get_async_session
    offenders = {
        "files.get_file_process_status": _session_dependency_params(
            files_router_module.get_file_process_status, get_async_session
        ),
        "knowledge.get_pending_knowledge_files": _session_dependency_params(
            knowledge_router_module.get_pending_knowledge_files, get_async_session
        ),
    }
    assert not any(offenders.values()), (
        f"SSE endpoints still pin a pooled connection: {offenders!r}"
    )


def test_short_lived_file_endpoints_keep_their_session_dependency(files_router_module) -> None:
    """Nearby: the fix must not strip the session from ordinary non-streaming handlers."""
    params = _session_dependency_params(
        files_router_module.get_file_data_content_by_id, files_router_module.get_async_session
    )
    assert params == ["db"]


@pytest.mark.asyncio
async def test_file_status_missing_file_is_404(files_router_module) -> None:
    """Nearby: the not-found path is unchanged."""
    with patch.object(files_router_module, "Files", _files_store(None)):
        with pytest.raises(files_router_module.HTTPException) as exc:
            await files_router_module.get_file_process_status(
                id="nope", stream=False, user=_user("u1")
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_file_status_denied_for_other_users_file(files_router_module) -> None:
    """Nearby: the access check still runs without a session."""
    with (
        patch.object(files_router_module, "Files", _files_store(_file("f1", owner="someone"))),
        patch.object(files_router_module, "has_access_to_file", AsyncMock(return_value=False)),
    ):
        with pytest.raises(files_router_module.HTTPException) as exc:
            await files_router_module.get_file_process_status(
                id="f1", stream=False, user=_user("u1")
            )
    assert exc.value.status_code == 404


# =============================================================================
# 126 — saving embedding settings must touch only the selected provider
# =============================================================================


class _FakeRetrievalConfig(SimpleNamespace):
    async def save(self):
        self.saved = True


def _embedding_config() -> _FakeRetrievalConfig:
    return _FakeRetrievalConfig(
        RAG_EMBEDDING_ENGINE="",
        RAG_EMBEDDING_MODEL="old-model",
        RAG_EMBEDDING_BATCH_SIZE=1,
        ENABLE_ASYNC_EMBEDDING=True,
        RAG_EMBEDDING_CONCURRENT_REQUESTS=0,
        RAG_OPENAI_API_BASE_URL="https://openai.example/v1",
        RAG_OPENAI_API_KEY="openai-key",
        RAG_OLLAMA_BASE_URL="http://ollama.example",
        RAG_OLLAMA_API_KEY="ollama-key",
        RAG_AZURE_OPENAI_BASE_URL="https://azure.example",
        RAG_AZURE_OPENAI_API_KEY="azure-key",
        RAG_AZURE_OPENAI_API_VERSION="2024-02-01",
        saved=False,
    )


async def _update_embedding_config(module, engine: str, config):
    """Send what the pre-fix UI sent: every provider block, whatever the engine."""
    form = module.EmbeddingModelUpdateForm(
        RAG_EMBEDDING_ENGINE=engine,
        RAG_EMBEDDING_MODEL="new-model",
        openai_config=module.OpenAIConfigForm(url="https://new.openai", key="new-openai-key"),
        ollama_config=module.OllamaConfigForm(url="", key=""),
        azure_openai_config=module.AzureOpenAIConfigForm(url="", key="", version=""),
    )
    with (
        patch.object(module, "get_retrieval_config", AsyncMock(return_value=config)),
        patch.object(module, "get_ef", MagicMock(return_value=None)),
        patch.object(module, "get_embedding_function", MagicMock(return_value=None)),
    ):
        return await module.update_embedding_config(_request(), form, user=_user(role="admin"))


@pytest.mark.asyncio
async def test_saving_openai_embedding_leaves_ollama_untouched(retrieval_router_module) -> None:
    """Narrow: picking OpenAI must not blank the Ollama address and key."""
    config = _embedding_config()
    await _update_embedding_config(retrieval_router_module, "openai", config)

    assert (config.RAG_OLLAMA_BASE_URL, config.RAG_OLLAMA_API_KEY) == (
        "http://ollama.example",
        "ollama-key",
    ), "Saving the OpenAI embedding config wiped the Ollama address or key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,untouched",
    [
        (
            "openai",
            ["RAG_OLLAMA_BASE_URL", "RAG_AZURE_OPENAI_BASE_URL", "RAG_AZURE_OPENAI_API_VERSION"],
        ),
        ("ollama", ["RAG_OPENAI_API_BASE_URL", "RAG_AZURE_OPENAI_BASE_URL"]),
        ("azure_openai", ["RAG_OPENAI_API_BASE_URL", "RAG_OLLAMA_BASE_URL"]),
    ],
)
async def test_only_the_selected_provider_is_written(
    retrieval_router_module, engine, untouched
) -> None:
    """Broad: whichever engine is selected, the other providers keep their settings."""
    config = _embedding_config()
    before = {name: getattr(config, name) for name in untouched}
    await _update_embedding_config(retrieval_router_module, engine, config)

    after = {name: getattr(config, name) for name in untouched}
    assert after == before, f"Saving engine {engine!r} overwrote other providers: {after!r}"


@pytest.mark.asyncio
async def test_selected_provider_is_written(retrieval_router_module) -> None:
    """Nearby: the provider you actually edited is still saved."""
    config = _embedding_config()
    await _update_embedding_config(retrieval_router_module, "openai", config)

    assert config.RAG_OPENAI_API_BASE_URL == "https://new.openai"
    assert config.RAG_OPENAI_API_KEY == "new-openai-key"
    assert config.RAG_EMBEDDING_MODEL == "new-model"
    assert config.saved is True


def test_provider_config_forms_accept_omitted_fields(retrieval_router_module) -> None:
    """Narrow: the forms must be optional so a client can send only its own provider."""
    for form_name in ("OpenAIConfigForm", "OllamaConfigForm", "AzureOpenAIConfigForm"):
        form = getattr(retrieval_router_module, form_name)()
        assert form.url is None and form.key is None, form_name


# =============================================================================
# 174 — emptying a knowledge base must clear its files
# =============================================================================


async def _reset_knowledge(module, files, user, vector_client, files_store, storage):
    kb = SimpleNamespace(id="kb-1", user_id="u1", meta={})
    knowledges = MagicMock()
    knowledges.get_knowledge_by_id = AsyncMock(return_value=kb)
    knowledges.get_files_by_id = AsyncMock(return_value=files)
    knowledges.reset_knowledge_by_id = AsyncMock(return_value=kb)
    grants = MagicMock()
    grants.has_access = AsyncMock(return_value=True)

    with (
        patch.object(module, "Knowledges", knowledges),
        patch.object(module, "AccessGrants", grants),
        patch.object(module, "ASYNC_VECTOR_DB_CLIENT", vector_client),
        patch.object(module, "Files", files_store),
        patch.object(module, "Storage", storage),
        patch.object(module, "publish_event", AsyncMock()),
    ):
        return await module.reset_knowledge_by_id(
            _request(), id="kb-1", include_directories=False, user=user, db=AsyncMock()
        )


@pytest.mark.asyncio
async def test_reset_deletes_the_backing_files(knowledge_router_module) -> None:
    """Regression for open-webui#27988: file rows, blobs and per-file vectors must go."""
    vector_client = _vector_client()
    files_store = _files_store()
    storage = MagicMock()
    await _reset_knowledge(
        knowledge_router_module, [_file("f1")], _user("u1"), vector_client, files_store, storage
    )

    files_store.delete_file_by_id.assert_awaited_once()
    assert files_store.delete_file_by_id.await_args.args[0] == "f1"
    storage.delete_file.assert_called_once_with("uploads/f1.txt")
    deleted = _deleted_collections(vector_client)
    assert {"kb-1", "file-f1"} <= deleted, (
        f"Regression of #27988: reset left file vectors behind; deleted={deleted!r}"
    )


@pytest.mark.asyncio
async def test_reset_keeps_files_when_retention_is_enabled(knowledge_router_module) -> None:
    """Nearby: with ENABLE_KNOWLEDGE_FILE_RETENTION on, nothing is removed.

    `create=True` because the flag does not exist on the pre-fix ref, where no file is
    deleted either way, so this stays a layer-3 test.
    """
    files_store = _files_store()
    storage = MagicMock()
    flag = "ENABLE_KNOWLEDGE_FILE_RETENTION"
    with patch.object(knowledge_router_module, flag, True, create=True):
        await _reset_knowledge(
            knowledge_router_module,
            [_file("f1")],
            _user("u1"),
            _vector_client(),
            files_store,
            storage,
        )

    files_store.delete_file_by_id.assert_not_awaited()
    storage.delete_file.assert_not_called()


@pytest.mark.asyncio
async def test_reset_leaves_other_users_files_alone(knowledge_router_module) -> None:
    """Nearby: a non-admin only clears files they own."""
    files_store = _files_store()
    storage = MagicMock()
    await _reset_knowledge(
        knowledge_router_module,
        [_file("f1", owner="someone-else")],
        _user("u1"),
        _vector_client(),
        files_store,
        storage,
    )

    files_store.delete_file_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_still_drops_the_knowledge_collection(knowledge_router_module) -> None:
    """Nearby: the original behaviour is intact for a knowledge base with no files."""
    vector_client = _vector_client()
    await _reset_knowledge(
        knowledge_router_module, [], _user("u1"), vector_client, _files_store(), MagicMock()
    )

    vector_client.delete_collection.assert_awaited_once_with(collection_name="kb-1")


# =============================================================================
# 185 — an empty embedding key must not send an Authorization header
# =============================================================================


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.text = ""
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncResponse:
    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeAiohttpSession:
    """Captures the headers of the single post the embedding call makes."""

    def __init__(self, captured, payload):
        self._captured = captured
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, **kwargs):
        self._captured["headers"] = headers
        return _FakeAsyncResponse(self._payload)


_OPENAI_PAYLOAD = {"data": [{"embedding": [0.1, 0.2]}]}
_OLLAMA_PAYLOAD = {"embeddings": [[0.1, 0.2]]}


def _sync_embedding_headers(module, func_name: str, key) -> dict:
    payload = _OPENAI_PAYLOAD if "openai" in func_name else _OLLAMA_PAYLOAD
    post = MagicMock(return_value=_FakeResponse(payload))
    with patch.object(module.requests, "post", post):
        getattr(module, func_name)(model="m", texts=["hello"], url="http://embed.example", key=key)
    return post.call_args.kwargs["headers"]


async def _async_embedding_headers(module, func_name: str, key) -> dict:
    payload = _OPENAI_PAYLOAD if "openai" in func_name else _OLLAMA_PAYLOAD
    captured: dict = {}
    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda **kwargs: _FakeAiohttpSession(captured, payload),
        ClientTimeout=lambda **kwargs: None,
    )
    with patch.object(module, "aiohttp", fake_aiohttp):
        await getattr(module, func_name)(
            model="m", texts=["hello"], url="http://embed.example", key=key
        )
    return captured["headers"]


@pytest.mark.parametrize(
    "func_name", ["generate_openai_batch_embeddings", "generate_ollama_batch_embeddings"]
)
@pytest.mark.parametrize("key", ["", None, "   "])
def test_sync_embeddings_omit_authorization_without_a_key(
    retrieval_utils_module, func_name, key
) -> None:
    """Regression for open-webui#28683: an empty key must send no Authorization header,
    because a password-protected embedding server rejects `Bearer `."""
    headers = _sync_embedding_headers(retrieval_utils_module, func_name, key)
    assert "Authorization" not in headers, (
        f"Regression of #28683: {func_name} sent {headers.get('Authorization')!r} for an empty key"
    )
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "func_name", ["agenerate_openai_batch_embeddings", "agenerate_ollama_batch_embeddings"]
)
async def test_async_embeddings_omit_authorization_without_a_key(
    retrieval_utils_module, func_name
) -> None:
    """Broad: the async paths carry the same rule."""
    headers = await _async_embedding_headers(retrieval_utils_module, func_name, "")
    assert "Authorization" not in headers, (
        f"Regression of #28683: {func_name} sent {headers.get('Authorization')!r} for an empty key"
    )


@pytest.mark.parametrize(
    "func_name", ["generate_openai_batch_embeddings", "generate_ollama_batch_embeddings"]
)
def test_embeddings_still_send_the_key_when_present(retrieval_utils_module, func_name) -> None:
    """Nearby: a real key is still forwarded as a bearer token."""
    headers = _sync_embedding_headers(retrieval_utils_module, func_name, "sk-secret")
    assert headers["Authorization"] == "Bearer sk-secret"


def test_json_bearer_header_helper(headers_module) -> None:
    """Nearby: the helper itself, including the whitespace-only key."""
    assert headers_module.get_json_bearer_headers("") == {"Content-Type": "application/json"}
    assert headers_module.get_json_bearer_headers(None) == {"Content-Type": "application/json"}
    assert headers_module.get_json_bearer_headers("  ") == {"Content-Type": "application/json"}
    assert headers_module.get_json_bearer_headers(" sk-x ") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-x",
    }
