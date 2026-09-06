"""Document-ingestion regressions fixed in v0.11.0.

- Embedding query prefixes (commit c4f5ac6, PR #26958, issue #26353): the
  Qdrant, Milvus and pgvector external retrievers called
  `embedding_function(query)` with no prefix, so a query was embedded as if it
  were a passage and E5/BGE-style models returned worse matches. The memory and
  knowledge-description write paths were missing the content prefix the same way.
- Text recognition package (PR #26851, issues #26646/#26994):
  `rapidocr-onnxruntime==1.4.4` was unresolvable, so the image-text extraction
  dependency could not be installed. Repinned to `rapidocr==3.9.2`.
- .msg uploads (PR #26704, commit e17db990a, issue #26690): the loader imported
  `OutlookMessageLoader` at module level; it needs `extract_msg`, which conflicts
  with `beautifulsoup4<4.14` and is therefore not installed, so every .msg upload
  raised ImportError. Now `UnstructuredEmailLoader` is imported lazily.
- PaddleOCR-VL routing (PR #27529, commit 225e23885, issues #24988/#26759): with
  the engine selected, the dispatch matched every file type, so Word, markdown
  and spreadsheet uploads were sent to an OCR endpoint that rejects them.
- Special tokens (commit 33cf3fb, issue #27094): the tiktoken splitter was built
  without `disallowed_special`, so its default 'all' raised ValueError on any
  document containing `<|endoftext|>`.
- Knowledge-base upload reliability (commit f5b196c): the KB link was written
  before `process_file`, so the upload reported success before the KB vector
  write finished, and media whose mime type is enabled for extraction was
  rejected unless the engine was literally 'external'.
- Milvus (PR #26911, commit f4a6ea930; PR #27521, commit a15e44a5f, issue
  #26978): the ORM-style PyMilvus API (`Collection`, `connections`, `utility`)
  was replaced with `MilvusClient`, and scalar index creation now falls back to
  an explicit INVERTED index and never fails collection creation, which is what
  broke on embedded Milvus Lite.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (pre-fix the three external
retrievers embed the query with no prefix, requirements.txt pins the old OCR
distribution, a .msg upload raises ImportError, PaddleOCR-VL swallows every file
type, token splitting raises on `<|endoftext|>`, the KB link is written before
processing and configured media mime types are rejected, and the Milvus
multitenancy client drives the ORM API).
"""

from __future__ import annotations

import ast
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


class _Probe(BaseException):
    """Stops a production function at a known point. BaseException survives `except Exception`."""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def external_retrieval_module(owui_module):
    """`open_webui.retrieval.external` (the external vector-DB retrievers)."""
    return owui_module("open_webui.retrieval.external")


@pytest.fixture(scope="session")
def loaders_main_module(owui_module):
    """`open_webui.retrieval.loaders.main` (the Loader dispatch)."""
    return owui_module("open_webui.retrieval.loaders.main")


@pytest.fixture(scope="session")
def retrieval_router_module(owui_module):
    """`open_webui.routers.retrieval` (save_docs_to_vector_db)."""
    return owui_module("open_webui.routers.retrieval")


@pytest.fixture(scope="session")
def files_router_module(owui_module):
    """`open_webui.routers.files` (process_uploaded_file)."""
    return owui_module("open_webui.routers.files")


@pytest.fixture(scope="session")
def milvus_mt_module(owui_module):
    """`open_webui.retrieval.vector.dbs.milvus_multitenancy`."""
    pytest.importorskip("pymilvus", reason="pymilvus not installed in this env")
    return owui_module("open_webui.retrieval.vector.dbs.milvus_multitenancy")


@pytest.fixture(scope="session")
def knowledge_model(owui_module):
    knowledge = owui_module("open_webui.models.knowledge")
    return knowledge.KnowledgeModel(
        id="kb-1",
        user_id="u-1",
        name="External KB",
        description="",
        meta={"external": {"source": {"name": "docs", "config": {}}}},
        created_at=0,
        updated_at=0,
    )


def _source(backend: Path, relative: str) -> str:
    return (backend / relative).read_text(encoding="utf-8")


# =============================================================================
# 1. Embedding query prefix on external retrievers (commit c4f5ac6)
# =============================================================================


class _RecordingEmbedder:
    """Records the call, then aborts the retriever before it touches a vector DB."""

    def __init__(self) -> None:
        self.args: tuple = ()
        self.kwargs: dict = {}

    async def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        raise _Probe("embedding call recorded")


@pytest.mark.parametrize(
    "retriever_name", ["_retrieve_qdrant", "_retrieve_milvus", "_retrieve_pgvector"]
)
def test_external_retriever_embeds_query_with_query_prefix(
    external_retrieval_module, knowledge_model, retriever_name
):
    retriever = getattr(external_retrieval_module, retriever_name)
    embedder = _RecordingEmbedder()

    async def _run():
        with pytest.raises(_Probe):
            await retriever(
                {"endpoint": "http://localhost:1", "config": {}},
                {},
                knowledge_model,
                "what is open webui",
                3,
                embedder,
            )

    asyncio.run(asyncio.wait_for(_run(), timeout=15))

    assert embedder.args[0] == "what is open webui"
    assert embedder.kwargs.get("prefix") == external_retrieval_module.RAG_EMBEDDING_QUERY_PREFIX


def _embedding_calls_without_prefix(source: str) -> list[str]:
    """Awaited embedding-function calls in `source` that pass no prefix."""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name not in ("EMBEDDING_FUNCTION", "embedding_function"):
            continue
        if any(keyword.arg == "prefix" for keyword in node.keywords):
            continue
        offenders.append(f"line {node.lineno}: {name}()")
    return offenders


@pytest.mark.parametrize(
    "relative",
    [
        "open_webui/retrieval/external.py",
        "open_webui/routers/memories.py",
        "open_webui/routers/knowledge.py",
    ],
)
def test_every_embedding_call_passes_a_prefix(open_webui_backend, relative):
    offenders = _embedding_calls_without_prefix(_source(open_webui_backend, relative))
    assert offenders == [], f"{relative} embeds without a prefix at {offenders}"


def test_external_retriever_without_embedding_function_is_rejected(
    external_retrieval_module, knowledge_model
):
    async def _run():
        with pytest.raises(RuntimeError, match="Embedding function"):
            await external_retrieval_module._retrieve_qdrant(
                {"config": {}}, {}, knowledge_model, "q", 3, None
            )

    asyncio.run(asyncio.wait_for(_run(), timeout=15))


# =============================================================================
# 2. Text recognition package pin (PR #26851)
# =============================================================================


def _requirement_names(requirements: str) -> dict[str, str]:
    pins = {}
    for raw_line in requirements.splitlines():
        line = raw_line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
        pins[name] = line
    return pins


def test_ocr_dependency_uses_the_installable_rapidocr_distribution(open_webui_backend):
    pins = _requirement_names(_source(open_webui_backend, "requirements.txt"))
    assert "rapidocr-onnxruntime" not in pins, "the unresolvable OCR pin is back"
    assert "rapidocr" in pins, "no OCR distribution pinned"
    assert "==" in pins["rapidocr"], "OCR distribution is not pinned to a version"


def test_requirements_still_pin_the_rest_of_the_image_stack(open_webui_backend):
    pins = _requirement_names(_source(open_webui_backend, "requirements.txt"))
    for package in ("pillow", "opencv-python-headless", "onnxruntime"):
        assert package in pins, f"{package} missing from requirements"


# =============================================================================
# 3. .msg uploads (PR #26704)
# =============================================================================


@pytest.fixture()
def msg_file(tmp_path) -> str:
    path = tmp_path / "mail.msg"
    path.write_bytes(b"\xd0\xcf\x11\xe0not-a-real-msg")
    return str(path)


def test_msg_upload_uses_the_unstructured_email_loader(loaders_main_module, msg_file):
    loader = loaders_main_module.Loader(engine="")._get_loader(
        "mail.msg", "application/vnd.ms-outlook", msg_file
    )
    assert type(loader).__name__ == "UnstructuredEmailLoader"


class _ModuleWithoutEmailLoader(types.ModuleType):
    """`langchain_community.document_loaders` as it looks without `unstructured` installed."""

    def __init__(self, wrapped) -> None:
        super().__init__(wrapped.__name__)
        self._wrapped = wrapped

    def __getattr__(self, name):
        if name == "UnstructuredEmailLoader":
            raise AttributeError(name)
        return getattr(self._wrapped, name)


def test_msg_upload_without_unstructured_reports_a_readable_error(
    monkeypatch, loaders_main_module, msg_file
):
    import langchain_community.document_loaders as document_loaders

    monkeypatch.setitem(
        sys.modules,
        "langchain_community.document_loaders",
        _ModuleWithoutEmailLoader(document_loaders),
    )

    with pytest.raises(ValueError, match=r"requires the 'unstructured' package"):
        loaders_main_module.Loader(engine="")._get_loader(
            "mail.msg", "application/vnd.ms-outlook", msg_file
        )


def test_loaders_module_does_not_import_outlook_message_loader(open_webui_backend):
    source = _source(open_webui_backend, "open_webui/retrieval/loaders/main.py")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level == 0
        for alias in node.names
    }
    assert "OutlookMessageLoader" not in imported


def test_odt_still_uses_the_unstructured_odt_loader(loaders_main_module, tmp_path):
    path = tmp_path / "doc.odt"
    path.write_bytes(b"PK\x03\x04")
    loader = loaders_main_module.Loader(engine="")._get_loader(
        "doc.odt", "application/vnd.oasis.opendocument.text", str(path)
    )
    assert type(loader).__name__ == "UnstructuredODTLoader"


# =============================================================================
# 4. PaddleOCR-VL routing (PR #27529)
# =============================================================================


PADDLE_KWARGS = {
    "PADDLEOCR_VL_BASE_URL": "http://localhost:8080",
    "PADDLEOCR_VL_TOKEN": "token",
}


def _paddle_loader(loaders_main_module, tmp_path, filename: str, content_type: str, **overrides):
    path = tmp_path / filename
    path.write_bytes(b"payload")
    kwargs = {**PADDLE_KWARGS, **overrides}
    loader = loaders_main_module.Loader(engine="paddleocr_vl", **kwargs)
    return loader._get_loader(filename, content_type, str(path))


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("notes.md", "text/markdown"),
        ("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ],
)
def test_paddleocr_vl_does_not_swallow_non_ocr_file_types(
    loaders_main_module, tmp_path, filename, content_type
):
    loader = _paddle_loader(loaders_main_module, tmp_path, filename, content_type)
    assert type(loader).__name__ != "PaddleOCRVLLoader"


@pytest.mark.parametrize(
    "filename,content_type", [("scan.pdf", "application/pdf"), ("scan.png", "image/png")]
)
def test_paddleocr_vl_still_handles_pdf_and_images(
    loaders_main_module, tmp_path, filename, content_type
):
    loader = _paddle_loader(loaders_main_module, tmp_path, filename, content_type)
    assert type(loader).__name__ == "PaddleOCRVLLoader"


def test_paddleocr_vl_without_base_url_falls_back(loaders_main_module, tmp_path):
    loader = _paddle_loader(
        loaders_main_module, tmp_path, "scan.pdf", "application/pdf", PADDLEOCR_VL_BASE_URL=""
    )
    assert type(loader).__name__ != "PaddleOCRVLLoader"


def test_paddleocr_vl_without_token_falls_back(loaders_main_module, tmp_path):
    loader = _paddle_loader(
        loaders_main_module, tmp_path, "scan.pdf", "application/pdf", PADDLEOCR_VL_TOKEN=""
    )
    assert type(loader).__name__ != "PaddleOCRVLLoader"


# =============================================================================
# 5. Documents containing special tokens (commit 33cf3fb)
# =============================================================================


def _token_splitter_config(retrieval_router_module):
    return retrieval_router_module.RetrievalConfig(
        {
            "TEXT_SPLITTER": "token",
            "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
            "TIKTOKEN_ENCODING_NAME": "cl100k_base",
            "CHUNK_SIZE": 64,
            "CHUNK_OVERLAP": 0,
            "CHUNK_MIN_SIZE_TARGET": 0,
            "RAG_EMBEDDING_ENGINE": "",
            "RAG_EMBEDDING_MODEL": "test-model",
        }
    )


class _AbortingVectorClient:
    """Stops save_docs_to_vector_db right after splitting."""

    def has_collection(self, collection_name: str):
        raise _Probe("reached the vector DB")


def _split_document(monkeypatch, retrieval_router_module, text: str):
    pytest.importorskip("tiktoken", reason="tiktoken not installed in this env")
    document = retrieval_router_module.Document(page_content=text, metadata={"name": "doc"})
    monkeypatch.setattr(retrieval_router_module, "get_vector_db_client", _AbortingVectorClient)
    with pytest.raises(_Probe):
        retrieval_router_module.save_docs_to_vector_db(
            SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            [document],
            "collection",
            _token_splitter_config(retrieval_router_module),
        )


def test_token_splitting_accepts_reserved_special_token_markers(
    monkeypatch, retrieval_router_module
):
    _split_document(monkeypatch, retrieval_router_module, "intro <|endoftext|> outro")


def test_token_splitting_still_works_on_plain_text(monkeypatch, retrieval_router_module):
    _split_document(monkeypatch, retrieval_router_module, "plain text with no reserved markers")


def test_token_splitter_is_built_with_disallowed_special(monkeypatch, retrieval_router_module):
    """Pre-fix the splitter took tiktoken's 'all' default, which raises on reserved markers."""
    real_splitter = retrieval_router_module.TokenTextSplitter
    built: list[dict] = []

    def recording_splitter(**kwargs):
        built.append(kwargs)
        return real_splitter(**kwargs)

    monkeypatch.setattr(retrieval_router_module, "TokenTextSplitter", recording_splitter)
    _split_document(monkeypatch, retrieval_router_module, "intro <|endoftext|> outro")

    assert built, "save_docs_to_vector_db never built a TokenTextSplitter"
    assert all("disallowed_special" in kwargs for kwargs in built), (
        f"the token splitter was built without disallowed_special: {built}"
    )
    assert all(kwargs["disallowed_special"] == () for kwargs in built), built


# =============================================================================
# 6. Knowledge-base upload reliability (commit f5b196c)
# =============================================================================


class _CallLog:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, name: str) -> None:
        self.calls.append(name)


def _install_upload_doubles(monkeypatch, files_router_module, log: _CallLog, config_values: dict):
    """Replace only the I/O boundaries process_uploaded_file talks to."""

    async def fake_config_get(key, default=None):
        return config_values.get(key, default)

    async def fake_process_file(request, form, user=None, db=None):
        log.record(f"process_file:{form.collection_name or 'file'}")

    async def fake_update_file_data(file_id, data, db=None):
        log.record(f"status:{data.get('status')}")

    async def fake_add_file_to_knowledge(**kwargs):
        log.record("link_knowledge")
        return SimpleNamespace(id="kf-1")

    async def fake_get_knowledge(id, db=None):
        return SimpleNamespace(id=id, user_id="u-1")

    monkeypatch.setattr(files_router_module, "Config", SimpleNamespace(get=fake_config_get))
    monkeypatch.setattr(files_router_module, "process_file", fake_process_file)
    monkeypatch.setattr(
        files_router_module,
        "Files",
        SimpleNamespace(update_file_data_by_id=fake_update_file_data),
    )
    monkeypatch.setattr(
        files_router_module,
        "Knowledges",
        SimpleNamespace(
            get_knowledge_by_id=fake_get_knowledge,
            add_file_to_knowledge_by_id=fake_add_file_to_knowledge,
        ),
    )
    monkeypatch.setattr(files_router_module, "_cleanup_local_cache", lambda *a, **k: None)
    monkeypatch.setattr(files_router_module, "_is_text_file", lambda *a, **k: False)


def _run_upload(files_router_module, content_type: str, metadata: dict):
    upload = SimpleNamespace(content_type=content_type)
    file_item = SimpleNamespace(id="f-1")
    user = SimpleNamespace(id="u-1", role="user")

    async def _run():
        await files_router_module.process_uploaded_file(
            SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            upload,
            "/tmp/does-not-matter",
            file_item,
            metadata,
            user,
            db=SimpleNamespace(),
        )

    asyncio.run(asyncio.wait_for(_run(), timeout=20))


def test_knowledge_link_is_written_after_the_knowledge_vector_write(
    monkeypatch, files_router_module
):
    log = _CallLog()
    _install_upload_doubles(
        monkeypatch, files_router_module, log, {"audio.stt.supported_content_types": []}
    )

    _run_upload(files_router_module, "text/plain", {"knowledge_id": "kb-1"})

    assert "link_knowledge" in log.calls, f"file was never linked: {log.calls}"
    assert log.calls.index("process_file:kb-1") < log.calls.index("link_knowledge"), log.calls
    assert log.calls.index("status:processing") < log.calls.index("process_file:kb-1"), log.calls


def test_image_with_a_configured_extraction_mime_type_is_extracted(
    monkeypatch, files_router_module
):
    log = _CallLog()
    _install_upload_doubles(
        monkeypatch,
        files_router_module,
        log,
        {
            "audio.stt.supported_content_types": [],
            "rag.content_extraction_engine": "docling",
            "rag.content_extraction.supported_media_mime_types": ["image/png"],
        },
    )

    _run_upload(files_router_module, "image/png", {})

    assert "process_file:file" in log.calls, (
        f"configured image mime type was not extracted: {log.calls}"
    )
    assert "status:failed" not in log.calls, log.calls


def test_image_outside_the_configured_mime_types_is_still_rejected(
    monkeypatch, files_router_module
):
    log = _CallLog()
    _install_upload_doubles(
        monkeypatch,
        files_router_module,
        log,
        {
            "audio.stt.supported_content_types": [],
            "rag.content_extraction_engine": "docling",
            "rag.content_extraction.supported_media_mime_types": ["image/png"],
        },
    )

    _run_upload(files_router_module, "image/gif", {})

    assert "process_file:file" not in log.calls, log.calls
    assert "status:failed" in log.calls, log.calls


def test_video_is_still_stored_without_extraction(monkeypatch, files_router_module):
    log = _CallLog()
    _install_upload_doubles(
        monkeypatch,
        files_router_module,
        log,
        {"audio.stt.supported_content_types": [], "rag.content_extraction_engine": ""},
    )

    _run_upload(files_router_module, "video/mp4", {})

    assert "process_file:file" not in log.calls, log.calls
    assert "status:completed" in log.calls, log.calls


def test_documents_are_processed_normally(monkeypatch, files_router_module):
    log = _CallLog()
    _install_upload_doubles(
        monkeypatch, files_router_module, log, {"audio.stt.supported_content_types": []}
    )

    _run_upload(files_router_module, "application/pdf", {})

    assert log.calls == ["process_file:file"], log.calls


# =============================================================================
# 7+8. Milvus (PR #26911, PR #27521)
# =============================================================================


class _FakeIndexParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeSchema:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fields: list[dict] = []

    def add_field(self, **kwargs) -> None:
        self.fields.append(kwargs)


class _FakeMilvusClient:
    """Stands in for pymilvus.MilvusClient; records the calls the wrapper makes."""

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict]] = []

    def create_schema(self, **kwargs):
        self.calls.append(("create_schema", kwargs))
        return _FakeSchema(**kwargs)

    def prepare_index_params(self, **kwargs):
        self.calls.append(("prepare_index_params", kwargs))
        return _FakeIndexParams(**kwargs)

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))

    def has_collection(self, *args, **kwargs):
        self.calls.append(("has_collection", kwargs))
        return False


class _FakeOrmCollection:
    """Records any use of the deprecated ORM API."""

    constructed: list[str] = []

    def __init__(self, name, *args, **kwargs) -> None:
        _FakeOrmCollection.constructed.append(name)

    def create_index(self, *args, **kwargs) -> None:
        pass

    def load(self) -> None:
        pass


@pytest.fixture()
def milvus_client(monkeypatch, milvus_mt_module):
    """A multitenancy client whose pymilvus boundary is faked, on either ref shape."""
    _FakeOrmCollection.constructed = []
    fake_client = _FakeMilvusClient()

    monkeypatch.setattr(milvus_mt_module, "Client", lambda **kwargs: fake_client, raising=False)
    monkeypatch.setattr(milvus_mt_module, "Collection", _FakeOrmCollection, raising=False)
    monkeypatch.setattr(
        milvus_mt_module, "CollectionSchema", lambda *a, **k: SimpleNamespace(), raising=False
    )
    monkeypatch.setattr(
        milvus_mt_module, "FieldSchema", lambda **k: SimpleNamespace(**k), raising=False
    )
    monkeypatch.setattr(
        milvus_mt_module,
        "connections",
        SimpleNamespace(connect=lambda **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        milvus_mt_module,
        "utility",
        SimpleNamespace(has_collection=lambda name: False, drop_collection=lambda name: None),
        raising=False,
    )

    client = milvus_mt_module.MilvusClient()
    return client, fake_client


def test_collection_creation_drives_the_milvus_client_api(milvus_client, milvus_mt_module):
    client, fake_client = milvus_client

    client._create_shared_collection("owui_knowledge", 4)

    called = [name for name, _ in fake_client.calls]
    assert "create_collection" in called, (
        f"collection was not created through MilvusClient: {called}"
    )
    assert _FakeOrmCollection.constructed == [], "the deprecated ORM Collection API is still used"


def test_resource_id_scalar_index_is_created_and_never_fails_creation(
    milvus_client, milvus_mt_module
):
    client, fake_client = milvus_client
    resource_field = milvus_mt_module.RESOURCE_ID_FIELD

    client._create_shared_collection("owui_knowledge", 4)

    prepared = [
        kwargs.get("field_name")
        for name, kwargs in fake_client.calls
        if name == "prepare_index_params"
    ]
    assert resource_field in prepared, (
        f"no scalar index prepared for {resource_field}: {fake_client.calls}"
    )
    index_calls = [kwargs for name, kwargs in fake_client.calls if name == "create_index"]
    assert len(index_calls) >= 2, (
        f"scalar index was not created through MilvusClient: {fake_client.calls}"
    )


def test_scalar_index_failure_falls_back_to_an_inverted_index(
    monkeypatch, milvus_client, milvus_mt_module
):
    from pymilvus.exceptions import MilvusException

    client, fake_client = milvus_client
    resource_field = milvus_mt_module.RESOURCE_ID_FIELD

    def failing_create_index(**kwargs):
        fake_client.calls.append(("create_index", kwargs))
        params = kwargs.get("index_params")
        prepared = getattr(params, "kwargs", {})
        if prepared.get("field_name") == resource_field and not prepared.get("index_type"):
            raise MilvusException(message="index type not supported")

    monkeypatch.setattr(fake_client, "create_index", failing_create_index)

    client._create_shared_collection("owui_knowledge", 4)

    index_types = [
        getattr(kwargs.get("index_params"), "kwargs", {}).get("index_type")
        for name, kwargs in fake_client.calls
        if name == "create_index"
    ]
    assert "INVERTED" in index_types, f"no INVERTED fallback for the scalar index: {index_types}"


@pytest.mark.parametrize(
    "relative",
    [
        "open_webui/retrieval/vector/dbs/milvus.py",
        "open_webui/retrieval/vector/dbs/milvus_multitenancy.py",
    ],
)
def test_milvus_modules_do_not_import_the_deprecated_orm_api(open_webui_backend, relative):
    source = _source(open_webui_backend, relative)
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pymilvus")
        for alias in node.names
    }
    deprecated = imported & {
        "Collection",
        "CollectionSchema",
        "FieldSchema",
        "connections",
        "utility",
    }
    assert deprecated == set(), (
        f"{relative} still imports the deprecated ORM API: {sorted(deprecated)}"
    )


def test_collection_name_mapping_is_unchanged(milvus_client):
    client, _ = milvus_client
    assert client._get_collection_and_resource_id("user-memory-u1")[0] == client.MEMORY_COLLECTION
    assert client._get_collection_and_resource_id("file-f1")[0] == client.FILE_COLLECTION
    assert (
        client._get_collection_and_resource_id("web-search-w1")[0] == client.WEB_SEARCH_COLLECTION
    )
    assert (
        client._get_collection_and_resource_id("some-knowledge")[0] == client.KNOWLEDGE_COLLECTION
    )
