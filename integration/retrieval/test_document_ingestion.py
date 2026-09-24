"""Document ingestion regressions fixed in v0.11.0, seen through uploads and embedding calls.

* A `.msg` upload failed: the loader used `OutlookMessageLoader`, which needs `extract_msg`, a
  package that cannot be installed next to `beautifulsoup4>=4.14` (PR #26704, `e17db990a`,
  issue #26690). `.msg` files now go through unstructured, which also reads an RFC 822 mail
  saved under that name.
* With the token splitter, a document containing `<|endoftext|>` failed: the tiktoken splitter
  kept its `disallowed_special="all"` default and raised (`33cf3fb`, issue #27094).
* With PaddleOCR-VL as the extraction engine every file type went to the OCR endpoint, which
  rejects Markdown and spreadsheets (PR #27529, `225e23885`, issues #24988/#26759).
* An upload into a knowledge base was linked before its vectors were written and a failed write
  was swallowed, so a file that never got indexed still showed as part of the knowledge base
  (`f5b196c`). Media whose mime type is enabled for extraction was refused unless the engine
  was literally `external` (`db2d24896`).
* Memories, knowledge base descriptions, the external Qdrant, Milvus and pgvector retrievers
  and the knowledge base search tool embedded without `RAG_EMBEDDING_CONTENT_PREFIX` or
  `RAG_EMBEDDING_QUERY_PREFIX`, so E5 and BGE style models matched worse (`c4f5ac6`, PR #26958,
  issue #26353). The scripted provider records every text sent to `/embeddings`.

Twin of unit/retrieval/test_document_ingestion.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the Outlook loader back for
`.msg`, the splitter built without `disallowed_special`, PaddleOCR-VL taking every file type, the
knowledge link written first with its failure swallowed, media extraction gated on `external`,
the prefixes dropped): the upload fails or reaches the wrong service, the duplicate file is
linked, and the embedded texts carry no prefix.
"""

from __future__ import annotations

import base64
import json
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
PREFIX_ENV = {"RAG_EMBEDDING_QUERY_PREFIX": "query: ", "RAG_EMBEDDING_CONTENT_PREFIX": "passage: "}

PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PIXEL_GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

MAIL = (
    b"From: Alice <alice@example.com>\n"
    b"To: Bob <bob@example.com>\n"
    b"Subject: Quarterly numbers\n"
    b"MIME-Version: 1.0\n"
    b'Content-Type: text/plain; charset="utf-8"\n'
    b"\n"
    b"The harbour lighthouse budget was approved.\n"
)


def _upload(client: httpx.Client, name: str, content: bytes, content_type: str, **metadata) -> dict:
    """Upload a file, process it before answering, and return it as stored."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, content, content_type)},
        data={"metadata": json.dumps(metadata)} if metadata else None,
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = client.get(f"/api/v1/files/{uploaded.json()['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.fixture
def retrieval_settings(preserve, admin):
    """`update(**settings)` changes the document settings for this test."""
    preserve(RETRIEVAL_CONFIG)
    client = admin.client()

    def update(**settings) -> None:
        updated = client.post(RETRIEVAL_CONFIG[1], json=settings)
        assert updated.status_code == 200, updated.text

    yield update
    client.close()


# ---------------------------------------------------------------- .msg uploads


def test_a_msg_upload_is_read(retrieval_settings, make_user):
    with make_user().client() as client:
        stored = _upload(client, "mail.msg", MAIL, "application/vnd.ms-outlook")

    assert stored["data"].get("status") == "completed", (
        f"a .msg upload failed (#26690): {stored['data'].get('error')}"
    )
    assert "harbour lighthouse budget" in stored["data"]["content"]


# ---------------------------------------------------------------- special tokens


@pytest.mark.parametrize(
    "text",
    ["intro <|endoftext|> outro", "plain text with no reserved markers"],
    ids=["reserved-marker", "plain"],
)
def test_the_token_splitter_accepts_any_text(retrieval_settings, make_user, text):
    retrieval_settings(TEXT_SPLITTER="token")
    with make_user().client() as client:
        stored = _upload(client, "notes.txt", text.encode(), "text/plain")

    assert stored["data"].get("status") == "completed", (
        f"splitting by tokens failed on {text!r} (#27094): {stored['data'].get('error')}"
    )


# ---------------------------------------------------------------- PaddleOCR-VL routing


@pytest.fixture
def paddle_ocr(retrieval_settings, listener):
    listener.route(
        "POST",
        "/layout-parsing",
        json_answer({"result": {"layoutParsingResults": [{"markdown": {"text": "OCR text"}}]}}),
    )
    retrieval_settings(
        CONTENT_EXTRACTION_ENGINE="paddleocr_vl",
        PADDLEOCR_VL_BASE_URL=listener.base_url,
        PADDLEOCR_VL_TOKEN="token",
    )
    return listener


@pytest.mark.parametrize(
    ("name", "content", "content_type"),
    [
        ("notes.md", b"# Notes\n\nplain markdown", "text/markdown"),
        ("table.csv", b"city,rank\nParis,1\n", "text/csv"),
    ],
)
def test_paddleocr_vl_leaves_non_ocr_files_to_the_local_loaders(
    paddle_ocr, make_user, name, content, content_type
):
    with make_user().client() as client:
        stored = _upload(client, name, content, content_type)

    assert paddle_ocr.requests_to("/layout-parsing") == [], (
        f"{name} was sent to the OCR endpoint, which rejects it (#26759)"
    )
    assert stored["data"].get("status") == "completed", stored["data"].get("error")


def test_paddleocr_vl_still_reads_a_pdf(paddle_ocr, make_user):
    with make_user().client() as client:
        stored = _upload(client, "scan.pdf", b"%PDF-1.4 scanned page", "application/pdf")

    assert len(paddle_ocr.requests_to("/layout-parsing")) == 1
    assert stored["data"]["content"] == "OCR text"


# ---------------------------------------------------------------- knowledge base uploads


def _knowledge_file_ids(client: httpx.Client, knowledge_id: str) -> list[str]:
    listed = client.get(f"/api/v1/knowledge/{knowledge_id}/files")
    assert listed.status_code == 200, listed.text
    return [item["id"] for item in listed.json()["items"]]


@pytest.fixture
def knowledge_base(retrieval_settings, admin):
    with admin.client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": f"kb-{uuid.uuid4().hex[:8]}", "description": ""},
        )
        assert created.status_code == 200, created.text
        yield client, created.json()["id"]
        client.delete(f"/api/v1/knowledge/{created.json()['id']}/delete")


def test_a_knowledge_upload_that_is_never_indexed_is_not_linked(knowledge_base):
    """A second copy of the same text fails the knowledge base's duplicate check."""
    client, knowledge_id = knowledge_base
    text = f"quarterly report {uuid.uuid4().hex}".encode()
    first = _upload(client, "report.txt", text, "text/plain", knowledge_id=knowledge_id)
    duplicate = _upload(client, "copy.txt", text, "text/plain", knowledge_id=knowledge_id)

    linked = _knowledge_file_ids(client, knowledge_id)
    assert duplicate["id"] not in linked, (
        "a file whose knowledge base vectors were never written was linked anyway"
    )
    assert duplicate["data"].get("status") == "failed"
    assert first["id"] in linked


def test_a_knowledge_upload_is_linked_once_indexed(knowledge_base):
    client, knowledge_id = knowledge_base
    stored = _upload(
        client, "notes.txt", b"distinct notes", "text/plain", knowledge_id=knowledge_id
    )

    assert stored["data"].get("status") == "completed", stored["data"].get("error")
    assert stored["id"] in _knowledge_file_ids(client, knowledge_id)


# ---------------------------------------------------------------- media extraction


@pytest.fixture
def docling(retrieval_settings, listener):
    listener.route(
        "POST",
        "/v1/convert/file",
        json_answer({"status": "success", "document": {"md_content": "a red pixel"}}),
    )
    retrieval_settings(
        CONTENT_EXTRACTION_ENGINE="docling",
        DOCLING_SERVER_URL=listener.base_url,
        CONTENT_EXTRACTION_SUPPORTED_MEDIA_MIME_TYPES=["image/png"],
    )
    return listener


def test_an_image_enabled_for_extraction_is_extracted(docling, make_user):
    with make_user().client() as client:
        stored = _upload(client, "pixel.png", PIXEL_PNG, "image/png")

    assert stored["data"].get("status") == "completed", (
        f"an image enabled for extraction was refused by a non-external engine: "
        f"{stored['data'].get('error')}"
    )
    assert stored["data"]["content"] == "a red pixel"


def test_an_image_outside_the_enabled_types_is_still_refused(docling, make_user):
    with make_user().client() as client:
        stored = _upload(client, "pixel.gif", PIXEL_GIF, "image/gif")

    assert stored["data"].get("status") == "failed"
    assert docling.requests_to("/v1/convert/file") == []


def test_a_video_is_still_stored_without_extraction(docling, make_user):
    with make_user().client() as client:
        stored = _upload(client, "clip.mp4", b"\x00\x00\x00\x18ftypmp42\x00\x01\x02", "video/mp4")

    assert stored["data"].get("status") == "completed"
    assert docling.requests_to("/v1/convert/file") == []


# ---------------------------------------------------------------- embedding prefixes


@pytest.fixture
def prefixed(instance_with):
    """An instance embedding with a query and a content prefix, and its admin."""
    launched = instance_with(PREFIX_ENV)
    with launched.client() as client:
        yield launched, client


def _embedded_texts(upstream) -> list[str]:
    texts: list[str] = []
    for request in upstream.requests_to("/embeddings"):
        inputs = request.body["input"]
        texts.extend(inputs if isinstance(inputs, list) else [inputs])
    return texts


def _assert_all_prefixed(upstream, prefix: str) -> None:
    texts = _embedded_texts(upstream)
    assert texts, "nothing was embedded"
    assert all(text.startswith(prefix) for text in texts), (
        f"embedded without the {prefix!r} prefix (#26353): {texts}"
    )


def _add_memory(client: httpx.Client, content: str = "I live by the harbour") -> str:
    added = client.post("/api/v1/memories/add", json={"content": content})
    assert added.status_code == 200, added.text
    return added.json()["id"]


def test_a_new_memory_is_embedded_as_a_passage(prefixed):
    launched, client = prefixed
    _add_memory(client)

    _assert_all_prefixed(launched.upstream, "passage: ")


def test_a_knowledge_base_description_is_embedded_as_a_passage(prefixed):
    launched, client = prefixed
    created = client.post(
        "/api/v1/knowledge/create",
        json={"name": f"kb-{uuid.uuid4().hex[:8]}", "description": "harbour records"},
    )
    assert created.status_code == 200, created.text

    _assert_all_prefixed(launched.upstream, "passage: ")


def _edit_memory(client: httpx.Client, memory_id: str) -> None:
    client.post(f"/api/v1/memories/{memory_id}/update", json={"content": "I moved inland"})


def _add_through_the_batch_route(client: httpx.Client, memory_id: str) -> None:
    client.post(
        "/api/v1/memories/update",
        json={"operations": [{"action": "add", "content": "I keep bees"}]},
    )


def _rebuild_the_vectors(client: httpx.Client, memory_id: str) -> None:
    client.post("/api/v1/memories/reset")


@pytest.mark.parametrize(
    "write", [_edit_memory, _add_through_the_batch_route, _rebuild_the_vectors]
)
def test_every_memory_write_embeds_passages(prefixed, write):
    launched, client = prefixed
    memory_id = _add_memory(client)
    launched.upstream.reset()

    write(client, memory_id)

    _assert_all_prefixed(launched.upstream, "passage: ")


EXTERNAL_SOURCES = {
    "qdrant": ("http://127.0.0.1:9", {"content_field": "payload.text"}),
    "milvus": ("invalid://nowhere", {"content_field": "text", "vector_field": "vector"}),
    "pgvector": (
        "postgresql://reader@127.0.0.1:9/vectors",
        {"content_field": "text", "vector_field": "vector"},
    ),
}


@pytest.mark.parametrize("provider", EXTERNAL_SOURCES)
def test_an_external_knowledge_query_is_embedded_as_a_query(prefixed, provider):
    """Each endpoint fails the search at once; the query is embedded before that."""
    launched, client = prefixed
    endpoint, source_config = EXTERNAL_SOURCES[provider]
    client.post(
        "/api/v1/knowledge/external/source/test",
        json={
            "connection": {"name": "vectors", "provider": provider, "endpoint": endpoint},
            "source": {"name": "docs", "config": source_config},
            "query": "what is open webui",
        },
    )

    _assert_all_prefixed(launched.upstream, "query: ")


def test_the_knowledge_base_search_tool_embeds_a_query(prefixed):
    launched, client = prefixed
    launched.upstream.queue(
        reply.tool_call("query_knowledge_bases", {"query": "harbour records"}),
        reply.text("searched"),
    )

    ask(client, "which knowledge base covers the harbour?")

    _assert_all_prefixed(launched.upstream, "query: ")


def test_a_memory_query_is_still_embedded_as_a_query(prefixed):
    launched, client = prefixed
    _add_memory(client)
    launched.upstream.reset()

    client.post("/api/v1/memories/query", json={"content": "where do I live?"})

    _assert_all_prefixed(launched.upstream, "query: ")
