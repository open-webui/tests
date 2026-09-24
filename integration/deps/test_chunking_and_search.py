"""Dependency smoke: chunking and hybrid search, driven through the retrieval API.

`save_docs_to_vector_db` in `routers/retrieval.py` cuts every processed file with a
langchain-text-splitters splitter chosen by the admin's `TEXT_SPLITTER`: the recursive character
splitter (""), the tiktoken-measured `TokenTextSplitter` ("token") and, before either, the
`MarkdownHeaderTextSplitter` when `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` is on. With
`ENABLE_RAG_HYBRID_SEARCH`, `/api/v1/retrieval/query/doc` ranks the chunks with rank_bm25's
`BM25Okapi` inside langchain-classic's ensemble and compression retrievers. A bump that breaks a
splitter leaves a file as one chunk; one that breaks BM25 fails or misranks the query. The
provider embeds every text as the same vector, so only the keyword ranking can pick a chunk.

Discriminates: passes on dev bbfa876af. A backend copy without the character splitter's
`split_documents` fails only the character test; one without the token splitter's
`split_documents`, the markdown `split_text` and `BM25Okapi` fails only the other four.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
DEAD_PROXY = "http://127.0.0.1:9"

PARAGRAPHS = [
    f"Paragraph {index} of the harbour log records the tides, the ferries and the weather."
    for index in range(12)
]


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


def _upload(client: httpx.Client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    stored = client.get(f"/api/v1/files/{file_id}").json()
    assert stored["data"].get("status") == "completed", stored["data"].get("error")
    return file_id


def _query(client: httpx.Client, file_id: str, query: str, **options) -> list[str]:
    """The chunks `/query/doc` returns for the file, best first."""
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": f"file-{file_id}", "query": query, **options},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()["documents"][0]


def _all_chunks(client: httpx.Client, file_id: str) -> list[str]:
    return _query(client, file_id, "harbour", k=100)


def _tiktoken_loads_offline(encoding_name: str) -> bool:
    """Whether the instance can build the token splitter without downloading its BPE file."""
    import tiktoken

    with pytest.MonkeyPatch.context() as patch:
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            patch.setenv(name, DEAD_PROXY)
        for name in ("NO_PROXY", "no_proxy"):
            patch.delenv(name, raising=False)
        try:
            tiktoken.get_encoding(encoding_name)
        except OSError:  # the download, refused by the dead proxy
            return False
    return True


# ---------------------------------------------------------------- splitters


def test_the_character_splitter_cuts_by_length(retrieval_settings, make_user):
    retrieval_settings(
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", "\n\n".join(PARAGRAPHS)))

    assert len(chunks) > 1, "the document was stored as one chunk"
    assert max(map(len, chunks)) <= 120, [len(chunk) for chunk in chunks]


def test_the_token_splitter_cuts_by_tokens(retrieval_settings, make_user):
    # env-only; the scratch instance inherits this process's environment
    encoding_name = os.environ.get("TIKTOKEN_ENCODING_NAME", "cl100k_base")
    if not _tiktoken_loads_offline(encoding_name):
        pytest.skip(f"the {encoding_name} BPE file is not cached and may not be downloaded")
    retrieval_settings(
        TEXT_SPLITTER="token",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=30,
        CHUNK_OVERLAP=0,
    )
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.txt", "\n\n".join(PARAGRAPHS)))

    assert len(chunks) > 1, "the document was stored as one chunk"
    # 30 tokens of English run far past 30 characters.
    assert max(map(len, chunks)) > 60, [len(chunk) for chunk in chunks]


def test_the_markdown_splitter_cuts_at_headers(retrieval_settings, make_user):
    retrieval_settings(
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True,
        CHUNK_MIN_SIZE_TARGET=0,
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=0,
    )
    sections = ["# Tides", "## Ferries", "## Weather"]
    markdown = "\n\n".join(f"{header}\n\n{PARAGRAPHS[0]}" for header in sections)
    with make_user().client() as client:
        chunks = _all_chunks(client, _upload(client, "log.md", markdown))

    # The whole file fits one character chunk, so only the header split can cut it.
    assert sorted(chunk.splitlines()[0].strip() for chunk in chunks) == sorted(sections), chunks


# ---------------------------------------------------------------- hybrid search


@pytest.mark.parametrize("term", ["zephyrquartz", "obsidianwharf"])
def test_bm25_finds_the_one_chunk_with_the_term(retrieval_settings, make_user, term):
    retrieval_settings(
        ENABLE_RAG_HYBRID_SEARCH=True,
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=False,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=0,
    )
    paragraphs = list(PARAGRAPHS)
    paragraphs[4] += " zephyrquartz"
    paragraphs[9] += " obsidianwharf"
    with make_user().client() as client:
        file_id = _upload(client, "log.txt", "\n\n".join(paragraphs))
        best = _query(client, file_id, term, k=1, k_reranker=1, hybrid_bm25_weight=1)

    assert len(best) == 1 and term in best[0], best
