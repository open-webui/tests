"""Dependency contract: the langchain split distributions.

Open WebUI builds retrieval on three langchain distributions and imports each directly; the
umbrella ``langchain`` package (e07e8ed0d) and langchain-community (05484aa05) are gone, BM25 and
the document loaders now live in the backend.

  - ``langchain_core``: ``Document``, the content unit of every loader, splitter and retriever;
    ``BaseRetriever`` with ``CallbackManagerForRetrieverRun`` (the backend's
    ``VectorSearchRetriever`` and ``BM25Retriever`` subclass it, the vector one async);
    ``BaseDocumentCompressor`` with ``Callbacks`` (``RerankCompressor`` overrides
    ``acompress_documents``); ``BaseLoader`` (the PDF, external and web loaders); and
    ``convert_to_openai_function`` (tool specs in ``utils/tools.py``).
  - ``langchain_text_splitters``: ``RecursiveCharacterTextSplitter`` (with a
    ``length_function`` for the "token_transformers" splitter), ``TokenTextSplitter`` (with
    ``disallowed_special``) and ``MarkdownHeaderTextSplitter``, as ``routers/retrieval.py``
    builds them.
  - ``langchain_classic``: ``EnsembleRetriever(retrievers=, weights=, id_key=)`` fusing BM25 and
    vector search, inside ``ContextualCompressionRetriever(base_compressor=, base_retriever=)``
    which the hybrid search awaits with ``ainvoke``.

These distributions version independently and move symbols between releases. This module pins
the import paths and runs each call shape on stand-ins built here, never on open_webui modules,
so a backend refactor cannot break it. Open WebUI's own chunking and hybrid search are covered
over HTTP in integration/deps/test_chunking_and_search.py. Offline: a tiktoken encoding whose BPE
file is not cached skips rather than downloads.

Discriminates: with ``split_documents`` returning its input, the markdown splitter keeping only
the first section, or the compression retriever skipping its compressor (a pytest plugin
patching the installed libraries), the matching tests go red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytestmark = pytest.mark.depcheck

CORE_IMPORT = "langchain_core"
CORE_DIST = "langchain-core"
SPLITTERS_IMPORT = "langchain_text_splitters"
SPLITTERS_DIST = "langchain-text-splitters"
CLASSIC_IMPORT = "langchain_classic"
CLASSIC_DIST = "langchain-classic"

DEAD_PROXY = "http://127.0.0.1:9"

CORE_SYMBOLS = [
    "documents.Document",
    "documents.BaseDocumentCompressor",
    "document_loaders.BaseLoader",
    "retrievers.BaseRetriever",
    "callbacks.CallbackManagerForRetrieverRun",
    "callbacks.Callbacks",
    "utils.function_calling.convert_to_openai_function",
]

SPLITTERS_SYMBOLS = [
    "RecursiveCharacterTextSplitter",
    "MarkdownHeaderTextSplitter",
    "TokenTextSplitter",
]

CLASSIC_SYMBOLS = [
    "retrievers.ContextualCompressionRetriever",
    "retrievers.EnsembleRetriever",
]


def _document_class(depcheck):
    return depcheck.resolve(depcheck.load(CORE_IMPORT), "documents.Document")


def _offline_encoding(depcheck, name: str = "cl100k_base") -> str:
    """The tiktoken encoding name, once its BPE file loads without a download."""
    tiktoken = depcheck.load("tiktoken")
    with pytest.MonkeyPatch.context() as patch:
        for variable in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            patch.setenv(variable, DEAD_PROXY)
        for variable in ("NO_PROXY", "no_proxy"):
            patch.delenv(variable, raising=False)
        try:
            tiktoken.get_encoding(name)
        except OSError:
            pytest.skip(f"the {name} BPE file is not cached and may not be downloaded")
    return name


# --------------------------------------------------------------------------- #
# langchain_core
# --------------------------------------------------------------------------- #
def test_core_symbols_exist(depcheck):
    depcheck.assert_symbols(depcheck.load(CORE_IMPORT), CORE_SYMBOLS)


def test_core_version_reported(depcheck):
    depcheck.load(CORE_IMPORT)
    assert depcheck.dist_version(CORE_DIST) is not None


def test_document_carries_content_and_metadata(depcheck):
    Document = _document_class(depcheck)

    document = Document(page_content="hello world", metadata={"source": "x"})
    bare = Document(page_content="no metadata")

    assert document.page_content == "hello world"
    assert document.metadata == {"source": "x"}
    assert bare.metadata == {}


def test_convert_to_openai_function_describes_a_pydantic_model(depcheck):
    convert = depcheck.resolve(
        depcheck.load(CORE_IMPORT), "utils.function_calling.convert_to_openai_function"
    )
    pydantic = depcheck.load("pydantic")

    class Lookup(pydantic.BaseModel):
        """Look a thing up."""

        x: int = pydantic.Field(description="the x value")

    spec = convert(Lookup)

    assert spec["name"] == "Lookup"
    assert "x" in spec["parameters"]["properties"]


# --------------------------------------------------------------------------- #
# langchain_text_splitters
# --------------------------------------------------------------------------- #
def test_splitters_symbols_exist(depcheck):
    depcheck.assert_symbols(depcheck.load(SPLITTERS_IMPORT), SPLITTERS_SYMBOLS)


def test_splitters_version_reported(depcheck):
    depcheck.load(SPLITTERS_IMPORT)
    assert depcheck.dist_version(SPLITTERS_DIST) is not None


def test_the_recursive_splitter_cuts_documents_by_length(depcheck):
    splitters = depcheck.load(SPLITTERS_IMPORT)
    Document = _document_class(depcheck)
    splitter = splitters.RecursiveCharacterTextSplitter(
        chunk_size=20, chunk_overlap=5, add_start_index=True
    )
    document = Document(page_content="abcdefghij " * 12, metadata={"a": 1})

    chunks = splitter.split_documents([document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 20 for chunk in chunks)
    assert all(chunk.metadata["a"] == 1 and "start_index" in chunk.metadata for chunk in chunks)


def test_the_recursive_splitter_measures_with_a_length_function(depcheck):
    """The "token_transformers" splitter passes a tokenizer's count as `length_function`."""
    splitters = depcheck.load(SPLITTERS_IMPORT)
    Document = _document_class(depcheck)
    splitter = splitters.RecursiveCharacterTextSplitter(
        chunk_size=3,
        chunk_overlap=0,
        length_function=lambda text: len(text.split()),
        add_start_index=True,
    )

    chunks = splitter.split_documents([Document(page_content="one two three four five six")])

    assert [chunk.page_content for chunk in chunks] == ["one two three", "four five six"]


def test_the_markdown_splitter_cuts_at_headers_and_keeps_them(depcheck):
    splitters = depcheck.load(SPLITTERS_IMPORT)
    splitter = splitters.MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
        strip_headers=False,
    )

    sections = splitter.split_text("# Title\nintro text\n## Section\nbody text")

    assert len(sections) == 2
    assert sections[0].metadata.get("Header 1") == "Title"
    assert sections[0].page_content.startswith("# Title")


def test_the_token_splitter_cuts_by_tokens_and_allows_special_tokens(depcheck):
    """`disallowed_special=()` lets a document holding `<|endoftext|>` be split (#27094)."""
    splitters = depcheck.load(SPLITTERS_IMPORT)
    Document = _document_class(depcheck)
    splitter = splitters.TokenTextSplitter(
        encoding_name=_offline_encoding(depcheck),
        chunk_size=5,
        chunk_overlap=0,
        add_start_index=True,
        disallowed_special=(),
    )
    text = "one two three four five <|endoftext|> six seven eight nine ten"

    chunks = splitter.split_documents([Document(page_content=text)])

    assert len(chunks) > 1
    assert "<|endoftext|>" in "".join(chunk.page_content for chunk in chunks)


# --------------------------------------------------------------------------- #
# langchain_core retrievers and compressors inside langchain_classic's composites
# --------------------------------------------------------------------------- #
def _keyword_retrievers(depcheck):
    """Stand-ins shaped like the backend's: a sync BM25-style one and an async vector-style one."""
    core = depcheck.load(CORE_IMPORT)
    BaseRetriever = depcheck.resolve(core, "retrievers.BaseRetriever")
    Document = _document_class(depcheck)
    texts = ["alpha beta", "beta gamma", "gamma delta"]
    docs = [Document(page_content=text, metadata={"hash": f"h{i}"}) for i, text in enumerate(texts)]

    class KeywordRetriever(BaseRetriever):
        docs: list
        scorer: Any  # the backend keeps an untyped vectorizer or embedding function here
        k: int

        def _get_relevant_documents(self, query, *, run_manager):
            ranked = sorted(self.docs, key=lambda doc: -self.scorer(query, doc.page_content))
            return [doc for doc in ranked if self.scorer(query, doc.page_content)][: self.k]

    class AsyncKeywordRetriever(KeywordRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return []

        async def _aget_relevant_documents(self, query, *, run_manager):
            return KeywordRetriever._get_relevant_documents(self, query, run_manager=run_manager)

    def scorer(query: str, text: str) -> int:
        return text.split().count(query)

    return (
        KeywordRetriever(docs=docs, scorer=scorer, k=2),
        AsyncKeywordRetriever(docs=docs, scorer=scorer, k=2),
    )


def _top_one_compressor(depcheck):
    """Shaped like RerankCompressor: async-only, scores into metadata, keeps `top_n`."""
    core = depcheck.load(CORE_IMPORT)
    BaseDocumentCompressor = depcheck.resolve(core, "documents.BaseDocumentCompressor")

    class TopOne(BaseDocumentCompressor):
        top_n: int

        def compress_documents(self, documents, query, callbacks=None):
            return []

        async def acompress_documents(self, documents, query, callbacks=None):
            ranked = sorted(documents, key=lambda doc: doc.page_content)
            for doc in ranked:
                doc.metadata["score"] = 1.0
            return ranked[: self.top_n]

    return TopOne(top_n=1)


def test_a_retriever_subclass_answers_invoke_and_ainvoke(depcheck):
    sync_retriever, async_retriever = _keyword_retrievers(depcheck)

    found = sync_retriever.invoke("beta")
    found_async = asyncio.run(async_retriever.ainvoke("beta"))

    assert [doc.page_content for doc in found] == ["alpha beta", "beta gamma"]
    assert [doc.page_content for doc in found_async] == ["alpha beta", "beta gamma"]


def test_classic_symbols_exist(depcheck):
    depcheck.assert_symbols(depcheck.load(CLASSIC_IMPORT), CLASSIC_SYMBOLS)


def test_classic_version_reported(depcheck):
    depcheck.load(CLASSIC_IMPORT)
    assert depcheck.dist_version(CLASSIC_DIST) is not None


def test_the_ensemble_fuses_retrievers_and_dedupes_on_the_id_key(depcheck):
    retrievers = depcheck.resolve(depcheck.load(CLASSIC_IMPORT), "retrievers")
    sync_retriever, async_retriever = _keyword_retrievers(depcheck)
    ensemble = retrievers.EnsembleRetriever(
        retrievers=[sync_retriever, async_retriever], weights=[0.5, 0.5], id_key="hash"
    )

    fused = asyncio.run(ensemble.ainvoke("gamma"))

    assert sorted(doc.metadata["hash"] for doc in fused) == ["h1", "h2"]


def test_the_compression_retriever_awaits_the_async_compressor(depcheck):
    """The hybrid search: `ContextualCompressionRetriever(...).ainvoke(query)`."""
    retrievers = depcheck.resolve(depcheck.load(CLASSIC_IMPORT), "retrievers")
    sync_retriever, async_retriever = _keyword_retrievers(depcheck)
    ensemble = retrievers.EnsembleRetriever(
        retrievers=[sync_retriever, async_retriever], weights=[0.5, 0.5], id_key="hash"
    )
    compression = retrievers.ContextualCompressionRetriever(
        base_compressor=_top_one_compressor(depcheck), base_retriever=ensemble
    )

    best = asyncio.run(compression.ainvoke("beta"))

    assert [(doc.page_content, doc.metadata["score"]) for doc in best] == [("alpha beta", 1.0)]


def test_the_base_loader_is_subclassed_through_lazy_load(depcheck):
    """`PDFLoader` implements only `lazy_load`; the dispatcher calls `load()`."""
    BaseLoader = depcheck.resolve(depcheck.load(CORE_IMPORT), "document_loaders.BaseLoader")
    Document = _document_class(depcheck)

    class OnePage(BaseLoader):
        def lazy_load(self):
            yield Document(page_content="page one")

    assert [doc.page_content for doc in OnePage().load()] == ["page one"]
