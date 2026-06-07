"""Dependency contract: langchain (and its split ecosystem distributions).

`langchain` is the umbrella RAG framework Open WebUI builds retrieval on. In
modern langchain the surface is split across several PyPI distributions, and
Open WebUI imports from each of them:

  - `langchain`            (dist ``langchain``)            — umbrella package;
        depended on, but the backend imports its building blocks from the
        split distributions below rather than from the ``langchain.*``
        namespace directly. Pinned here as an import + version sanity check.
  - `langchain_core`       (dist ``langchain-core``)       — ``Document`` (the
        unit of content threaded through every loader/splitter/retriever),
        ``BaseRetriever`` / ``CallbackManagerForRetrieverRun`` / ``Callbacks``
        (subclassed by ``VectorSearchRetriever``), ``BaseDocumentCompressor``
        (subclassed by ``RerankCompressor``), ``BaseLoader``, and
        ``convert_to_openai_function`` (tool-spec generation).
  - `langchain_community`  (dist ``langchain-community``)  — ``BM25Retriever``
        (lexical half of hybrid search) and the document loaders
        (``WebBaseLoader``, ``PlaywrightURLLoader``, ``PyPDFLoader``, the
        Unstructured* family, etc.).
  - `langchain_text_splitters` (dist ``langchain-text-splitters``) — the
        chunkers (``RecursiveCharacterTextSplitter``,
        ``MarkdownHeaderTextSplitter``, ``TokenTextSplitter``,
        ``CharacterTextSplitter``) that turn ingested docs into RAG chunks.
  - `langchain_classic`    (dist ``langchain-classic``)    — the composite
        retrievers (``EnsembleRetriever``, ``ContextualCompressionRetriever``)
        that wire BM25 + vector + rerank together.

These distributions version *independently* and routinely relocate symbols
between releases (the whole "core / community / classic" split is exactly that
churn). This module pins the exact import paths and offline behaviours the
backend relies on, so a bump that moved or renamed any of them fails loudly
here instead of as a runtime ImportError/AttributeError deep in an ingest or
query path. Everything below is fully offline: no URL fetches, no model
downloads, no transcript APIs.

Pattern: symbol-existence checks (per distribution) + offline behavioural
contracts. Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck


# --------------------------------------------------------------------------- #
# Import-name -> distribution-name map (these differ, and the dist names are
# what the bump tooling pins in requirements.txt).
# --------------------------------------------------------------------------- #
LANGCHAIN_IMPORT = "langchain"
LANGCHAIN_DIST = "langchain"

CORE_IMPORT = "langchain_core"
CORE_DIST = "langchain-core"

COMMUNITY_IMPORT = "langchain_community"
COMMUNITY_DIST = "langchain-community"

SPLITTERS_IMPORT = "langchain_text_splitters"
SPLITTERS_DIST = "langchain-text-splitters"

CLASSIC_IMPORT = "langchain_classic"
CLASSIC_DIST = "langchain-classic"


# Symbols the Open WebUI backend resolves from each distribution. Import paths
# move between langchain versions, so each dotted path is a contract.
CORE_SYMBOLS = [
    # retrieval/utils.py, routers/retrieval.py, every loader: the content unit.
    "documents.Document",
    # retrieval/utils.py RerankCompressor base class.
    "documents.BaseDocumentCompressor",
    # retrieval/loaders/{external_web,external_document,tavily}.py base class.
    "document_loaders.BaseLoader",
    # retrieval/utils.py VectorSearchRetriever base + run-manager type.
    "retrievers.BaseRetriever",
    "callbacks.CallbackManagerForRetrieverRun",
    # retrieval/utils.py RerankCompressor.compress_documents callbacks arg type.
    "callbacks.Callbacks",
    # utils/tools.py: pydantic model -> OpenAI function spec.
    "utils.function_calling.convert_to_openai_function",
]

COMMUNITY_SYMBOLS = [
    # retrieval/utils.py: lexical retriever for hybrid search.
    "retrievers.BM25Retriever",
    # retrieval/web/utils.py: web page loaders.
    "document_loaders.WebBaseLoader",
    "document_loaders.PlaywrightURLLoader",
    "document_loaders.base.BaseLoader",
    # retrieval/loaders/main.py: file loaders imported at module top.
    "document_loaders.YoutubeLoader",
    "document_loaders.AzureAIDocumentIntelligenceLoader",
    "document_loaders.BSHTMLLoader",
    "document_loaders.CSVLoader",
    "document_loaders.Docx2txtLoader",
    "document_loaders.OutlookMessageLoader",
    "document_loaders.PyPDFLoader",
    "document_loaders.TextLoader",
    # retrieval/loaders/main.py: lazily-imported Unstructured* loaders.
    "document_loaders.UnstructuredRSTLoader",
    "document_loaders.UnstructuredXMLLoader",
    "document_loaders.UnstructuredEPubLoader",
    "document_loaders.UnstructuredWordDocumentLoader",
    "document_loaders.UnstructuredExcelLoader",
    "document_loaders.UnstructuredPowerPointLoader",
    "document_loaders.UnstructuredODTLoader",
]

SPLITTERS_SYMBOLS = [
    # routers/retrieval.py imports the first three at module top; CharacterText
    # is the documented "character" fallback splitter family.
    "RecursiveCharacterTextSplitter",
    "MarkdownHeaderTextSplitter",
    "TokenTextSplitter",
    "CharacterTextSplitter",
]

CLASSIC_SYMBOLS = [
    # retrieval/utils.py: composite retrievers for hybrid + rerank, imported
    # from `langchain_classic.retrievers`.
    "retrievers.ContextualCompressionRetriever",
    "retrievers.EnsembleRetriever",
]


# --------------------------------------------------------------------------- #
# langchain (umbrella distribution)
# --------------------------------------------------------------------------- #
def test_langchain_import(depcheck):
    """The umbrella `langchain` distribution must remain importable; it's a
    declared backend dependency even though the load-bearing symbols come from
    the split distributions tested below."""
    mod = depcheck.load(LANGCHAIN_IMPORT)
    assert mod.__name__ == "langchain"


def test_langchain_version_reported(depcheck):
    """Installed `langchain` distribution version is resolvable, so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(LANGCHAIN_DIST) is not None


# --------------------------------------------------------------------------- #
# langchain_core — Document, base classes, function-calling
# --------------------------------------------------------------------------- #
def test_core_import(depcheck):
    mod = depcheck.load(CORE_IMPORT)
    assert mod.__name__ == "langchain_core"


def test_core_symbols_exist(depcheck):
    """Every langchain_core symbol the backend imports must still resolve at its
    current dotted path."""
    mod = depcheck.load(CORE_IMPORT)
    depcheck.assert_symbols(mod, CORE_SYMBOLS)


def test_core_version_reported(depcheck):
    assert depcheck.dist_version(CORE_DIST) is not None


def test_document_construction_contract(depcheck):
    """`Document(page_content=..., metadata=...)` is the constructor shape the
    backend uses everywhere (loaders build them, splitters re-emit them,
    retrievers return them). page_content/metadata must round-trip; metadata
    must default to an empty dict so `doc.metadata.get(...)` is always safe."""
    docs = depcheck.resolve(depcheck.load(CORE_IMPORT), "documents")
    Document = docs.Document

    d = Document(page_content="hello world", metadata={"source": "x", "hash": "h"})
    assert d.page_content == "hello world"
    assert d.metadata == {"source": "x", "hash": "h"}

    # routers/retrieval.py and the loaders read .metadata.get(...) on docs that
    # may have been built without explicit metadata.
    d2 = Document(page_content="no meta")
    assert d2.metadata == {}
    assert d2.metadata.get("score") is None


def test_base_retriever_subclass_contract(depcheck):
    """retrieval/utils.py's VectorSearchRetriever subclasses BaseRetriever and
    implements `_get_relevant_documents(self, query, *, run_manager)`; calling
    code invokes it via `.invoke(query)` / `.ainvoke(query)`. Pin that a minimal
    subclass with exactly that hook is usable through the public invoke API."""
    core = depcheck.load(CORE_IMPORT)
    BaseRetriever = depcheck.resolve(core, "retrievers").BaseRetriever
    Document = depcheck.resolve(core, "documents").Document

    # The keyword-only run_manager hook must remain the override point.
    sig = inspect.signature(BaseRetriever._get_relevant_documents)
    assert "run_manager" in sig.parameters

    class _Probe(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):  # noqa: ANN001
            return [Document(page_content=query, metadata={"score": 1.0})]

    out = _Probe().invoke("ping")
    assert isinstance(out, list) and len(out) == 1
    assert out[0].page_content == "ping"
    assert out[0].metadata.get("score") == 1.0


def test_base_document_compressor_subclass_contract(depcheck):
    """retrieval/utils.py's RerankCompressor subclasses BaseDocumentCompressor
    and overrides (a)compress_documents(documents, query, callbacks=None).
    Confirm it's subclassable with pydantic-style extra fields and that the
    method signature still accepts those positional args."""
    core = depcheck.load(CORE_IMPORT)
    docs = depcheck.resolve(core, "documents")
    BaseDocumentCompressor = docs.BaseDocumentCompressor
    Document = docs.Document

    assert isinstance(BaseDocumentCompressor, type)
    params = inspect.signature(BaseDocumentCompressor.compress_documents).parameters
    for name in ("documents", "query"):
        assert name in params, f"compress_documents lost the {name!r} parameter"

    class _Probe(BaseDocumentCompressor):
        top_n: int = 1

        def compress_documents(self, documents, query, callbacks=None):  # noqa: ANN001
            return list(documents)[: self.top_n]

    out = _Probe(top_n=1).compress_documents(
        [Document(page_content="a"), Document(page_content="b")], "q"
    )
    assert len(out) == 1 and out[0].page_content == "a"


def test_convert_to_openai_function_contract(depcheck):
    """utils/tools.py turns a pydantic args model into an OpenAI function spec
    via convert_to_openai_function. The result must carry name/description/
    parameters so it can be handed to a model's tool list."""
    core = depcheck.load(CORE_IMPORT)
    convert = depcheck.resolve(core, "utils.function_calling.convert_to_openai_function")
    assert callable(convert)

    pydantic = depcheck.try_load("pydantic")
    if pydantic is None:
        pytest.skip("pydantic not importable in this env")

    class _Args(pydantic.BaseModel):
        """Do a thing."""

        x: int = pydantic.Field(description="the x value")

    spec = convert(_Args)
    assert spec["name"] == "_Args"
    assert "parameters" in spec
    # the parameter schema must surface our field so the model can fill it.
    assert "x" in spec["parameters"].get("properties", {})


# --------------------------------------------------------------------------- #
# langchain_text_splitters — chunkers
# --------------------------------------------------------------------------- #
def test_splitters_import(depcheck):
    mod = depcheck.load(SPLITTERS_IMPORT)
    assert mod.__name__ == "langchain_text_splitters"


def test_splitters_symbols_exist(depcheck):
    mod = depcheck.load(SPLITTERS_IMPORT)
    depcheck.assert_symbols(mod, SPLITTERS_SYMBOLS)


def test_splitters_version_reported(depcheck):
    assert depcheck.dist_version(SPLITTERS_DIST) is not None


def test_recursive_character_splitter_contract(depcheck):
    """routers/retrieval.py constructs RecursiveCharacterTextSplitter(
    chunk_size=, chunk_overlap=, add_start_index=True) then calls
    split_documents(docs). Contract: long input splits into >1 chunk, each
    chunk is a Document, original metadata is preserved, and add_start_index
    injects a `start_index` into chunk metadata."""
    splitters = depcheck.load(SPLITTERS_IMPORT)
    Document = depcheck.resolve(depcheck.load(CORE_IMPORT), "documents").Document
    RCTS = splitters.RecursiveCharacterTextSplitter

    sp = RCTS(chunk_size=20, chunk_overlap=5, add_start_index=True)
    src = "abcdefghij " * 12  # 132 chars, far larger than chunk_size
    chunks = sp.split_documents([Document(page_content=src, metadata={"source": "s"})])

    assert len(chunks) > 1, "expected the oversized doc to split into chunks"
    assert all(type(c).__name__ == "Document" for c in chunks)
    # original metadata threaded through + start_index added.
    assert chunks[0].metadata.get("source") == "s"
    assert "start_index" in chunks[0].metadata
    # split_text returns the raw string chunks the backend also relies on.
    text_chunks = sp.split_text(src)
    assert len(text_chunks) > 1 and all(isinstance(t, str) for t in text_chunks)


def test_markdown_header_splitter_contract(depcheck):
    """routers/retrieval.py uses MarkdownHeaderTextSplitter(headers_to_split_on=
    [('#','Header 1'), ...], strip_headers=False).split_text(text). Contract:
    splitting a multi-section markdown doc yields >1 Document, each carries the
    matched header(s) in metadata, and strip_headers=False keeps the '#' marker
    in page_content."""
    splitters = depcheck.load(SPLITTERS_IMPORT)
    MHTS = splitters.MarkdownHeaderTextSplitter

    md = MHTS(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
        strip_headers=False,
    )
    text = "# Title\nintro paragraph text\n## Section\nbody under the section"
    out = md.split_text(text)

    assert len(out) > 1, "expected the headed markdown to split per section"
    assert all(type(c).__name__ == "Document" for c in out)
    # the H1 must surface in the first chunk's metadata under our key.
    assert out[0].metadata.get("Header 1") == "Title"
    # strip_headers=False keeps the literal header line in the content.
    assert "#" in out[0].page_content


def test_token_text_splitter_contract(depcheck):
    """routers/retrieval.py uses TokenTextSplitter(encoding_name=, chunk_size=,
    chunk_overlap=, add_start_index=True).split_documents(docs) for the 'token'
    splitter mode (tiktoken-backed). Contract: a doc longer than chunk_size in
    tokens splits into >1 Document offline."""
    tiktoken = depcheck.try_load("tiktoken")
    if tiktoken is None:
        pytest.skip("tiktoken not importable in this env")

    splitters = depcheck.load(SPLITTERS_IMPORT)
    Document = depcheck.resolve(depcheck.load(CORE_IMPORT), "documents").Document
    TTS = splitters.TokenTextSplitter

    ts = TTS(encoding_name="cl100k_base", chunk_size=5, chunk_overlap=0)
    src = "one two three four five six seven eight nine ten eleven twelve"
    chunks = ts.split_documents([Document(page_content=src, metadata={})])
    assert len(chunks) > 1
    assert all(type(c).__name__ == "Document" for c in chunks)


def test_character_text_splitter_contract(depcheck):
    """CharacterTextSplitter is the base "character" splitter family; pin that it
    constructs with chunk_size/chunk_overlap and splits on its separator."""
    splitters = depcheck.load(SPLITTERS_IMPORT)
    CTS = splitters.CharacterTextSplitter

    cts = CTS(chunk_size=10, chunk_overlap=0)  # default separator "\n\n"
    out = cts.split_text("para one body\n\npara two body\n\npara three body")
    assert len(out) > 1 and all(isinstance(t, str) for t in out)


# --------------------------------------------------------------------------- #
# langchain_community — BM25 retriever + loaders
# --------------------------------------------------------------------------- #
def test_community_import(depcheck):
    mod = depcheck.load(COMMUNITY_IMPORT)
    assert mod.__name__ == "langchain_community"


def test_community_symbols_exist(depcheck):
    """Every langchain_community symbol the backend imports (BM25 + the loader
    family) must still resolve at its current dotted path."""
    mod = depcheck.load(COMMUNITY_IMPORT)
    depcheck.assert_symbols(mod, COMMUNITY_SYMBOLS)


def test_community_version_reported(depcheck):
    assert depcheck.dist_version(COMMUNITY_DIST) is not None


def test_bm25_retriever_contract(depcheck):
    """retrieval/utils.py builds BM25Retriever.from_texts(texts=, metadatas=),
    sets `.k`, and the ensemble invokes it. Contract (offline, no embeddings):
    from_texts builds a retriever, `.k` caps results, and invoke(query) returns
    Documents whose metadata is the per-text metadata we supplied."""
    community = depcheck.load(COMMUNITY_IMPORT)
    BM25Retriever = depcheck.resolve(community, "retrievers").BM25Retriever

    # rank_bm25 backs BM25Retriever; skip cleanly if the extra isn't present.
    if depcheck.try_load("rank_bm25") is None:
        pytest.skip("rank_bm25 (BM25Retriever backend) not importable in this env")

    texts = ["the cat sat on the mat", "dogs run fast", "a feline on a rug"]
    metas = [{"i": 0}, {"i": 1}, {"i": 2}]
    r = BM25Retriever.from_texts(texts=texts, metadatas=metas)
    r.k = 2

    out = r.invoke("cat")
    assert isinstance(out, list) and len(out) == 2
    assert all(type(d).__name__ == "Document" for d in out)
    # metadata threaded through from the per-text metadatas.
    assert all("i" in d.metadata for d in out)


def test_loader_subclassing_base_contract(depcheck):
    """retrieval/web/utils.py imports document_loaders.base.BaseLoader and the
    backend's custom loaders subclass langchain_core.document_loaders.BaseLoader.
    The community `base.BaseLoader` must remain the same class object as the core
    one (community re-exports core) so both import sites agree, and `.load()`
    must remain the abstract entry point."""
    community = depcheck.load(COMMUNITY_IMPORT)
    core = depcheck.load(CORE_IMPORT)
    community_base = depcheck.resolve(community, "document_loaders.base").BaseLoader
    core_base = depcheck.resolve(core, "document_loaders").BaseLoader

    assert community_base is core_base, (
        "community document_loaders.base.BaseLoader diverged from core's "
        "BaseLoader; the backend imports both and expects one class."
    )
    assert hasattr(core_base, "load"), "BaseLoader lost its .load() entry point"


# --------------------------------------------------------------------------- #
# langchain_classic — composite retrievers
# --------------------------------------------------------------------------- #
def test_classic_import(depcheck):
    mod = depcheck.load(CLASSIC_IMPORT)
    assert mod.__name__ == "langchain_classic"


def test_classic_symbols_exist(depcheck):
    """EnsembleRetriever / ContextualCompressionRetriever moved into the
    `langchain-classic` distribution in the modern split; the backend imports
    them from `langchain_classic.retrievers`."""
    mod = depcheck.load(CLASSIC_IMPORT)
    depcheck.assert_symbols(mod, CLASSIC_SYMBOLS)


def test_classic_version_reported(depcheck):
    assert depcheck.dist_version(CLASSIC_DIST) is not None


def test_ensemble_retriever_contract(depcheck):
    """retrieval/utils.py builds EnsembleRetriever(retrievers=[...], weights=[...],
    id_key=CHUNK_HASH_KEY) and invokes it. Contract (offline, BM25-only member):
    construction with an explicit id_key works, and invoke(query) returns
    Documents (RRF-fused) from the wrapped retriever."""
    classic = depcheck.load(CLASSIC_IMPORT)
    community = depcheck.load(COMMUNITY_IMPORT)
    EnsembleRetriever = depcheck.resolve(classic, "retrievers").EnsembleRetriever
    BM25Retriever = depcheck.resolve(community, "retrievers").BM25Retriever

    if depcheck.try_load("rank_bm25") is None:
        pytest.skip("rank_bm25 (BM25Retriever backend) not importable in this env")

    # id_key is the dedup key the backend passes (CHUNK_HASH_KEY) so enriched
    # BM25 texts don't defeat RRF. The RRF path reads doc.metadata[id_key], so
    # every text must carry that key — mirror the backend, whose chunks do.
    bm25 = BM25Retriever.from_texts(
        texts=["alpha beta", "beta gamma", "gamma delta"],
        metadatas=[{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}],
    )
    bm25.k = 2

    ens = EnsembleRetriever(retrievers=[bm25], weights=[1.0], id_key="hash")
    assert ens.id_key == "hash"

    out = ens.invoke("beta")
    assert isinstance(out, list) and len(out) >= 1
    assert all(type(d).__name__ == "Document" for d in out)


def test_contextual_compression_retriever_is_constructible(depcheck):
    """retrieval/utils.py builds ContextualCompressionRetriever(base_compressor=,
    base_retriever=) and awaits .ainvoke(query). Pin that the constructor still
    takes those two keyword args (offline; we don't run the rerank/embedding)."""
    classic = depcheck.load(CLASSIC_IMPORT)
    CCR = depcheck.resolve(classic, "retrievers").ContextualCompressionRetriever

    params = inspect.signature(CCR.__init__).parameters
    # pydantic-model retrievers accept **data; only assert names when no var-kw.
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not has_var_kw:
        for name in ("base_compressor", "base_retriever"):
            assert name in params, f"ContextualCompressionRetriever lost {name!r}"
    # invoke/ainvoke are the public entry points the backend calls.
    assert callable(getattr(CCR, "ainvoke", None))
    assert callable(getattr(CCR, "invoke", None))
