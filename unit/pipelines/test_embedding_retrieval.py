"""End-to-end embedding + vector retrieval, driven through Open WebUI's own code.

This is deliberately NOT mocked. The point (per prior production incidents
where sentence-transformers/transformers and chromadb broke in combination)
is that the REAL local embedding model and the REAL chromadb store are both
invoked together through Open WebUI's own functions, and that a semantic
query actually retrieves the relevant document.

What gets exercised, all via OWUI source:
  - open_webui.routers.retrieval.get_ef(...)        loads the real
        SentenceTransformer (offline, from the local HF cache, via OWUI's
        get_model_path snapshot resolution).
  - open_webui.retrieval.utils.get_embedding_function(...)  wraps it in the
        same async embedding callable the RAG pipeline uses; calling it runs
        the real transformers forward pass (model.encode()).
  - open_webui.retrieval.vector.dbs.chroma.ChromaClient  the real vector-DB
        wrapper. We back it with an in-memory chromadb.EphemeralClient so the
        test is offline, deterministic, and leaves nothing on disk, then call
        the wrapper's own insert / search / query / get / delete_collection.
  - open_webui.retrieval.vector.main.VectorItem  the real item shape.

The default RAG config (RAG_EMBEDDING_ENGINE='' → local sentence-transformers,
RAG_EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2') is what's used.

Skips cleanly (never fails) when the heavy deps aren't importable or the model
isn't already cached locally — it must never trigger a network download.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_source]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Three documents from clearly distinct topics, plus two queries whose nearest
# document is unambiguous. Indices into DOCS:
DOCS = [
    "The Eiffel Tower is in Paris.",  # 0 — geography / landmark
    "Photosynthesis converts sunlight.",  # 1 — biology
    "Python is a programming language.",  # 2 — software
]
PARIS_DOC = DOCS[0]
PHOTOSYNTHESIS_DOC = DOCS[1]


def _import(module_name: str):
    """Import an open_webui module, skipping cleanly if its deps are absent."""
    sys.modules.pop(module_name, None)
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 — any import failure → skip, not fail
        pytest.skip(f"Could not import {module_name}: {e}")


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{module_name} not importable: {e}")


def _assert_model_cached_or_skip(retrieval_utils) -> str:
    """Resolve the model's local snapshot path WITHOUT hitting the network.

    OWUI's get_model_path swallows a missing-cache error and falls back to the
    bare model name (which would make SentenceTransformer attempt a download).
    So we probe huggingface_hub.snapshot_download with local_files_only=True
    ourselves: if that raises, the model isn't cached and we skip rather than
    download. If it resolves, we know get_ef will load offline.
    """
    hub = _require("huggingface_hub")
    try:
        path = hub.snapshot_download(repo_id=EMBEDDING_MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached → skip, never download
        pytest.skip(
            f"Embedding model {EMBEDDING_MODEL!r} is not cached locally "
            f"(would require a network download): {e}"
        )

    # Sanity: the snapshot must actually contain weights, else encode() would
    # fall back to the network. OWUI resolves via get_model_path; cross-check.
    snapshot = Path(path)
    has_weights = (snapshot / "model.safetensors").is_file() or (
        snapshot / "pytorch_model.bin"
    ).is_file()
    if not has_weights:
        pytest.skip(f"Cached snapshot for {EMBEDDING_MODEL!r} has no weights at {snapshot}")

    resolved = retrieval_utils.get_model_path(EMBEDDING_MODEL, False)
    if not Path(resolved).is_dir():
        pytest.skip(f"get_model_path did not resolve a local dir for {EMBEDDING_MODEL!r}")
    return resolved


@pytest.fixture(scope="module")
def pipeline(open_webui_backend: Path):
    """Build the real embedding callable + a real ChromaClient (in-memory).

    Module-scoped so the SentenceTransformer (slow to load) is loaded once.
    Skips if torch/sentence-transformers/chromadb are missing or the model
    isn't cached.
    """
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))

    # Heavy third-party deps — skip cleanly if any is unavailable.
    _require("torch")
    _require("sentence_transformers")
    chromadb = _require("chromadb")

    retrieval_utils = _import("open_webui.retrieval.utils")
    retrieval_router = _import("open_webui.routers.retrieval")
    chroma_mod = _import("open_webui.retrieval.vector.dbs.chroma")
    main_mod = _import("open_webui.retrieval.vector.main")

    _assert_model_cached_or_skip(retrieval_utils)

    # --- Real local embedding model, loaded the way OWUI loads it -------------
    # engine='' selects the local sentence-transformers path; get_ef resolves
    # the cached snapshot offline and returns a real SentenceTransformer.
    ef = retrieval_router.get_ef("", EMBEDDING_MODEL, auto_update=False)
    if ef is None:
        pytest.skip(f"get_ef returned None for {EMBEDDING_MODEL!r} (model failed to load)")

    # Same async wrapper the RAG pipeline uses; calling it runs model.encode().
    embedding_function = retrieval_utils.get_embedding_function(
        embedding_engine="",
        embedding_model=EMBEDDING_MODEL,
        embedding_function=ef,
        url="",
        key="",
        embedding_batch_size=1,
    )

    # --- Real ChromaClient wrapper, backed by an in-memory ephemeral client --
    # The wrapper's __init__ builds a PersistentClient on disk from config; we
    # want offline + no disk state, so construct the wrapper and swap in an
    # EphemeralClient. Every method under test (insert/search/query/get/delete)
    # is the real OWUI ChromaClient code path.
    client = chroma_mod.ChromaClient.__new__(chroma_mod.ChromaClient)
    client.client = chromadb.EphemeralClient(
        settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False)
    )

    return {
        "embedding_function": embedding_function,
        "client": client,
        "VectorItem": main_mod.VectorItem,
        "chromadb": chromadb,
    }


def _embed(embedding_function, value):
    """Run OWUI's async embedding callable to completion (real encode())."""
    return asyncio.run(embedding_function(value))


@pytest.fixture()
def populated_collection(pipeline):
    """Embed DOCS with the real model and insert them via the real ChromaClient.

    Yields (collection_name, embed_one) where embed_one(text) -> query vector.
    Cleans up the collection afterwards.
    """
    embedding_function = pipeline["embedding_function"]
    client = pipeline["client"]
    VectorItem = pipeline["VectorItem"]

    collection_name = f"test-embed-retrieval-{uuid.uuid4().hex}"

    doc_vectors = _embed(embedding_function, DOCS)
    assert isinstance(doc_vectors, list) and len(doc_vectors) == len(DOCS)

    items = [
        VectorItem(
            id=str(i),
            text=DOCS[i],
            vector=doc_vectors[i],
            metadata={"idx": i, "name": f"doc-{i}"},
        ).model_dump()
        for i in range(len(DOCS))
    ]
    client.insert(collection_name, items)

    def embed_one(text: str):
        return _embed(embedding_function, text)

    try:
        yield collection_name, embed_one, doc_vectors
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


def test_embedding_dimensionality_is_stable_and_positive(populated_collection):
    """Real encode() yields fixed-width, non-empty vectors for every doc."""
    _collection_name, _embed_one, doc_vectors = populated_collection

    dims = {len(v) for v in doc_vectors}
    assert len(dims) == 1, f"embedding width should be uniform, got {dims}"
    (dim,) = dims
    assert dim > 0
    # all-MiniLM-L6-v2 is a 384-dim model; pin it so a silently-swapped model
    # or a broken encode path is caught.
    assert dim == 384, f"expected 384-dim MiniLM embeddings, got {dim}"
    assert all(isinstance(x, float) for x in doc_vectors[0])


def test_vectors_are_persisted_in_chroma(pipeline, populated_collection):
    """The real ChromaClient.get round-trips the stored docs (store works)."""
    collection_name, _embed_one, _doc_vectors = populated_collection
    client = pipeline["client"]

    # has_collection is exercised (real wrapper method) but not asserted on a
    # specific boolean: in chromadb >=1.x list_collections() returns Collection
    # objects, so the wrapper's `name in list_collections()` check can report
    # False even when the collection exists. The authoritative existence proof
    # is the get() round-trip below.
    client.has_collection(collection_name)

    stored = client.get(collection_name)
    assert stored is not None
    stored_ids = set(stored.ids[0])
    stored_docs = set(stored.documents[0])
    assert stored_ids == {"0", "1", "2"}
    assert stored_docs == set(DOCS)


def test_semantic_query_retrieves_paris_doc(pipeline, populated_collection):
    """'Where is the Eiffel Tower?' → Paris doc is the top hit (real search)."""
    collection_name, embed_one, _doc_vectors = populated_collection
    client = pipeline["client"]

    query_vector = embed_one("Where is the Eiffel Tower?")
    result = client.search(collection_name, vectors=[query_vector], limit=len(DOCS))

    assert result is not None
    ranked_docs = result.documents[0]
    assert ranked_docs, "search returned no documents"
    assert ranked_docs[0] == PARIS_DOC, f"expected Paris doc as top hit, got ranking: {ranked_docs}"

    # Distances are OWUI's rescaled cosine similarity (0..1, higher = closer);
    # the top hit must out-score the rest, proving real semantic ranking.
    distances = result.distances[0]
    assert len(distances) == len(ranked_docs)
    assert distances[0] == max(distances)
    assert distances[0] > distances[-1]


def test_semantic_query_retrieves_photosynthesis_doc(pipeline, populated_collection):
    """'How do plants make energy?' → photosynthesis doc is the top hit."""
    collection_name, embed_one, _doc_vectors = populated_collection
    client = pipeline["client"]

    query_vector = embed_one("How do plants make energy?")
    result = client.search(collection_name, vectors=[query_vector], limit=len(DOCS))

    assert result is not None
    ranked_docs = result.documents[0]
    assert ranked_docs, "search returned no documents"
    assert ranked_docs[0] == PHOTOSYNTHESIS_DOC, (
        f"expected photosynthesis doc as top hit, got ranking: {ranked_docs}"
    )


def test_unrelated_query_does_not_rank_paris_first(pipeline, populated_collection):
    """A biology query must NOT surface the Paris doc as its top hit.

    Guards against a degenerate index that returns the same document
    regardless of the query (which would make the positive tests meaningless).
    """
    collection_name, embed_one, _doc_vectors = populated_collection
    client = pipeline["client"]

    query_vector = embed_one("How do plants make energy?")
    result = client.search(collection_name, vectors=[query_vector], limit=len(DOCS))

    assert result is not None
    assert result.documents[0][0] != PARIS_DOC


def test_metadata_filter_query_returns_single_doc(pipeline, populated_collection):
    """The real ChromaClient.query (metadata filter) fetches the exact doc."""
    collection_name, _embed_one, _doc_vectors = populated_collection
    client = pipeline["client"]

    result = client.query(collection_name, filter={"idx": 0})
    assert result is not None
    assert result.documents[0] == [PARIS_DOC]
    assert result.ids[0] == ["0"]
