"""Local embeddings and the Chroma store, driven together through Open WebUI's own code.

Deliberately not mocked: past releases broke when sentence-transformers, transformers and
chromadb shifted under each other, which only shows when the real model encodes and the real
store ranks. `routers.retrieval.get_ef` loads the default RAG model offline the way the server
does, `retrieval.utils.get_embedding_function` wraps it as the RAG pipeline does, and a real
`ChromaClient` (a persistent store under the scratch `DATA_DIR`) inserts, fetches, filters and
ranks.

Stays a unit test: the integration instance embeds through the scripted provider, which returns
one fixed vector, so it cannot rank anything. Skips only when the model is not in the local
Hugging Face cache or `sentence_transformers` is not installed; it never downloads.

Discriminates: passes on bbfa876af; fails with `ChromaClient.search` handing back Chroma's raw
cosine distances instead of the rescaled similarity (the best hit no longer scores highest).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_source]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PARIS = "The Eiffel Tower is in Paris."
PHOTOSYNTHESIS = "Photosynthesis converts sunlight."
DOCS = [PARIS, PHOTOSYNTHESIS, "Python is a programming language."]


@pytest.fixture(scope="module")
def embed(owui_module):
    """The RAG pipeline's own async embedding function over the real local model."""
    pytest.importorskip("sentence_transformers", reason="the local embedding engine is optional")
    hub = pytest.importorskip("huggingface_hub")
    try:
        hub.snapshot_download(repo_id=EMBEDDING_MODEL, local_files_only=True)
    except hub.utils.LocalEntryNotFoundError:
        pytest.skip(f"{EMBEDDING_MODEL} is not in the local Hugging Face cache")

    model = owui_module("open_webui.routers.retrieval").get_ef(
        engine="", embedding_model=EMBEDDING_MODEL, auto_update=False
    )
    assert model is not None, f"get_ef could not load the cached {EMBEDDING_MODEL}"
    embedding_function = owui_module("open_webui.retrieval.utils").get_embedding_function(
        embedding_engine="",
        embedding_model=EMBEDDING_MODEL,
        embedding_function=model,
        url="",
        key="",
        embedding_batch_size=1,
    )
    return lambda texts: asyncio.run(embedding_function(texts))


@pytest.fixture(scope="module")
def store(owui_module, embed):
    """A collection holding DOCS, embedded by the real model; yields (client, name)."""
    chroma = owui_module("open_webui.retrieval.vector.dbs.chroma").ChromaClient()
    vector_item = owui_module("open_webui.retrieval.vector.main").VectorItem
    name = f"embedding-retrieval-{uuid.uuid4().hex}"
    vectors = embed(DOCS)
    chroma.insert(
        collection_name=name,
        items=[
            vector_item(
                id=str(index), text=doc, vector=vectors[index], metadata={"idx": index}
            ).model_dump()
            for index, doc in enumerate(DOCS)
        ],
    )
    yield chroma, name
    chroma.delete_collection(collection_name=name)


def test_every_document_is_embedded_at_the_models_width(embed):
    widths = {len(vector) for vector in embed(DOCS)}

    assert widths == {384}, f"all-MiniLM-L6-v2 encodes 384 dimensions, got {widths}"


def test_the_stored_documents_round_trip(store):
    chroma, name = store

    stored = chroma.get(collection_name=name)

    assert set(stored.ids[0]) == {"0", "1", "2"}
    assert set(stored.documents[0]) == set(DOCS)


@pytest.mark.parametrize(
    ("question", "answer"),
    [("Where is the Eiffel Tower?", PARIS), ("How do plants make energy?", PHOTOSYNTHESIS)],
)
def test_a_question_ranks_its_answer_first(embed, store, question, answer):
    chroma, name = store

    ranked = chroma.search(collection_name=name, vectors=embed([question]), limit=len(DOCS))

    assert ranked.documents[0][0] == answer, ranked.documents[0]
    scores = ranked.distances[0]
    assert scores[0] == max(scores) > min(scores), f"the best hit does not score highest: {scores}"


def test_a_metadata_filter_fetches_exactly_its_document(store):
    chroma, name = store

    found = chroma.query(collection_name=name, filter={"idx": 0})

    assert found.documents[0] == [PARIS]
