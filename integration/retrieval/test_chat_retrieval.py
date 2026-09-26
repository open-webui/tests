"""Journey: what a chat with an attached file or knowledge base hands the model, and cites.

A file attached to a chat is cut into chunks when it is uploaded; each question then embeds the
question, takes the `TOP_K` nearest chunks and wraps them in the RAG template ahead of the
question, and the reply keeps them as citations. Full context sends every chunk, and bypassing
embedding sends the whole text without embedding the question at all. Hybrid search reranks the
candidates and drops those under the relevance threshold; an external reranker, when set, is the
one that scores them, and `TOP_K_RERANKER` keeps its best. Chunk overlap repeats text between
neighbouring chunks, and a minimum chunk size merges small markdown sections.

The instance embeds through a local service whose vectors follow a few keywords, so a question
about herons is nearest the chunks about herons and nothing else; a local service also plays the
reranker.

Discriminates: in a backend copy, the files handler asking for `TOP_K` + 1 chunks turned both
top-k tests red; `get_all_items_from_collections` returning nothing turned the full context test
red; the file branch ignoring the bypass switch turned the bypass test red; `RerankCompressor`
ignoring `r_score` turned the threshold test red; the external reranker pairing its scores with
the wrong chunks turned the reranker test red; the reply dropping its sources turned the citation
test red; the character splitter built without `chunk_overlap` turned the overlap test red; and
`merge_docs_to_target_size` returning its input turned the minimum size test red.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.keyword_embeddings import (
    embedded_texts,
    keyword_embedding_env,
    serve_keyword_embeddings,
)
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import ReceivedRequest, json_answer, listening
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

KEYWORDS = ["heron", "otter", "grebe"]
PARAGRAPHS = [
    "The grey heron waits in the shallows at dawn.",
    "Otters slide down the muddy bank after rain.",
    "A second heron stalks frogs along the ditch.",
    "Great crested grebes dance on the open water.",
    "Reed warblers sing from the tall stems.",
    "Coots squabble over weed near the jetty.",
]
HERON_PARAGRAPHS = {PARAGRAPHS[0], PARAGRAPHS[2]}
FIELD_NOTES = "\n\n".join(PARAGRAPHS)
# one paragraph per chunk: two paragraphs together run past it
ONE_PARAGRAPH_CHUNKS = {
    "TEXT_SPLITTER": "",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
    "CHUNK_SIZE": 60,
    "CHUNK_OVERLAP": 0,
    "CHUNK_MIN_SIZE_TARGET": 0,
}


@pytest.fixture(scope="module")
def embedder():
    with listening() as service:
        serve_keyword_embeddings(service, KEYWORDS)
        yield service


@pytest.fixture
def ranked(instance_with, embedder, preserve):
    """The keyword-embedding instance, its retrieval settings restored after the test."""
    launched = instance_with(keyword_embedding_env(embedder))
    preserve(RETRIEVAL_CONFIG, on=launched)
    return launched


@pytest.fixture
def settings(ranked):
    """`settings(**changes)` saves retrieval settings on the keyword-embedding instance."""
    with admin_of(ranked).client() as client:

        def save(**changes) -> None:
            saved = client.post(RETRIEVAL_CONFIG[1], json=changes)
            assert saved.status_code == 200, saved.text

        save(**ONE_PARAGRAPH_CHUNKS)
        yield save


@pytest.fixture
def reader(ranked):
    return create_user(ranked)


def upload(client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def ask_about(ranked, actor, question: str, files: list[dict]) -> tuple[str, dict]:
    """Ask with `files` attached; returns what the model was sent and the stored reply."""
    ranked.upstream.queue(reply.text("noted", match=reply.answering(question)))
    with actor.client() as client:
        _, message = ask(client, question, files=files)
    sent = ranked.upstream.chat_requests()[-1]["messages"]
    return "\n".join(str(entry.get("content")) for entry in sent), message


def attached_file(actor, text: str = FIELD_NOTES, filename: str = "field-notes.txt") -> dict:
    with actor.client() as client:
        file_id = upload(client, filename, text)
    return {"type": "file", "id": file_id, "name": filename}


def paragraphs_in(sent: str) -> set[str]:
    return {paragraph for paragraph in PARAGRAPHS if paragraph in sent}


# --- which chunks the model gets ----------------------------------------------------------------


def test_the_top_k_nearest_chunks_are_sent(ranked, settings, reader):
    settings(TOP_K=2, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False)
    notes = attached_file(reader)

    sent, _ = ask_about(ranked, reader, "Where does the heron hunt?", [notes])

    assert paragraphs_in(sent) == HERON_PARAGRAPHS, f"expected the two heron chunks: {sent}"


def test_a_larger_top_k_sends_more_chunks(ranked, settings, reader):
    settings(TOP_K=4, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False)
    notes = attached_file(reader)

    sent, _ = ask_about(ranked, reader, "Where does the heron hunt?", [notes])

    assert HERON_PARAGRAPHS < paragraphs_in(sent) and len(paragraphs_in(sent)) == 4, sent


def test_full_context_sends_every_chunk(ranked, settings, reader):
    settings(TOP_K=1, RAG_FULL_CONTEXT=True, ENABLE_RAG_HYBRID_SEARCH=False)
    notes = attached_file(reader)

    sent, _ = ask_about(ranked, reader, "What lives by the heron pond?", [notes])

    assert paragraphs_in(sent) == set(PARAGRAPHS), f"full context left chunks out: {sent}"


def test_a_file_attached_in_full_sends_its_whole_text(ranked, settings, reader):
    settings(TOP_K=1, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False)
    notes = {**attached_file(reader), "context": "full"}

    sent, _ = ask_about(ranked, reader, "Tell me about the heron.", [notes])

    assert FIELD_NOTES in sent


def test_bypassing_embedding_sends_the_text_without_embedding_the_question(
    ranked, settings, reader, embedder
):
    settings(BYPASS_EMBEDDING_AND_RETRIEVAL=True, TOP_K=1)
    notes = attached_file(reader)
    embedded_before = len(embedded_texts(embedder))

    sent, _ = ask_about(ranked, reader, "Anything about otters?", [notes])

    assert FIELD_NOTES in sent
    assert embedded_texts(embedder)[embedded_before:] == [], "the question was embedded"


def test_a_knowledge_base_attached_to_the_chat_is_searched(ranked, settings, reader):
    settings(TOP_K=2, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False)
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with (
        admin_of(ranked).client() as client,
        knowledge_base(client, "Field guide", [grant]) as guide_id,
    ):
        add_text_file(client, guide_id, "field-notes.txt", FIELD_NOTES)
        guide = {"type": "collection", "id": guide_id, "name": "Field guide"}

        sent, _ = ask_about(ranked, reader, "Where does the heron hunt?", [guide])

    assert paragraphs_in(sent) == HERON_PARAGRAPHS, sent


# --- hybrid search and reranking ----------------------------------------------------------------


def test_the_relevance_threshold_drops_weak_chunks(ranked, settings, reader):
    settings(
        TOP_K=6,
        TOP_K_RERANKER=6,
        ENABLE_RAG_HYBRID_SEARCH=True,
        HYBRID_BM25_WEIGHT=0.5,
        RELEVANCE_THRESHOLD=0.5,
        RAG_FULL_CONTEXT=False,
    )
    notes = attached_file(reader)

    strict, _ = ask_about(ranked, reader, "Where does the heron hunt?", [notes])
    settings(RELEVANCE_THRESHOLD=0.0)
    lenient, _ = ask_about(ranked, reader, "Where does the heron hunt?", [notes])

    assert paragraphs_in(strict) == HERON_PARAGRAPHS, f"weak chunks passed the threshold: {strict}"
    assert paragraphs_in(lenient) == set(PARAGRAPHS)


@pytest.fixture
def reranker():
    """A reranker that scores the grebe paragraph highest; records what it was asked."""

    def rerank(request: ReceivedRequest):
        documents = request.json()["documents"]
        scores = [0.9 if "grebe" in document else 0.1 for document in documents]
        results = [{"index": index, "relevance_score": score} for index, score in enumerate(scores)]
        return json_answer({"results": list(reversed(results))})

    with listening() as service:
        service.route("POST", "/rerank", rerank)
        yield service


def test_an_external_reranker_picks_the_chunks(ranked, settings, reader, reranker):
    settings(
        TOP_K=6,
        TOP_K_RERANKER=1,
        ENABLE_RAG_HYBRID_SEARCH=True,
        HYBRID_BM25_WEIGHT=0.5,
        RELEVANCE_THRESHOLD=0.0,
        RAG_FULL_CONTEXT=False,
        RAG_RERANKING_ENGINE="external",
        RAG_RERANKING_MODEL="test-reranker",
        RAG_EXTERNAL_RERANKER_URL=f"{reranker.base_url}/rerank",
        RAG_EXTERNAL_RERANKER_API_KEY="rerank-key",
    )
    notes = attached_file(reader)

    sent, _ = ask_about(ranked, reader, "Where does the heron hunt?", [notes])

    asked = reranker.requests_to("/rerank")
    assert asked, "the external reranker was never asked"
    assert asked[-1].headers.get("Authorization") == "Bearer rerank-key"
    body = asked[-1].json()
    assert (body["model"], body["query"]) == ("test-reranker", "Where does the heron hunt?")
    assert paragraphs_in(sent) == {PARAGRAPHS[3]}, f"the reranker's pick was not sent: {sent}"


# --- citations and the template -----------------------------------------------------------------


def test_the_reply_cites_the_file_and_its_chunks(ranked, settings, reader):
    settings(TOP_K=2, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False)
    notes = attached_file(reader)

    _, message = ask_about(ranked, reader, "Where does the heron hunt?", [notes])

    [source] = message["sources"]
    assert source["source"]["id"] == notes["id"]
    assert set(source["document"]) == HERON_PARAGRAPHS
    assert {entry["file_id"] for entry in source["metadata"]} == {notes["id"]}
    assert {entry["name"] for entry in source["metadata"]} == {"field-notes.txt"}


def test_a_custom_rag_template_frames_the_context(ranked, settings, reader):
    template = "Answer only from these notes:\n{{CONTEXT}}\nThe question was: {{QUERY}}"
    settings(TOP_K=1, RAG_FULL_CONTEXT=False, ENABLE_RAG_HYBRID_SEARCH=False, RAG_TEMPLATE=template)
    notes = attached_file(reader)

    sent, _ = ask_about(ranked, reader, "Which grebe dances?", [notes])

    assert "Answer only from these notes:" in sent
    assert "The question was: Which grebe dances?" in sent
    assert f">{PARAGRAPHS[3]}</source>" in sent, sent


# --- chunking -----------------------------------------------------------------------------------


def stored_chunks(actor, file_id: str) -> list[str]:
    """The file's chunks in document order."""
    with actor.client() as client:
        answered = client.post(
            "/api/v1/retrieval/query/doc",
            json={"collection_name": f"file-{file_id}", "query": "heron", "k": 100},
        )
    assert answered.status_code == 200, answered.text
    result = answered.json()
    ordered = sorted(
        zip(result["metadatas"][0], result["documents"][0]),
        key=lambda pair: pair[0]["start_index"],
    )
    return [document for _, document in ordered]


def test_chunk_overlap_repeats_text_between_neighbouring_chunks(ranked, settings, reader):
    settings(CHUNK_SIZE=40, CHUNK_OVERLAP=15)
    words = " ".join(f"word{index:02d}" for index in range(30))
    notes = attached_file(reader, words, "words.txt")

    chunks = stored_chunks(reader, notes["id"])

    assert len(chunks) > 2 and max(map(len, chunks)) <= 40, chunks
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.split()[0] in earlier.split(), f"no overlap between {earlier!r} and {later!r}"
    settings(CHUNK_OVERLAP=0)
    without = stored_chunks(reader, attached_file(reader, words, "plain.txt")["id"])
    assert " ".join(without) == words, "without overlap the chunks should tile the text"


def test_small_markdown_sections_are_merged_up_to_the_minimum(ranked, settings, reader):
    settings(
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True,
        CHUNK_SIZE=1000,
        CHUNK_MIN_SIZE_TARGET=60,
    )
    sections = "\n\n".join(f"## Bird {index}\n\nA heron." for index in range(6))
    notes = attached_file(reader, sections, "birds.md")

    merged = stored_chunks(reader, notes["id"])
    settings(CHUNK_MIN_SIZE_TARGET=0)
    unmerged = stored_chunks(reader, attached_file(reader, sections, "birds-2.md")["id"])

    assert len(unmerged) == 6, unmerged
    assert 1 < len(merged) < 6, f"the small sections were not merged: {merged}"
    assert all(chunk.startswith("## Bird") for chunk in merged)
