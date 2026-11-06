"""Regression: uploads and hybrid search failed on Qdrant with strict mode enabled.

Two fixes for issue open-webui/open-webui#31459:

- Paged reads (8a90f0fc9, open-webui/open-webui#31461). Open WebUI read collections from Qdrant
  in one scroll asking for 999,999,999 points. Strict mode caps that with `max_query_limit` and
  answers "Limit exceeded", so in the default multitenancy mode every upload after the first
  failed its duplicate check, a second file could not join a knowledge base, and hybrid search,
  which reads the whole collection, found nothing. Reads now go in pages of 1,000 points.
- Fallback (eda8d8536, open-webui/open-webui#31460). With hybrid search on, a collection that
  could not be read was skipped quietly, so a chat got no documents and never fell back to
  vector search. Such a collection now counts as failed, and the chat falls back.

Qdrant is the local stand-in from harness/qdrant_server.py, which refuses a scroll or query over
its `max_query_limit` the way strict mode does. Both storage modes are run.

Discriminates: passes on dev efe63bd34. With 8a90f0fc9 reverted the stand-in refuses the
999,999,999-point scroll: the upload and knowledge base tests fail in multitenancy mode (the
other mode swallows the refused duplicate check) and both hybrid search tests fail in either
mode. With eda8d8536 reverted the fallback test fails in both (the model gets no chunk).
"""

from __future__ import annotations

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.file_context import file_context_ticked
from harness.knowledge_bases import knowledge_base
from harness.qdrant_server import FakeQdrant, qdrant_env, serving_qdrant
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

STRICT_LIMIT = 1000
LINE_CHUNKS = {
    "TEXT_SPLITTER": "",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": False,
    "CHUNK_SIZE": 12,
    "CHUNK_OVERLAP": 0,
    "CHUNK_MIN_SIZE_TARGET": 0,
    "ENABLE_RAG_HYBRID_SEARCH": True,
    "RAG_FULL_CONTEXT": False,
    "BYPASS_EMBEDDING_AND_RETRIEVAL": False,
    # a chat would get a small file whole instead of searching it
    "RAG_FULL_TEXT_CHAR_THRESHOLD": 0,
    "TOP_K": 5,
}


def tide_lines(first: int, count: int) -> str:
    return "\n".join(f"tide {number:05d}" for number in range(first, first + count))


@pytest.fixture(scope="module")
def qdrant() -> FakeQdrant:
    with serving_qdrant() as fake:
        yield fake


@pytest.fixture(scope="module", params=[True, False], ids=["multitenancy", "collections"])
def on_qdrant(request, instance_with, qdrant):
    # large embedding batches keep the 2,500-chunk upload to a few requests
    env = {**qdrant_env(qdrant, multitenancy=request.param), "RAG_EMBEDDING_BATCH_SIZE": "4096"}
    launched = instance_with(env)
    with admin_of(launched).client() as client:
        original = client.get(RETRIEVAL_CONFIG[0]).json()
        saved = client.post(RETRIEVAL_CONFIG[1], json=LINE_CHUNKS)
        assert saved.status_code == 200, saved.text
        try:
            yield launched
        finally:
            client.post(RETRIEVAL_CONFIG[1], json=original)


@pytest.fixture
def admin_client(on_qdrant, qdrant):
    """The admin of the Qdrant instance, with strict mode on at `STRICT_LIMIT`."""
    qdrant.max_query_limit = STRICT_LIMIT
    qdrant.refused_limits.clear()
    with admin_of(on_qdrant).client() as client:
        yield client
    qdrant.max_query_limit = None


def upload(client: httpx.Client, filename: str, text: str) -> httpx.Response:
    return client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )


def uploaded(client: httpx.Client, filename: str, text: str) -> str:
    answer = upload(client, filename, text)
    assert answer.status_code == 200, answer.text
    file_id = answer.json()["id"]
    stored = client.get(f"/api/v1/files/{file_id}").json()
    assert stored["data"].get("status") == "completed", stored["data"].get("error")
    return file_id


def query_doc(client: httpx.Client, file_id: str, query: str) -> httpx.Response:
    return client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": f"file-{file_id}", "query": query, "k": 3},
    )


def test_every_upload_succeeds_under_strict_mode(admin_client, qdrant):
    for index in range(3):
        answer = upload(admin_client, f"tides-{index}.txt", tide_lines(index * 100, 20))
        assert answer.status_code == 200, (
            f"upload {index + 1} failed under strict mode (#31459): {answer.text}"
        )
        stored = admin_client.get(f"/api/v1/files/{answer.json()['id']}").json()
        assert stored["data"].get("status") == "completed", stored["data"].get("error")
    assert qdrant.refused_limits == [], "a read asked Qdrant for more than strict mode allows"


def test_a_second_file_joins_a_knowledge_base_under_strict_mode(admin_client):
    first = uploaded(admin_client, "north.txt", tide_lines(0, 10))
    second = uploaded(admin_client, "south.txt", tide_lines(500, 10))
    with knowledge_base(admin_client, "Tide tables") as knowledge_id:
        for file_id in (first, second):
            added = admin_client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
            )
            assert added.status_code == 200, (
                f"adding a file to the knowledge base failed under strict mode: {added.text}"
            )


def test_hybrid_search_finds_the_chunk_under_strict_mode(admin_client):
    file_id = uploaded(admin_client, "harbour.txt", tide_lines(0, 50))

    answered = query_doc(admin_client, file_id, "tide 00042")

    assert answered.status_code == 200, answered.text
    assert "tide 00042" in answered.json()["documents"][0], (
        "hybrid search found nothing under strict mode (#31459)"
    )


def test_hybrid_search_reads_a_collection_longer_than_one_page(admin_client, qdrant):
    qdrant.max_query_limit = None
    file_id = uploaded(admin_client, "long.txt", tide_lines(0, 2500))
    qdrant.max_query_limit = STRICT_LIMIT

    answered = query_doc(admin_client, file_id, "tide 02400")

    assert answered.status_code == 200, answered.text
    assert "tide 02400" in answered.json()["documents"][0], "a chunk past the first page was lost"


def test_a_chat_falls_back_to_vector_search_when_the_collection_cannot_be_read(
    admin_client, on_qdrant, qdrant
):
    file_id = uploaded(admin_client, "ferry.txt", "ferry 09:00")  # one chunk
    qdrant.max_query_limit = 100  # below the 1,000-point pages, so no read gets through
    question = "when does the ferry leave?"
    on_qdrant.upstream.queue(reply.text("noted", match=reply.answering(question)))

    with file_context_ticked(admin_of(on_qdrant)):
        ask(admin_client, question, files=[{"type": "file", "id": file_id, "name": "ferry.txt"}])

    sent = on_qdrant.upstream.chat_requests()[-1]["messages"]
    assert any("ferry 09:00" in str(message.get("content")) for message in sent), (
        "the unreadable collection was skipped and the chat never fell back to vector search"
    )
    assert qdrant.refused_limits, "the collection read was not refused, so nothing was tested"


def test_without_strict_mode_hybrid_search_still_answers(admin_client, qdrant):
    qdrant.max_query_limit = None
    file_id = uploaded(admin_client, "open.txt", tide_lines(0, 30))

    answered = query_doc(admin_client, file_id, "tide 00007")

    assert answered.status_code == 200, answered.text
    assert "tide 00007" in answered.json()["documents"][0]
