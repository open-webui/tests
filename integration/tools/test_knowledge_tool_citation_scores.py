"""Regression: citations from the knowledge search tools carried no relevance scores.

Fix PR open-webui/open-webui#31307 (merged as `1c813902e`, issue open-webui/open-webui#29776)
in `utils/middleware.py`. With native function calling, `query_knowledge_files` and
`query_chat_files` return a distance per chunk, but the step that groups a tool's chunks into
the reply's citation sources dropped it, so those citations never showed the relevance
percentage that classic retrieval shows for the same knowledge base. Each grouped source now
carries a `distances` list aligned with its documents.

The scripted model calls each tool and the test reads the sources stored on the reply, which is
what the citation list renders after a reload. Since dev 98f7a6146 a small chat file goes to the
model whole, unsearched, so the chat file test turns the full-text threshold off.

Twin of e2e/tools/test_knowledge_tool_citation_scores.py.

Discriminates: passes on dev bc2416c5d; with the fix reverted the stored sources of both tools
have no `distances`; the file-and-text and empty-search tests pass on both.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import without_citation_context
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

NOTES = "Grey herons nest in colonies near water and hunt fish in the shallows."
LEDGER = "The heron survey budget for the spring is four thousand."


def read_grant(account) -> list[dict]:
    return [{"principal_type": "user", "principal_id": account.id, "permission": "read"}]


@pytest.fixture
def library(admin, make_user):
    """A knowledge base of two files shared with a reader; yields the reader, its id, the files."""
    reader = make_user()
    tag = uuid.uuid4().hex[:8]
    with (
        admin.client() as client,
        knowledge_base(client, f"Birds {tag}", read_grant(reader)) as knowledge_id,
    ):
        notes = add_text_file(client, knowledge_id, f"herons-{tag}.txt", NOTES)
        ledger = add_text_file(client, knowledge_id, f"ledger-{tag}.txt", LEDGER)
        yield reader, knowledge_id, {notes, ledger}


@pytest.fixture
def file_reader(admin):
    """A preset that reads a chat's files through tools, so `query_chat_files` is offered."""
    model_id = f"file-reader-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "File reader",
        "meta": {"capabilities": {"file_upload": True, "file_context": False}},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        client.post("/api/v1/models/create", json=form).raise_for_status()
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def cited_by_tool(actor, upstream, tool: str, arguments: dict, **chat) -> tuple[list, list]:
    """The sources stored on the reply, and the chunks the tool handed the model."""
    upstream.queue(reply.tool_call(tool, arguments), reply.text("done"))
    with actor.client() as client:
        _, message = ask(client, f"use {tool}", **chat)
    sent_back = reply.conversation(upstream.chat_requests()[-1])
    [tool_result] = [entry["content"] for entry in sent_back if entry["role"] == "tool"]
    return message.get("sources") or [], json.loads(without_citation_context(tool_result))


def assert_every_document_is_scored(sources: list[dict]) -> None:
    assert sources, "the reply cites nothing"
    for source in sources:
        scores = source.get("distances")
        assert scores, f"a cited source carries no relevance scores: {source}"
        assert len(scores) == len(source["document"]), source
        assert all(isinstance(score, (int, float)) for score in scores), scores


# --- narrow: the knowledge search's citations keep their scores -----------------------------


def test_knowledge_file_citations_carry_their_relevance_scores(library, upstream):
    reader, knowledge_id, _ = library

    sources, _ = cited_by_tool(
        reader,
        upstream,
        "query_knowledge_files",
        {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10},
    )

    assert_every_document_is_scored(sources)


def test_chat_file_citations_carry_their_relevance_scores(
    admin, preserve, library, file_reader, upstream
):
    reader, _, file_ids = library
    preserve(RETRIEVAL_CONFIG)
    # since 98f7a6146 a small chat file goes whole, unsearched and so unscored
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": 0})
        assert saved.status_code == 200, saved.text
    files = [{"type": "file", "id": file_id, "name": file_id} for file_id in sorted(file_ids)]

    sources, _ = cited_by_tool(
        reader, upstream, "query_chat_files", {"query": "heron"}, model=file_reader, files=files
    )

    assert_every_document_is_scored(sources)


# --- broad: the scores are the tool's own, one per chunk, grouped by file -------------------


def test_each_score_is_the_distance_the_tool_returned(library, upstream):
    reader, knowledge_id, file_ids = library

    sources, chunks = cited_by_tool(
        reader,
        upstream,
        "query_knowledge_files",
        {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10},
    )

    returned = {(chunk["file_id"], chunk["content"], chunk["distance"]) for chunk in chunks}
    cited = {
        (source["source"]["id"], document, score)
        for source in sources
        for document, score in zip(source["document"], source["distances"])
    }
    assert cited == returned
    assert {source["source"]["id"] for source in sources} == file_ids


# --- nearby: the citations themselves are unchanged -----------------------------------------


def test_the_citations_still_name_their_files_and_text(library, upstream):
    reader, knowledge_id, file_ids = library

    sources, _ = cited_by_tool(
        reader,
        upstream,
        "query_knowledge_files",
        {"query": "heron", "knowledge_ids": [knowledge_id], "count": 10},
    )

    assert {source["source"]["id"] for source in sources} == file_ids
    assert {document for source in sources for document in source["document"]} == {NOTES, LEDGER}
    for source in sources:
        assert len(source["metadata"]) == len(source["document"])


def test_an_empty_search_cites_nothing(admin, make_user, upstream):
    reader = make_user()
    with (
        admin.client() as client,
        knowledge_base(client, "Empty", read_grant(reader)) as knowledge_id,
    ):
        sources, chunks = cited_by_tool(
            reader,
            upstream,
            "query_knowledge_files",
            {"query": "heron", "knowledge_ids": [knowledge_id]},
        )

    assert chunks == []
    assert sources == []
