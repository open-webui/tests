"""The model's knowledge-base search lists only knowledge bases the asker can open.

The builtin `query_knowledge_bases` tool searches the knowledge-base embeddings with
`filter={'knowledge_base_id': {'$in': [...]}}`, the knowledge bases the caller may read, and
reports each hit's name and description. Commit 1d6d4e6e6 (v0.11.1) fixed eleven vector backends
that dropped that filter. The default Chroma store always applied it, so this test cannot see that
fix; it guards the other half, the tool sending the filter. The backends stay covered by the unit
test.

Twin of unit/security/test_knowledge_search_collection_acl.py.

Discriminates: passes on bbfa876af with or without 1d6d4e6e6 (Chroma applies the filter), fails with
the `filter` argument removed from the search in `query_knowledge_bases` (the reader is shown the
admin's private knowledge base).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.mock_embeddings import embed_through

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@dataclass
class KnowledgeBases:
    readable_id: str
    private_id: str
    reader: Actor = field(repr=False)


def create_knowledge_base(client: httpx.Client, name: str, access_grants: list[dict]) -> str:
    created = client.post(
        "/api/v1/knowledge/create",
        json={"name": name, "description": f"{name} notes", "access_grants": access_grants},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture
def mock_embeddings(upstream, admin: Actor, preserve) -> None:
    embed_through(upstream, admin, preserve)


@pytest.fixture
def knowledge_bases(admin: Actor, make_user, mock_embeddings) -> KnowledgeBases:
    """Two of the admin's knowledge bases, one shared with a reader and one private."""
    reader = make_user()
    read_grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with admin.client() as client:
        readable_id = create_knowledge_base(client, "Handbook", [read_grant])
        private_id = create_knowledge_base(client, "Salary reviews", [])
    return KnowledgeBases(readable_id, private_id, reader)


def knowledge_bases_found_by_the_model(actor: Actor, upstream) -> set[str]:
    # every text embeds to the same vector, so ask for all of them
    upstream.queue(
        reply.tool_call("query_knowledge_bases", {"query": "notes", "count": 100}),
        reply.text("done"),
    )
    with actor.client() as client:
        ask(client, "which knowledge base has the notes?")
    replayed = upstream.chat_requests()[-1]["messages"]
    tool_results = [entry["content"] for entry in replayed if entry["role"] == "tool"]
    assert len(tool_results) == 1, f"the tool result was not replayed: {replayed}"
    listed = json.loads(tool_results[0])
    assert isinstance(listed, list), f"query_knowledge_bases failed: {listed}"
    return {entry["id"] for entry in listed}


def test_the_search_tool_hides_knowledge_bases_the_reader_cannot_open(knowledge_bases, upstream):
    found = knowledge_bases_found_by_the_model(knowledge_bases.reader, upstream)
    assert knowledge_bases.private_id not in found, (
        "query_knowledge_bases listed a knowledge base the caller cannot open"
    )
    assert knowledge_bases.readable_id in found


def test_the_search_tool_finds_both_for_their_owner(admin, knowledge_bases, upstream):
    found = knowledge_bases_found_by_the_model(admin, upstream)
    assert {knowledge_bases.readable_id, knowledge_bases.private_id} <= found
