"""The model's knowledge-base search lists only knowledge bases the asker can open, on every store.

The builtin `query_knowledge_bases` tool searches the knowledge-base embeddings with
`filter={'knowledge_base_id': {'$in': [...]}}`, the knowledge bases the caller may read, and
reports each hit's name and description. Before commit 1d6d4e6e6 (v0.11.1) eleven bundled vector
backends took that filter and dropped it, so the store answered with every neighbour, forbidden
knowledge bases included: Qdrant sent no `query_filter` in either storage mode and Elasticsearch
only the collection term. The default Chroma store always applied it.

The same search runs on an instance of its own on the default Chroma store and on instances booted
on the local Qdrant stand-in, in both storage modes, the Elasticsearch stand-in and the Pinecone
stand-in, each of which applies the filter it is sent. The Chroma one is not the shared instance:
the scripted provider embeds every text alike, and on a store that has collected and deleted the
knowledge bases of earlier tests, Chroma's filtered search then leaves some matches out.

Twin of unit/security/test_knowledge_search_collection_acl.py, which keeps the audit that no
vector backend ignores a `filter` argument.

Discriminates: passes on ef67cc3fa; with the filter dropped from `search` in the Qdrant, Qdrant
multitenancy, Elasticsearch or Pinecone backend the reader is shown the owner's private knowledge
base on that store, and with the `filter` argument removed from the search in
`query_knowledge_bases` on every store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor, admin_of, create_user
from harness.chat import ask
from harness.elasticsearch_server import elasticsearch_env, serving_elasticsearch
from harness.pinecone_server import pinecone_env, serving_pinecone
from harness.qdrant_server import qdrant_env, serving_qdrant

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# each store boots an instance of its own
STORES = [
    pytest.param("chroma", marks=pytest.mark.slow),
    pytest.param("qdrant-multitenancy", marks=pytest.mark.slow),
    pytest.param("qdrant-collections", marks=pytest.mark.slow),
    pytest.param("elasticsearch", marks=pytest.mark.slow),
    pytest.param("pinecone", marks=pytest.mark.slow),
]


@dataclass
class KnowledgeBases:
    readable_id: str
    private_id: str
    owner: Actor = field(repr=False)
    reader: Actor = field(repr=False)
    upstream: object = field(repr=False)


@pytest.fixture(scope="module")
def outside_stores(instance_with, tmp_path_factory):
    """Instances on the Qdrant, Elasticsearch and Pinecone stand-ins, booted when a test asks."""
    with (
        serving_qdrant() as qdrant,
        serving_elasticsearch() as elasticsearch,
        serving_pinecone() as pinecone,
    ):
        envs = {
            "qdrant-multitenancy": qdrant_env(qdrant, multitenancy=True),
            "qdrant-collections": qdrant_env(qdrant, multitenancy=False),
            "elasticsearch": elasticsearch_env(elasticsearch),
            # the scripted provider embeds every text as three numbers
            "pinecone": pinecone_env(pinecone, tmp_path_factory.mktemp("pinecone"), dimension=3),
        }
        yield lambda store: instance_with(envs[store])


@pytest.fixture(params=STORES)
def store(request):
    if request.param == "chroma":
        return request.getfixturevalue("instance_with")({})
    return request.getfixturevalue("outside_stores")(request.param)


def create_knowledge_base(client: httpx.Client, name: str, access_grants: list[dict]) -> str:
    created = client.post(
        "/api/v1/knowledge/create",
        json={"name": name, "description": f"{name} notes", "access_grants": access_grants},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture
def knowledge_bases(store) -> KnowledgeBases:
    """Two of an account's knowledge bases, one shared with a reader and one private.

    Not the admin's: the admin reads every knowledge base the instance has collected, and with
    every text embedded alike the store's nearest hundred of those need not include these two.
    """
    owner = create_user(store)
    make_group(admin_of(store), [owner], {"workspace": {"knowledge": True}})
    reader = create_user(store)
    read_grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        readable_id = create_knowledge_base(client, "Handbook", [read_grant])
        private_id = create_knowledge_base(client, "Salary reviews", [])
    store.upstream.reset()
    return KnowledgeBases(readable_id, private_id, owner, reader, store.upstream)


def knowledge_bases_found_by_the_model(actor: Actor, upstream) -> set[str]:
    # every text embeds to the same vector, so ask for all of them
    upstream.queue(
        reply.tool_call("query_knowledge_bases", {"query": "notes", "count": 100}),
        reply.text("done"),
    )
    with actor.client() as client:
        ask(client, "which knowledge base has the notes?")
    replayed = reply.conversation(upstream.chat_requests()[-1])
    tool_results = [entry["content"] for entry in replayed if entry["role"] == "tool"]
    assert len(tool_results) == 1, f"the tool result was not replayed: {replayed}"
    listed = json.loads(tool_results[0])
    assert isinstance(listed, list), f"query_knowledge_bases failed: {listed}"
    return {entry["id"] for entry in listed}


def test_the_search_tool_hides_knowledge_bases_the_reader_cannot_open(knowledge_bases):
    found = knowledge_bases_found_by_the_model(knowledge_bases.reader, knowledge_bases.upstream)
    assert knowledge_bases.private_id not in found, (
        "query_knowledge_bases listed a knowledge base the caller cannot open"
    )
    assert knowledge_bases.readable_id in found


def test_the_search_tool_finds_both_for_their_owner(knowledge_bases):
    found = knowledge_bases_found_by_the_model(knowledge_bases.owner, knowledge_bases.upstream)
    assert {knowledge_bases.readable_id, knowledge_bases.private_id} <= found
