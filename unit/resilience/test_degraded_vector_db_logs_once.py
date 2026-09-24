"""Guard: a partial vector DB outage keeps the hits of the collections that answered.

#29981 (ff7f35a30) stopped `query_collection` logging a traceback twice per failed collection
and query, collecting the failures and logging them once after the fan-out instead. That
collecting must not cost the results of the collections that did answer. The log count itself
is pinned from outside in `integration/resilience/test_degraded_vector_db_logs_once.py`; this
half stays here because a running instance has one vector DB, which is either up or down.

The vector DB client is the one boundary stubbed, specced from `VectorDBBase`.
Discriminates: passes on dev bbfa876af, fails once a copy of it returns an empty result when
any collection failed.
"""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest

COLLECTIONS = [f"kb-{i}" for i in range(5)]


async def _embed(queries, prefix=None):
    return [[0.1, 0.2]] * len(queries)


@pytest.fixture(scope="module")
def vector_main_module(owui_module):
    return owui_module("open_webui.retrieval.vector.main")


@pytest.mark.asyncio
async def test_a_healthy_collection_beside_failing_ones_still_yields_its_hits(
    retrieval_utils_module, vector_main_module, monkeypatch
):
    hit = vector_main_module.SearchResult(
        ids=[["d1"]], distances=[[0.1]], documents=[["text"]], metadatas=[[{"source": "d1"}]]
    )

    def search(collection_name, vectors, limit, **_):
        if collection_name == "kb-0":
            return hit
        raise ConnectionError("vector db unreachable")

    client = create_autospec(vector_main_module.VectorDBBase, instance=True)
    client.search.side_effect = search
    monkeypatch.setattr(retrieval_utils_module, "get_vector_db_client", lambda: client)

    result = await retrieval_utils_module.query_collection(
        request=None,
        collection_names=COLLECTIONS,
        queries=["first question"],
        embedding_function=_embed,
        k=3,
    )

    assert client.search.call_count == len(COLLECTIONS)
    assert result["documents"] == [["text"]]
