"""Guard: a vector DB that is down costs one log line per request.

`query_collection` fans out `collections x queries` searches in threads and already degrades
to an empty result when they all fail. On the way it renders a full traceback twice per failed
pair: once in `query_doc`, which re-raises, and again in the fan-out handler that catches it.
A chat with twenty knowledge bases and five expanded queries turns one outage into two
hundred identical stack traces per message, on every replica, at ERROR. The fix is one
aggregated warning after the gather.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where the fan-out logs 2 x N x M
tracebacks; that case is a strict `xfail`. The degrade itself holds today and is the control.
Unmarked: no issue filed yet.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

COLLECTIONS = [f"kb-{i}" for i in range(5)]
QUERIES = ["first question", "second question"]


class DownVectorDb:
    def search(self, collection_name, vectors, limit):
        raise ConnectionError("vector db unreachable")


async def _embed(queries, prefix=None):
    return [[0.1, 0.2]] * len(queries)


async def _config_without_a_database(*keys):
    return {key: None for key in keys}


@pytest.fixture
def query_collection(retrieval_utils_module, monkeypatch):
    monkeypatch.setattr(retrieval_utils_module, "get_vector_db_client", lambda: DownVectorDb())
    monkeypatch.setattr(
        retrieval_utils_module.Config, "get_many", staticmethod(_config_without_a_database)
    )
    return retrieval_utils_module.query_collection


@pytest.mark.asyncio
async def test_every_collection_failing_still_returns_an_empty_result(query_collection):
    """Control: the outage degrades to no sources; the request goes on."""
    result = await query_collection(None, COLLECTIONS, QUERIES, _embed, k=3)

    assert result == {"distances": [[]], "documents": [[]], "metadatas": [[]]}


@pytest.mark.xfail(
    raises=AssertionError, strict=True, reason="two tracebacks per collection x query pair"
)
@pytest.mark.asyncio
async def test_every_collection_failing_logs_at_most_one_traceback(query_collection, caplog):
    await query_collection(None, COLLECTIONS, QUERIES, _embed, k=3)

    tracebacks = [
        r for r in caplog.records if r.name == "open_webui.retrieval.utils" and r.exc_info
    ]
    assert len(tracebacks) <= 1, (
        f"{len(tracebacks)} tracebacks for {len(COLLECTIONS)} collections x {len(QUERIES)} "
        "queries; one outage, one log line"
    )


@pytest.mark.asyncio
async def test_a_healthy_collection_beside_a_failing_one_still_yields_its_hits(
    retrieval_utils_module, query_collection, monkeypatch
):
    """Control: a partial outage keeps the results the healthy collections returned."""
    hit_payload = {
        "ids": [["d1"]],
        "distances": [[0.1]],
        "documents": [["text"]],
        "metadatas": [[{"source": "d1"}]],
    }
    hit = SimpleNamespace(model_dump=lambda: hit_payload, **hit_payload)

    class HalfDown(DownVectorDb):
        def search(self, collection_name, vectors, limit):
            if collection_name == "kb-0":
                return hit
            raise ConnectionError("vector db unreachable")

    monkeypatch.setattr(retrieval_utils_module, "get_vector_db_client", lambda: HalfDown())

    result = await query_collection(None, COLLECTIONS, QUERIES[:1], _embed, k=3)

    assert result["documents"] == [["text"]]
