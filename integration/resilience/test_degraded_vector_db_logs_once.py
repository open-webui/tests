"""Guard: a chat over knowledge bases whose vector DB is down costs one log entry per base.

The instance's vector DB refuses every connection. A chat that references several knowledge
bases still answers (the retrieval step degrades to no sources), but the search fan-out used to
render a full traceback twice per knowledge base and query: once in `query_doc`, which
re-raised, and again in the handler that caught it. #29981 (ff7f35a30) logs one aggregated entry
per search instead. The chat still searches each base on its own, so the whole chat logs one
entry per base; that it should log one in total is the open half, a strict `xfail`. Entries are
counted as top-level tracebacks in the server log, chained exceptions not counted again.

Twin of unit/resilience/test_degraded_vector_db_logs_once.py.
Discriminates: passes on dev bbfa876af, fails with ff7f35a30 reverted in a copy of it (ten
entries for five bases, two per base).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

KNOWLEDGE_BASES = 5


@pytest.fixture(scope="module")
def knowledge_ids(degraded_instance) -> list[str]:
    ids = []
    with degraded_instance.client() as client:
        for _ in range(KNOWLEDGE_BASES):
            created = client.post(
                "/api/v1/knowledge/create",
                json={"name": f"kb-{uuid.uuid4().hex[:8]}", "description": "empty"},
            )
            created.raise_for_status()
            ids.append(created.json()["id"])
    return ids


def _chat_over(client: httpx.Client, knowledge_ids: list[str]) -> httpx.Response:
    return client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": "what do the documents say"}],
            "stream": False,
            "files": [{"type": "collection", "id": kb_id} for kb_id in knowledge_ids],
        },
    )


def _logged_failures(log: str) -> int:
    """Tracebacks in the log, not counting the causes chained onto one."""
    chained = log.count("During handling of the above exception") + log.count(
        "The above exception was the direct cause"
    )
    return log.count("Traceback (most recent call last)") - chained


def _failures_during_a_chat(instance, knowledge_ids: list[str]) -> int:
    instance.upstream.reset(mode="ok", text="answer")
    offset = instance.log_size()
    with instance.client() as client:
        _chat_over(client, knowledge_ids).raise_for_status()
    return _logged_failures(instance.log_since(offset))


def test_chat_still_answers_with_the_vector_db_down(degraded_instance, knowledge_ids):
    degraded_instance.upstream.reset(mode="ok", text="answer")

    with degraded_instance.client() as client:
        response = _chat_over(client, knowledge_ids)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "answer"


def test_each_knowledge_base_logs_its_failed_search_once(degraded_instance, knowledge_ids):
    failures = _failures_during_a_chat(degraded_instance, knowledge_ids)

    assert failures, "the failed searches were not logged at all; the count below means nothing"
    assert failures <= KNOWLEDGE_BASES, (
        f"{failures} tracebacks for one chat over {KNOWLEDGE_BASES} bases; each failed search "
        "is logged once per knowledge base (#29981)"
    )


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="one entry per knowledge base")
def test_chat_with_the_vector_db_down_logs_at_most_one_traceback(degraded_instance, knowledge_ids):
    failures = _failures_during_a_chat(degraded_instance, knowledge_ids)

    assert failures <= 1, f"{failures} tracebacks for one chat over {KNOWLEDGE_BASES} bases"
