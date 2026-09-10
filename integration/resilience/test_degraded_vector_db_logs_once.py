"""Guard: a chat over knowledge bases whose vector DB is down costs one log line.

The instance's vector DB refuses every connection. A chat that references several knowledge
bases still answers (the retrieval step degrades to no sources) but the fan-out renders a
full traceback twice per knowledge base and query. The assertion is on the server log written
during the request: at most one traceback. The answer arriving at all is the control.

Unit form: `unit/resilience/test_degraded_vector_db_logs_once.py`.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where five knowledge bases log two
tracebacks each per query; strict `xfail`. Unmarked: no issue filed yet.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from integration.conftest import MOCK_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

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


def test_chat_still_answers_with_the_vector_db_down(degraded_instance, knowledge_ids):
    degraded_instance.upstream.reset(mode="ok", text="answer")

    with degraded_instance.client() as client:
        response = _chat_over(client, knowledge_ids)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "answer"


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="two tracebacks per knowledge base")
def test_chat_with_the_vector_db_down_logs_at_most_one_traceback(degraded_instance, knowledge_ids):
    degraded_instance.upstream.reset(mode="ok", text="answer")
    offset = degraded_instance.log_size()

    with degraded_instance.client() as client:
        _chat_over(client, knowledge_ids).raise_for_status()

    tracebacks = degraded_instance.log_since(offset).count("Traceback (most recent call last)")
    assert tracebacks <= 1, f"{tracebacks} tracebacks for one chat over {KNOWLEDGE_BASES} bases"
