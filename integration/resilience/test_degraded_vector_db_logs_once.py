"""Guard: a vector DB outage costs one log entry per knowledge base and no answering hits.

The instance's vector DB refuses every connection. A chat that references several knowledge
bases still answers (the retrieval step degrades to no sources), but the search fan-out used to
render a full traceback twice per knowledge base and query: once in `query_doc`, which
re-raised, and again in the handler that caught it. #29981 (ff7f35a30) logs one aggregated entry
per search instead. The chat still searches each base on its own, so the whole chat logs one
entry per base. Entries are
counted as top-level tracebacks in the server log, chained exceptions not counted again.

Collecting the failures must not cost the hits of the collections that did answer. That half
runs on the Qdrant stand-in with one collection per knowledge base, four of five of them down:
a search over all five still returns the healthy base's document. Nearby, a chat over a failing
and a healthy base, each searched on its own, still gives the model the healthy one's text.

Discriminates: passes on dev bbfa876af, fails with ff7f35a30 reverted in a copy of it (ten
entries for five bases, two per base). The search test passes on ef67cc3fa and fails once a copy
of `query_collection` returns an empty result when any collection failed.
"""

from __future__ import annotations

import contextlib
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.qdrant_server import qdrant_env, serving_qdrant
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

KNOWLEDGE_BASES = 5
LOGBOOK = "The lighthouse keeper logs the tide at dawn."


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


@pytest.fixture(scope="module")
def partly_down(instance_with):
    """Five knowledge bases on the Qdrant stand-in, the first holding LOGBOOK, the rest down.

    Yields the instance and the ids, healthy base first.
    """
    with serving_qdrant() as qdrant:
        launched = instance_with(qdrant_env(qdrant, multitenancy=False))
        with admin_of(launched).client() as client, contextlib.ExitStack() as bases:
            ids = [
                bases.enter_context(knowledge_base(client, f"kb-{index}"))
                for index in range(KNOWLEDGE_BASES)
            ]
            add_text_file(client, ids[0], "logbook.txt", LOGBOOK)
            qdrant.unavailable = set(ids[1:])
            yield launched, ids
            qdrant.unavailable = set()


def test_a_search_over_partly_failing_collections_keeps_the_healthy_hits(partly_down):
    launched, knowledge_ids = partly_down

    with admin_of(launched).client() as client:
        searched = client.post(
            "/api/v1/retrieval/query/collection",
            json={"collection_names": knowledge_ids, "query": "who logs the tide?", "k": 3},
        )

    assert searched.status_code == 200, searched.text
    assert searched.json()["documents"] == [[LOGBOOK]], (
        "the answering collection's hit was dropped because the others failed (#29981)"
    )


def test_a_chat_over_a_healthy_and_a_failing_base_still_gets_the_healthy_text(partly_down):
    launched, knowledge_ids = partly_down
    question = f"who logs the tide? {uuid.uuid4().hex[:6]}"
    launched.upstream.queue(reply.text("the keeper", match=reply.answering(question)))
    bases = [{"type": "collection", "id": kb_id} for kb_id in knowledge_ids[1::-1]]

    with admin_of(launched).client() as client:
        _, answer = ask(client, question, chat_files=bases)

    assert answer["content"] == "the keeper"
    sent = next(filter(reply.answering(question), launched.upstream.chat_requests()))
    assert LOGBOOK in str(sent["messages"]), "the healthy base's text never reached the model"
