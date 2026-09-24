"""The initial title task crashed on every new chat, open-webui/open-webui#30339.

Fix commit `6b36f620c` (PR #30356). Two refactors in v0.10.0 drifted apart: the background task
handler started reading `ctx['model']` for the memory review, and the early title task in
`main.py` calls that handler with a context that has no `model`. So every new saved chat ended
its title task with `KeyError: 'model'`, logged as "Error generating initial chat title" once
PR #30106 raised it from DEBUG. The title was already written by then and the memory settings
played no part. The handler now reads `ctx.get('model')`, and the memory review, which accepts
no model, stops on the title path because that context carries no finished reply.

Twin of unit/chat/test_initial_title_background_context.py.

The log line only exists because PR #30106 made the handler log with `log.exception`; the
unit audit in unit/chat/test_initial_title_background_context.py keeps it that way.

Discriminates: with 6b36f620c reverted both tests that open a new chat fail on the logged
KeyError; the memory-feature gate passes on both.
"""

from __future__ import annotations

import time

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.instance import LaunchedInstance

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TITLE_ERROR = "Error generating initial chat title"
MEMORY_REVIEWER = "You are Open WebUI's private memory reviewer. Return only valid JSON."
MEMORY_REVIEW_ENV = {
    "ENABLE_MEMORIES": "true",
    "ENABLE_MEMORY_BACKGROUND_REVIEW": "true",
    "MEMORIES_REVIEW_INTERVAL_TURNS": "1",
}


def is_task(body: dict) -> bool:
    return not body.get("stream")


def is_memory_review(body: dict) -> bool:
    return body["messages"][0]["content"] == MEMORY_REVIEWER


def log_after_the_title_task(instance: LaunchedInstance, offset: int) -> str:
    # The crash came right after the title write; give it a moment to be logged.
    time.sleep(1)
    return instance.log_since(offset)


def chat_title(client, chat_id: str, expected: str) -> str:
    deadline = time.monotonic() + 15
    title = ""
    while time.monotonic() < deadline:
        title = client.get(f"/api/v1/chats/{chat_id}").json()["title"]
        if title == expected:
            break
        time.sleep(0.2)
    return title


@pytest.fixture
def title_generation_on(admin, preserve) -> None:
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        enabled = {**current, "ENABLE_TITLE_GENERATION": True}
        client.post("/api/v1/tasks/config/update", json=enabled).raise_for_status()


def test_a_new_chat_gets_its_title_without_an_error(instance, user, upstream, title_generation_on):
    upstream.queue(
        reply.text('{"title": "Paris trip plans"}', match=is_task), reply.text("Sure, let's plan.")
    )
    offset = instance.log_size()
    with user.client() as client:
        turn, message = ask(
            client, "help me plan a trip to Paris", background_tasks={"title_generation": True}
        )
        title = chat_title(client, turn.chat_id, "Paris trip plans")

    assert message["content"] == "Sure, let's plan."
    assert title == "Paris trip plans"
    assert TITLE_ERROR not in log_after_the_title_task(instance, offset)


@pytest.fixture(scope="module")
def memory_instance(instance_with) -> LaunchedInstance:
    return instance_with(MEMORY_REVIEW_ENV)


def memory_reviews(upstream, wait: float) -> list[dict]:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        reviews = [body for body in upstream.chat_requests() if is_memory_review(body)]
        if reviews:
            return reviews
        time.sleep(0.1)
    return [body for body in upstream.chat_requests() if is_memory_review(body)]


@pytest.mark.slow
def test_a_new_chat_is_reviewed_once_with_its_model_and_titled_without_an_error(memory_instance):
    upstream = memory_instance.upstream
    upstream.reset()
    upstream.queue(reply.text('{"operations": []}', match=is_memory_review), reply.text("Noted."))
    offset = memory_instance.log_size()
    member = create_user(memory_instance)
    with member.client() as client:
        turn, _ = ask(
            client,
            "I moved to Vienna last year.",
            features={"memory": True},
            background_tasks={"title_generation": True},
        )
        title = chat_title(client, turn.chat_id, "I moved to Vienna last year.")

    assert memory_reviews(upstream, wait=10), "the finished reply was never reviewed"
    logged = log_after_the_title_task(memory_instance, offset)
    reviews = memory_reviews(upstream, wait=0)
    assert [review["model"] for review in reviews] == ["mock-model"]
    assert "I moved to Vienna last year." in reviews[0]["messages"][-1]["content"]
    assert title == "I moved to Vienna last year."
    assert TITLE_ERROR not in logged


@pytest.mark.slow
def test_a_chat_without_the_memory_feature_is_not_reviewed(memory_instance):
    upstream = memory_instance.upstream
    upstream.reset()
    upstream.queue(reply.text("Noted."))
    member = create_user(memory_instance)
    with member.client() as client:
        ask(client, "I moved to Vienna last year.", features={"memory": False})

    assert memory_reviews(upstream, wait=1.5) == []
