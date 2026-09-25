"""Journey: the task routes run only on a model the caller may read, and only admins configure them.

Title, follow-up, tag, image prompt, query, autocompletion, emoji and mixture-of-agents
generation each take the chat's model and call it. A user naming a provider model with no
workspace entry, or one whose entry is shared with nobody, is refused and the provider never
sees the task; once the entry is shared with them the same request is answered by that model.
Saving the task settings is refused to a user and leaves them as they were.

Discriminates: in a backend copy, skipping the model check both in `generate_chat_completion`
of `utils/chat.py` and in the OpenAI route it hands the task to (`bypass_filter` forced on)
turns all 16 refusal rows red (the provider runs the task for the user); either one alone stays
green, as the other still refuses. Switching `update_task_config` to `get_verified_user` turns
the settings test red.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import pytest

from harness import upstream as reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GATED_MODEL = "task-gated-model"
TASK_CONFIG = ("/api/v1/tasks/config", "/api/v1/tasks/config/update")
TASKS_ON = {
    "ENABLE_TITLE_GENERATION": True,
    "ENABLE_FOLLOW_UP_GENERATION": True,
    "ENABLE_TAGS_GENERATION": True,
    "ENABLE_SEARCH_QUERY_GENERATION": True,
    "ENABLE_AUTOCOMPLETE_GENERATION": True,
}
MESSAGES = [
    {"role": "user", "content": "how do herons fish?"},
    {"role": "assistant", "content": "They stand still and strike."},
]

# task: (path, body without the model)
TASKS = {
    "title": ("/api/v1/tasks/title/completions", {"messages": MESSAGES}),
    "follow-up": ("/api/v1/tasks/follow_up/completions", {"messages": MESSAGES}),
    "tags": ("/api/v1/tasks/tags/completions", {"messages": MESSAGES}),
    "image prompt": ("/api/v1/tasks/image_prompt/completions", {"messages": MESSAGES}),
    "search queries": (
        "/api/v1/tasks/queries/completions",
        {"type": "web_search", "messages": MESSAGES},
    ),
    "autocompletion": (
        "/api/v1/tasks/auto/completions",
        {"type": "general", "prompt": "herons", "messages": MESSAGES},
    ),
    "emoji": ("/api/v1/tasks/emoji/completions", {"prompt": "herons fishing"}),
    "mixture of agents": (
        "/api/v1/tasks/moa/completions",
        {"prompt": "how do herons fish?", "responses": ["they wait", "they strike"]},
    ),
}


@pytest.fixture
def tasks_on(admin, preserve, upstream):
    """Every gated task switched on, with a provider model only the admin can use so far."""
    preserve("tasks")
    upstream.models.append(GATED_MODEL)
    with admin.client() as client:
        current = client.get(TASK_CONFIG[0]).json()
        client.post(TASK_CONFIG[1], json={**current, **TASKS_ON}).raise_for_status()
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield
    upstream.models.remove(GATED_MODEL)
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"})


@contextlib.contextmanager
def _workspace_entry(admin, access_grants: list[dict]) -> Iterator[None]:
    form = {
        "id": GATED_MODEL,
        "base_model_id": None,
        "name": GATED_MODEL,
        "meta": {},
        "params": {},
        "access_grants": access_grants,
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        try:
            yield
        finally:
            client.post("/api/v1/models/model/delete", json={"id": GATED_MODEL})


def _run_task(actor, task: str):
    path, body = TASKS[task]
    with actor.client() as client:
        return client.post(path, json={**body, "model": GATED_MODEL})


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("entry", ["unregistered", "unshared"])
def test_a_task_on_a_model_the_user_cannot_read_is_refused(
    task, entry, tasks_on, admin, make_user, upstream
):
    with contextlib.ExitStack() as stack:
        if entry == "unshared":
            stack.enter_context(_workspace_entry(admin, []))
        answered = _run_task(make_user(), task)

    assert answered.status_code >= 400, (
        f"a user ran {task} generation on the {entry} {GATED_MODEL}: {answered.text[:200]}"
    )
    assert upstream.chat_requests() == []


@pytest.mark.parametrize("task", TASKS)
def test_a_task_on_a_model_shared_with_the_user_runs(task, tasks_on, admin, make_user, upstream):
    reader = make_user()
    upstream.queue(reply.text('{"title": "Heron fishing"}'))
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with _workspace_entry(admin, [grant]):
        answered = _run_task(reader, task)

    assert answered.status_code == 200, answered.text
    assert [body["model"] for body in upstream.chat_requests()] == [GATED_MODEL]


def test_a_user_cannot_change_the_task_settings(admin, make_user, preserve):
    preserve("tasks")
    with admin.client() as client:
        before = client.get(TASK_CONFIG[0]).json()
    changed = {**before, "ENABLE_TITLE_GENERATION": not before["ENABLE_TITLE_GENERATION"]}

    with make_user().client() as client:
        refused = client.post(TASK_CONFIG[1], json=changed)
    with admin.client() as client:
        after = client.get(TASK_CONFIG[0]).json()

    assert refused.status_code == 401, refused.text
    assert after == before
