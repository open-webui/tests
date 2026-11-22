"""Three 0.11.0 chat-store fixes: folder paging, sidebar order and PostgreSQL search.

* `409fb39` (#26786) `GET /api/v1/folders/{id}/shared/chats`, which lists a folder's chats for
  its owner and for everyone it is shared with, returned one fixed page of sixty with no count,
  so chats past the sixtieth never appeared. It now takes `page`, `sort_by` and `sort_dir` and
  answers `total` and `has_more`.
* `f1ded94` / `a9617ca` background writes during a reply (sources, files, embeds, follow-up
  suggestions, the arena's pick, now a router model's, and the compaction summary) went through
  the chat upsert, which always bumped `updated_at`, so the sidebar re-sorted mid-reply. The
  upsert now takes `touch` and the background writers pass `touch=False`. The follow-ups and the
  router pick are written deep inside a live reply, so the scripted provider answers them a few
  seconds late: a write that still moved the chat would show as a later `updated_at`.
* `cc9a445` on PostgreSQL a chat search matched only the legacy flat `chat->'messages'` array,
  while current conversations keep their text in `chat_message` and under
  `chat->'history'->'messages'`. It now matches all three.

Twin of unit/models/test_chat_search_and_folder_paging.py, which keeps the sweep over every chat
upsert that writes one of those background fields, so a new call site is held to it too.

Discriminates: passes on upstream dev `bbfa876af`. With the folder route back on one unpaged call
the paging tests fail; with the chat upsert bumping `updated_at` regardless of `touch` the three
background event cases fail; on dev ef67cc3fa, with `touch=False` dropped from the follow-ups,
arena pick and compaction summary writes, those three tests fail (the arena pick test is
retargeted to router models on dev 53d741157, where it passes); with the PostgreSQL match
back on the legacy array alone the three history searches on PostgreSQL find nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ChatTurn, send_message
from harness.instance import LaunchedInstance, launch
from harness.router_models import (
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    default_routes,
    save_router_model,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FOLDER_SIZE = 65
PAGE_SIZE = 10


def create_folder(client: httpx.Client) -> str:
    created = client.post("/api/v1/folders/", json={"name": f"Paging {uuid.uuid4().hex[:6]}"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def create_chat(client: httpx.Client, chat: dict, folder_id: str | None = None) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": chat, "folder_id": folder_id})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def folder_page(client: httpx.Client, folder_id: str, **params) -> dict:
    response = client.get(f"/api/v1/folders/{folder_id}/shared/chats", params=params)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def full_folder(instance) -> Iterator[tuple[httpx.Client, str, list[str]]]:
    """An account's folder holding `FOLDER_SIZE` chats titled in creation order."""
    with create_user(instance).client() as client:
        folder_id = create_folder(client)
        titles = [f"Paging {index:03d}" for index in range(FOLDER_SIZE)]
        for title in titles:
            create_chat(client, {"title": title}, folder_id)
        yield client, folder_id, titles


# --- 409fb39: a folder pages through all of its chats --------------------------------------


def test_the_last_page_holds_the_tail_of_the_folder(full_folder):
    client, folder_id, titles = full_folder

    last_page = folder_page(client, folder_id, page=7, sort_by="title", sort_dir="asc")

    assert [chat["title"] for chat in last_page["chats"]] == titles[60:], (
        "chats past the sixtieth in a folder never appeared (#26786)"
    )
    assert last_page["total"] == FOLDER_SIZE
    assert last_page["has_more"] is False


def test_every_page_together_lists_each_chat_once(full_folder):
    client, folder_id, titles = full_folder

    listed = []
    for page in range(1, 8):
        chats = folder_page(client, folder_id, page=page)["chats"]
        assert len(chats) == min(PAGE_SIZE, FOLDER_SIZE - len(listed))
        listed += [chat["title"] for chat in chats]

    assert sorted(listed) == titles


def test_a_folder_sorts_by_title_both_ways(full_folder):
    client, folder_id, titles = full_folder

    ascending = folder_page(client, folder_id, page=1, sort_by="title", sort_dir="asc")
    descending = folder_page(client, folder_id, page=1, sort_by="title", sort_dir="desc")

    assert [chat["title"] for chat in ascending["chats"]] == titles[:PAGE_SIZE]
    assert [chat["title"] for chat in descending["chats"]] == titles[::-1][:PAGE_SIZE]
    assert ascending["has_more"] is True


def test_pinned_and_archived_chats_are_left_out_of_list_and_count(make_user):
    with make_user().client() as client:
        folder_id = create_folder(client)
        create_chat(client, {"title": "Plain"}, folder_id)
        pinned_id = create_chat(client, {"title": "Pinned"}, folder_id)
        archived_id = create_chat(client, {"title": "Archived"}, folder_id)
        assert client.post(f"/api/v1/chats/{pinned_id}/pin").status_code == 200
        assert client.post(f"/api/v1/chats/{archived_id}/archive").status_code == 200

        first_page = folder_page(client, folder_id, page=1)

    assert [chat["title"] for chat in first_page["chats"]] == ["Plain"]
    assert first_page["total"] == 1


def test_without_a_page_the_folder_still_answers_one_list(full_folder):
    client, folder_id, _titles = full_folder

    unpaged = folder_page(client, folder_id)

    assert len(unpaged["chats"]) == 60
    assert "total" not in unpaged


# --- f1ded94 / a9617ca: a background write leaves the chat where it is in the sidebar -------

LONG_AGO = 1_700_000_000
BACKGROUND_EVENTS = {
    "sources": {"type": "source", "data": {"source": {"name": "doc"}, "document": ["text"]}},
    "files": {"type": "files", "data": {"files": [{"type": "image", "url": "/image.png"}]}},
    "embeds": {"type": "embeds", "data": {"embeds": ["<p>embed</p>"]}},
}


@pytest.fixture
def chat_last_updated_long_ago(make_user) -> Iterator[tuple[httpx.Client, str]]:
    """An imported chat whose `updated_at`, the sidebar's sort key, lies in the past."""
    reply = {"id": "m1", "parentId": None, "childrenIds": [], "role": "assistant", "content": "hi"}
    chat = {"title": "Old", "history": {"currentId": "m1", "messages": {"m1": reply}}}
    with make_user().client() as client:
        imported = client.post(
            "/api/v1/chats/import",
            json={"chats": [{"chat": chat, "created_at": LONG_AGO, "updated_at": LONG_AGO}]},
        )
        assert imported.status_code == 200, imported.text
        yield client, imported.json()[0]["id"]


def send_event(client: httpx.Client, chat_id: str, event: dict) -> dict:
    sent = client.post(f"/api/v1/chats/{chat_id}/messages/m1/event", json=event)
    assert sent.json() is True, sent.text
    stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.mark.parametrize("field", BACKGROUND_EVENTS)
def test_a_background_write_keeps_the_chats_place(chat_last_updated_long_ago, field):
    client, chat_id = chat_last_updated_long_ago

    stored = send_event(client, chat_id, BACKGROUND_EVENTS[field])

    assert stored["chat"]["history"]["messages"]["m1"].get(field), f"{field} was not saved"
    assert stored["updated_at"] == LONG_AGO, f"writing {field} moved the chat up the sidebar"


def test_new_message_content_still_moves_the_chat_up(chat_last_updated_long_ago):
    client, chat_id = chat_last_updated_long_ago

    stored = send_event(client, chat_id, {"type": "message", "data": {"content": " more"}})

    assert stored["chat"]["history"]["messages"]["m1"]["content"] == "hi more"
    assert stored["updated_at"] > LONG_AGO


# --- cc9a445: PostgreSQL search reads where current conversations keep their text ---------


@pytest.fixture(scope="module")
def postgres_instance(mock_upstream, tmp_path_factory) -> Iterator[LaunchedInstance]:
    """A scratch instance on an embedded PostgreSQL."""
    pgserver = pytest.importorskip("pgserver", reason="needs pgserver (pip install pgserver)")
    server = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
    try:
        yield from launch(mock_upstream, {"DATABASE_URL": server.get_uri()})
    finally:
        server.cleanup()


@pytest.fixture(scope="module")
def postgres_client(postgres_instance) -> Iterator[httpx.Client]:
    with create_user(postgres_instance).client() as client:
        yield client


def history_with(content: str) -> dict:
    entry = {"id": "m1", "parentId": None, "childrenIds": [], "role": "user", "content": content}
    return {"currentId": "m1", "messages": {"m1": entry}}


def search(client: httpx.Client, text: str) -> list[str]:
    response = client.get("/api/v1/chats/search", params={"text": text})
    assert response.status_code == 200, response.text
    return [chat["id"] for chat in response.json()]


@pytest.mark.slow
@pytest.mark.requires_postgres
@pytest.mark.parametrize(
    "query", ["{word}", "{WORD}", "plan {word}"], ids=["word", "other-case", "words-reordered"]
)
def test_postgres_search_finds_text_in_the_history(postgres_client, query):
    word = f"pterodactyl{uuid.uuid4().hex[:8]}"
    chat_id = create_chat(
        postgres_client, {"title": "Untitled", "history": history_with(f"The {word} plan")}
    )

    found = search(postgres_client, query.format(word=word, WORD=word.upper()))

    assert found == [chat_id], "PostgreSQL search never read current conversations"


@pytest.mark.slow
@pytest.mark.requires_postgres
def test_postgres_search_still_finds_legacy_messages_and_nothing_of_other_accounts(
    postgres_instance, postgres_client
):
    word = f"pterodactyl{uuid.uuid4().hex[:8]}"
    legacy_id = create_chat(
        postgres_client, {"title": "Untitled", "messages": [{"role": "user", "content": word}]}
    )
    with create_user(postgres_instance).client() as other_client:
        create_chat(other_client, {"title": "Untitled", "history": history_with(word)})

    assert search(postgres_client, word) == [legacy_id]


# The writes below happen in the background of a live reply, so each is timed to land seconds
# after the last write that may move the chat: a write that moves it shows as a later updated_at.
LATE_BY = 2.5
TASKS_CONFIG = ("/api/v1/tasks/config", "/api/v1/tasks/config/update")
CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")


def stored_chat(client: httpx.Client, chat_id: str) -> dict:
    response = client.get(f"/api/v1/chats/{chat_id}")
    assert response.status_code == 200, response.text
    return response.json()


def wait_for_message(client: httpx.Client, turn: ChatTurn, ready, timeout: float = 30.0) -> dict:
    """The chat once its reply satisfies `ready`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chat = stored_chat(client, turn.chat_id)
        if ready(chat["chat"]["history"]["messages"].get(turn.assistant_message_id, {})):
            return chat
        time.sleep(0.1)
    raise AssertionError(f"the reply never reached the awaited state within {timeout}s")


def is_task_request(body: dict) -> bool:
    return not body.get("stream")


@pytest.fixture
def follow_ups_on(admin, preserve):
    preserve("tasks")
    with admin.client() as client:
        current = client.get(TASKS_CONFIG[0]).json()
        updated = client.post(
            TASKS_CONFIG[1], json={**current, "ENABLE_FOLLOW_UP_GENERATION": True}
        )
    assert updated.status_code == 200, updated.text


def test_saving_follow_ups_keeps_the_chats_place(follow_ups_on, make_user, upstream):
    suggestions = {"follow_ups": ["And tomorrow?"]}
    upstream.queue(
        reply.text("Sunny.", match=reply.answering("weather today")),
        reply.text(json.dumps(suggestions), delay=LATE_BY, match=is_task_request),
    )
    with make_user().client() as client:
        turn = send_message(
            client, "weather today", background_tasks={"follow_up_generation": True}
        )
        answered = wait_for_message(client, turn, lambda message: message.get("done"))
        with_follow_ups = wait_for_message(client, turn, lambda message: message.get("followUps"))

    assert with_follow_ups["updated_at"] == answered["updated_at"], (
        "saving the follow-up suggestions after the reply moved the chat up the sidebar"
    )


@pytest.fixture
def router_model(admin, preserve) -> str:
    """A public router model whose only route is the scripted model."""
    preserve(ROUTER_MODELS_CONFIG)
    router_id = f"router-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        save_router_model(
            client,
            router_id,
            f"Router {router_id}",
            routes=default_routes(MOCK_MODEL_ID),
            access_grants=[EVERYONE_READS],
        )
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return router_id


def test_saving_the_router_pick_keeps_the_chats_place(router_model, make_user, upstream):
    prompt = "pick a contender"
    pieces = ["The ", "pick ", "is ", "made."]
    upstream.queue(
        reply.text(pieces, delay=LATE_BY, chunk_delay=1.0, match=reply.answering(prompt))
    )
    with make_user().client() as client:
        turn = send_message(client, prompt, model=router_model)
        asked = wait_for_message(client, turn, lambda message: message)
        picked = wait_for_message(client, turn, lambda message: message.get("selectedModelId"))

    assert picked["chat"]["history"]["messages"][turn.assistant_message_id].get("done") is not True
    assert picked["updated_at"] == asked["updated_at"], (
        "saving which model the router picked moved the chat up the sidebar mid-reply"
    )


@pytest.fixture
def compaction_on(admin, preserve):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        updated = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_CONTEXT_COMPACTION": True})
    assert updated.status_code == 200, updated.text


def test_compacting_a_chat_keeps_its_place(compaction_on, make_user, upstream):
    question = {"id": "m1", "parentId": None, "childrenIds": ["m2"], "role": "user", "content": "q"}
    answer = {"id": "m2", "parentId": "m1", "childrenIds": [], "role": "assistant", "content": "a"}
    history = {"currentId": "m2", "messages": {"m1": question, "m2": answer}}
    upstream.queue(reply.text("A SUMMARY", match=is_task_request))
    with make_user().client() as client:
        imported = client.post(
            "/api/v1/chats/import",
            json={
                "chats": [{"chat": {"title": "Old", "history": history}, "updated_at": LONG_AGO}]
            },
        )
        assert imported.status_code == 200, imported.text
        chat_id = imported.json()[0]["id"]

        compacted = client.post(f"/api/v1/chats/{chat_id}/compact", json={"model": MOCK_MODEL_ID})
        stored = stored_chat(client, chat_id)

    assert compacted.status_code == 200, compacted.text
    assert compacted.json()["compacted"] is True, compacted.json()
    assert stored["chat"]["history"]["messages"]["m2"]["contextSummary"] == "A SUMMARY"
    assert stored["updated_at"] == LONG_AGO, "compacting the chat moved it up the sidebar"
