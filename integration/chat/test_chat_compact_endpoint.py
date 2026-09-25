"""Journey: compacting a chat on demand, the way the chat menu's compact action does.

`POST /api/v1/chats/{id}/compact` asks a model for a summary of the current branch up to its
last message, stores it on that message and answers how much was dropped along with the
context usage the next turn will be judged by. The next turn then sends the provider the system
prompt with the summary and only what follows the checkpoint. Automatic compaction during a
turn is covered in test_context_compaction.py, which never calls this endpoint.

Discriminates: in a backend copy, storing the summary on the first message of the branch in
place of the current one fails the checkpoint, next-turn and previous-summary tests, and
walking every message of the chat in place of the current branch fails the branch test.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SYSTEM_PROMPT = "You are a careful travel agent."
SUMMARY = "SUMMARY OF THE WHOLE TRIP"


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def summary_prompts(upstream) -> list[str]:
    return [
        body["messages"][0]["content"]
        for body in upstream.chat_requests()
        if is_summary_request(body)
    ]


def turns(count: int) -> list[dict]:
    messages = []
    for number in range(1, count + 1):
        messages.append({"role": "user", "content": f"question {number}"})
        messages.append({"role": "assistant", "content": f"answer {number}"})
    return [{**message, "id": str(uuid.uuid4())} for message in messages]


@pytest.fixture
def compaction_on(admin, preserve):
    """Compaction switched on with a threshold no turn in these tests reaches by itself."""
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        updated = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": 1_000_000,
                "CONTEXT_COMPACTION_TOKEN_CAP": 1_000_000,
            },
        )
    assert updated.status_code == 200, updated.text


def compact(client: httpx.Client, chat_id: str, **form) -> dict:
    answered = client.post(f"/api/v1/chats/{chat_id}/compact", json=form or None)
    assert answered.status_code == 200, answered.text
    return answered.json()


def stored_messages(client: httpx.Client, chat_id: str) -> dict:
    return client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]


def test_compacting_stores_the_summary_on_the_current_message(make_user, upstream, compaction_on):
    history = turns(3)
    upstream.queue(reply.text(SUMMARY, match=is_summary_request))
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, history)
        before = client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]
        result = compact(client, chat_id)
        stored = stored_messages(client, chat_id)

    assert result["compacted"] is True
    assert (result["dropped_messages"], result["kept_messages"]) == (5, 1)
    assert result["summary_chars"] == len(SUMMARY)
    assert result["context_usage"]["tokens"] < before["tokens"]
    assert stored[last_id]["contextSummary"] == SUMMARY
    assert [message_id for message_id, entry in stored.items() if entry.get("contextSummary")] == [
        last_id
    ]
    [prompt] = summary_prompts(upstream)
    compacted_part = prompt.split("### Messages Being Compacted:")[1]
    compacted_part, recent_part = compacted_part.split("### Recent Messages Kept In Context:")
    assert "question 1" in compacted_part and "answer 2" in compacted_part
    assert "answer 3" in recent_part and "question 1" not in recent_part


def test_the_next_turn_sends_the_summary_and_what_follows_it(make_user, upstream, compaction_on):
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, turns(3))
        compact(client, chat_id)
        ask(
            client,
            "question 4",
            chat_id=chat_id,
            parent_id=last_id,
            history=[{"role": "system", "content": SYSTEM_PROMPT}],
        )

    sent = [body for body in upstream.chat_requests() if body.get("stream")][-1]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith(SYSTEM_PROMPT)
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in sent[0]["content"]
    assert [message["content"] for message in sent[1:]] == ["answer 3", "question 4"]


def test_a_second_compaction_builds_on_the_previous_summary(make_user, upstream, compaction_on):
    upstream.queue(
        reply.text(SUMMARY, match=is_summary_request),
        reply.text("answer 4"),
        reply.text("SECOND SUMMARY", match=is_summary_request),
    )
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, turns(3))
        compact(client, chat_id)
        again_at_once = compact(client, chat_id)
        turn, _ = ask(client, "question 4", chat_id=chat_id, parent_id=last_id)
        result = compact(client, chat_id)
        stored = stored_messages(client, chat_id)

    assert again_at_once["compacted"] is False
    assert again_at_once["reason"] == "too_short"
    assert result["compacted"] is True
    assert stored[turn.assistant_message_id]["contextSummary"] == "SECOND SUMMARY"
    second_prompt = summary_prompts(upstream)[-1]
    assert f"### Previous Summary:\n{SUMMARY}" in second_prompt


def test_only_the_current_branch_is_summarized(make_user, upstream, compaction_on):
    history = turns(2)
    upstream.queue(reply.text(SUMMARY, match=is_summary_request))
    with make_user().client() as client:
        chat_id, last_id = seed_chat(client, history)
        # a sibling answer to question 2, left behind when the user went back to the first one
        chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
        messages = chat["history"]["messages"]
        sibling_id = str(uuid.uuid4())
        messages[sibling_id] = {
            **messages[last_id],
            "id": sibling_id,
            "content": "an abandoned answer",
        }
        messages[history[2]["id"]]["childrenIds"].append(sibling_id)
        updated = client.post(f"/api/v1/chats/{chat_id}", json={"chat": chat})
        assert updated.status_code == 200, updated.text
        result = compact(client, chat_id)

    assert result["compacted"] is True
    [prompt] = summary_prompts(upstream)
    assert "answer 2" in prompt
    assert "an abandoned answer" not in prompt


def test_the_model_can_be_chosen_and_must_exist(make_user, upstream, compaction_on):
    upstream.queue(reply.text(SUMMARY, match=is_summary_request))
    with make_user().client() as client:
        chat_id, _ = seed_chat(client, turns(2))
        result = compact(client, chat_id, model=MOCK_MODEL_ID)
        chat_id_without_model, _ = seed_chat(client, turns(2), model="")
        refused = client.post(f"/api/v1/chats/{chat_id_without_model}/compact", json={})

    assert result["compacted"] is True
    assert [body["model"] for body in upstream.chat_requests()] == [MOCK_MODEL_ID]
    assert refused.status_code == 400, refused.text


def test_nothing_is_compacted_while_compaction_is_off(make_user, upstream):
    with make_user().client() as client:
        chat_id, _ = seed_chat(client, turns(3))
        result = compact(client, chat_id)

    assert result == {"ok": True, "compacted": False, "reason": "disabled", "context_usage": None}
    assert upstream.chat_requests() == []


def test_a_single_message_is_too_short_to_compact(make_user, upstream, compaction_on):
    with make_user().client() as client:
        chat_id, _ = seed_chat(client, turns(1)[:1])
        result = compact(client, chat_id)

    assert (result["compacted"], result["reason"]) == (False, "too_short")
    assert upstream.chat_requests() == []


def test_another_users_chat_cannot_be_compacted(make_user, upstream, compaction_on):
    with make_user().client() as owner:
        chat_id, _ = seed_chat(owner, turns(3))
    with make_user().client() as stranger:
        refused = stranger.post(f"/api/v1/chats/{chat_id}/compact", json={})

    assert refused.status_code == 401, refused.text
    assert upstream.chat_requests() == []
