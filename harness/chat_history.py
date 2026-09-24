"""Seed a chat with a history of your choosing, the way an import or an earlier session leaves one.

`POST /api/v1/chats/new` stores the messages as given, `output` items, `usage` and errors
included, and the next turn replays them from the database. So a test can put a stored shape
the live pipeline rarely writes (an unanswered tool call, a failed turn, a stale usage block)
in front of the model and read what the provider is sent.
"""

from __future__ import annotations

import time
import uuid

import httpx

from harness.upstream import MOCK_MODEL_ID


def seed_chat(
    client: httpx.Client, messages: list[dict], model: str = MOCK_MODEL_ID
) -> tuple[str, str]:
    """Store `messages` as one branch, each the parent of the next; returns (chat id, last id)."""
    history: dict[str, dict] = {}
    parent_id = None
    for message in messages:
        message_id = message.get("id") or str(uuid.uuid4())
        history[message_id] = {
            "model": model,
            "timestamp": int(time.time()),
            "done": True,
            **message,
            "id": message_id,
            "parentId": parent_id,
            "childrenIds": [],
        }
        if parent_id:
            history[parent_id]["childrenIds"].append(message_id)
        parent_id = message_id
    chat = {
        "title": "Seeded chat",
        "models": [model],
        "history": {"currentId": parent_id, "messages": history},
    }
    created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, f"seeding the chat failed: {created.text}"
    return created.json()["id"], parent_id
