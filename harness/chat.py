"""Send chat messages the way the web client does, and wait for the stored reply.

The client posts the user message with a fresh assistant message id and a socket session id;
the server creates or extends the chat, answers `{"chat_id", "task_ids"}` at once and streams
the reply into the database in the background. `wait_for_reply` reads that stored message, so
a test asserts on what a user reloading the chat would see.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import httpx

from harness.upstream import MOCK_MODEL_ID

NO_BACKGROUND_TASKS = {
    "title_generation": False,
    "tags_generation": False,
    "follow_up_generation": False,
}


@dataclass
class ChatTurn:
    chat_id: str
    user_message_id: str
    assistant_message_id: str


def send_message(
    client: httpx.Client,
    content: str,
    *,
    model: str = MOCK_MODEL_ID,
    chat_id: str | None = None,
    parent_id: str | None = None,
    history: list[dict] | None = None,
    **extra,
) -> ChatTurn:
    """Start a chat (no `chat_id`) or continue one after the assistant message `parent_id`."""
    user_message_id, assistant_message_id = str(uuid.uuid4()), str(uuid.uuid4())
    user_message = {
        "id": user_message_id,
        "parentId": parent_id,
        "childrenIds": [assistant_message_id],
        "role": "user",
        "content": content,
        "models": [model],
        "timestamp": int(time.time()),
    }
    payload = {
        "model": model,
        "messages": [*(history or []), {"role": "user", "content": content}],
        "stream": True,
        "parent_id": parent_id,
        "id": assistant_message_id,
        "user_message": user_message,
        "session_id": f"harness-{uuid.uuid4().hex[:8]}",
        "background_tasks": NO_BACKGROUND_TASKS,
        **extra,
    }
    if chat_id:
        payload["chat_id"] = chat_id
    accepted = client.post("/api/chat/completions", json=payload)
    if accepted.status_code != 200:
        raise AssertionError(f"chat request refused: HTTP {accepted.status_code} {accepted.text}")
    return ChatTurn(accepted.json()["chat_id"], user_message_id, assistant_message_id)


def wait_for_reply(client: httpx.Client, turn: ChatTurn, timeout: float = 60.0) -> dict:
    """The stored assistant message once the server has marked it done."""
    deadline = time.monotonic() + timeout
    message: dict = {}
    while time.monotonic() < deadline:
        stored = client.get(f"/api/v1/chats/{turn.chat_id}")
        stored.raise_for_status()
        message = stored.json()["chat"]["history"]["messages"].get(turn.assistant_message_id, {})
        if message.get("done"):
            return message
        time.sleep(0.1)
    raise AssertionError(f"the reply never finished; last stored state: {message}")


def ask(
    client: httpx.Client, content: str, timeout: float = 60.0, **options
) -> tuple[ChatTurn, dict]:
    turn = send_message(client, content, **options)
    return turn, wait_for_reply(client, turn, timeout)
