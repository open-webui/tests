"""A second model connection that answers with exactly the bytes a test scripts.

The scripted provider in `harness.upstream` always emits well-formed chunks. Some regressions
only show on shapes it never produces: a tool call's arguments split across many deltas, a
`usage: null`, a keep-alive comment line, an empty delta, a token count. `connect` adds a
`listener` as an extra OpenAI connection, the way the admin's Connections page adds one, and
publishes `RAW_MODEL_ID` from it. The caller restores the connections with
`preserve(OPENAI_CONNECTIONS)` taken beforehand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from harness.actors import Actor
from harness.listener import Listener, json_answer

RAW_MODEL_ID = "raw-model"
OPENAI_CONNECTIONS = ("/openai/config", "/openai/config/update")


def sse(*events: dict | str) -> bytes:
    """An event-stream body ending in `[DONE]`; a dict is sent as `data:`, a string verbatim."""
    frames = [event if isinstance(event, str) else f"data: {json.dumps(event)}" for event in events]
    return "".join(f"{frame}\n\n" for frame in [*frames, "data: [DONE]"]).encode()


def chunk(delta: dict, finish_reason: str | None = None, **extra) -> dict:
    """One `chat.completion.chunk` carrying `delta`."""
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {"object": "chat.completion.chunk", "model": RAW_MODEL_ID, "choices": [choice], **extra}


@dataclass
class RawProvider:
    listener: Listener
    model_id: str = RAW_MODEL_ID

    def stream(self, body: bytes) -> None:
        """Answer every chat completion with this event-stream body."""
        answer = (200, {"Content-Type": "text/event-stream"}, body)
        self.listener.route("POST", "/chat/completions", answer)

    def complete(self, payload: dict) -> None:
        """Answer every chat completion with this JSON body."""
        self.listener.route("POST", "/chat/completions", json_answer(payload))

    def count_tokens(self, input_tokens: int) -> None:
        self.listener.route(
            "POST", "/messages/count_tokens", json_answer({"input_tokens": input_tokens})
        )


def connect(admin: Actor, listener: Listener) -> RawProvider:
    """Add `listener` as a connection serving `RAW_MODEL_ID`, visible to the admin."""
    listener.route(
        "GET", "/models", json_answer({"data": [{"id": RAW_MODEL_ID, "object": "model"}]})
    )
    with admin.client() as client:
        current = client.get(OPENAI_CONNECTIONS[0]).json()
        urls = [*current["OPENAI_API_BASE_URLS"], listener.base_url]
        keys = [*current["OPENAI_API_KEYS"], "sk-raw"]
        updated = client.post(
            OPENAI_CONNECTIONS[1],
            json={**current, "OPENAI_API_BASE_URLS": urls, "OPENAI_API_KEYS": keys},
        )
        assert updated.status_code == 200, f"adding the raw connection failed: {updated.text}"
        models = client.get("/api/models").json()["data"]
    assert RAW_MODEL_ID in {model["id"] for model in models}, (
        "the raw connection's model is missing"
    )
    return RawProvider(listener)
