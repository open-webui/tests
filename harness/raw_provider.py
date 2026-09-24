"""A second model connection that answers with exactly the bytes a test scripts.

The scripted provider in `harness.upstream` always emits well-formed chunks. Some regressions
only show on shapes it never produces: a tool call's arguments split across many deltas, a
`usage: null`, a keep-alive comment line, an empty delta, a token count. `connect` adds a
`listener` as an extra OpenAI connection through `harness.second_provider.attach` and publishes
`RAW_MODEL_ID` from it. The caller restores the connections with `preserve(OPENAI_CONFIG)`
taken beforehand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from harness.actors import Actor
from harness.listener import Listener, json_answer
from harness.second_provider import attach

RAW_MODEL_ID = "raw-model"


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
        self.listener.route("POST", "/v1/chat/completions", answer)

    def complete(self, payload: dict) -> None:
        """Answer every chat completion with this JSON body."""
        self.listener.route("POST", "/v1/chat/completions", json_answer(payload))

    def count_tokens(self, input_tokens: int) -> None:
        self.listener.route(
            "POST", "/v1/messages/count_tokens", json_answer({"input_tokens": input_tokens})
        )


def connect(admin: Actor, listener: Listener) -> RawProvider:
    """Add `listener` as a connection serving `RAW_MODEL_ID`, visible to the admin."""
    with admin.client() as client:
        attach(client, listener, RAW_MODEL_ID)
    return RawProvider(listener)
