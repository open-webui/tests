"""A browser tab that answers chat completions for a direct connection, over its socket.

A direct connection is a provider the user added in their own settings: the server never calls
it. For a chat with such a model the server asks the user's tab (`request:chat:completion` on
the tab's socket) to run the completion; the tab acknowledges, forwards every line of the
provider's stream on the channel it was given and ends with `{"done": true}`, or answers a
non-streamed request with the whole result, as `src/routes/+layout.svelte` does.

`answering(actor)` connects such a tab. `tab.stream(*lines)`, `tab.complete(body)` and
`tab.refuse(error)` line up its next answers; `tab.requests` is every completion request the
server made of it (`form_data`, `model`, `channel`). `direct_model(model_id)` is the
`model_item` the web client sends for a model of a direct connection.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from harness.actors import Actor
from harness.socket_client import SocketSession, connected


def direct_model(model_id: str) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "object": "model",
        "owned_by": "openai",
        "connection_type": "external",
        "direct": True,
        "urlIdx": 0,
        "info": {"meta": {"capabilities": {"builtin_tools": True}}},
    }


def chunk_line(delta: dict, finish_reason: str | None = None, **extra) -> str:
    """One SSE line of a chat completion stream, as the tab reads it off the provider."""
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    body = {"object": "chat.completion.chunk", "choices": [choice], **extra}
    return f"data: {json.dumps(body)}"


@dataclass
class DirectTab:
    session: SocketSession
    session_id: str
    requests: list[dict] = field(default_factory=list)
    answers: list[tuple[str, object]] = field(default_factory=list)

    def stream(self, *lines: str | dict) -> None:
        self.answers.append(("stream", lines))

    def complete(self, body: dict) -> None:
        self.answers.append(("complete", body))

    def refuse(self, error: dict) -> None:
        self.answers.append(("refuse", error))

    def handle(self, event: dict) -> dict | None:
        self.session.events.append(event)
        data = event.get("data") or {}
        if data.get("type") != "request:chat:completion":
            return None
        request = data["data"]
        self.requests.append(request)
        kind, answer = self.answers.pop(0) if self.answers else ("stream", ())
        channel = request["channel"]
        try:
            if kind == "stream":
                for line in answer:
                    self.session.client.emit(channel, line)
                return {"status": True}
            return answer
        finally:
            self.session.client.emit(channel, {"done": True})


@contextmanager
def answering(actor: Actor) -> Iterator[DirectTab]:
    with connected(actor) as session:
        tab = DirectTab(session, session.client.get_sid(namespace="/"))
        session.client.on("events", tab.handle)
        yield tab
