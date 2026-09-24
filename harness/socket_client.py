"""A Socket.IO connection signed in as one account: what a browser tab receives and can send.

`connected(actor)` joins the account's `user:{id}` room the way the web client does, records
every `events` message the server pushes to it, and `call(event, data)` sends an event and
returns once the server's handler has finished.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import socketio

from harness.actors import Actor


@dataclass
class SocketSession:
    client: socketio.Client
    events: list[dict] = field(default_factory=list)

    def call(self, event: str, data: dict) -> None:
        self.client.call(event, data, timeout=30)

    def events_of(self, chat_id: str) -> list[dict]:
        return [entry["data"] for entry in list(self.events) if entry.get("chat_id") == chat_id]

    def wait_for(self, chat_id: str, event_type: str, timeout: float = 30.0, **data) -> dict:
        """The first `event_type` event on the chat whose data carries every `data` item."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.events_of(chat_id):
                payload = event.get("data") if isinstance(event.get("data"), dict) else {}
                if event.get("type") == event_type and data.items() <= payload.items():
                    return event
            time.sleep(0.05)
        raise AssertionError(f"no {event_type} {data} event for chat {chat_id} within {timeout}s")


@contextmanager
def connected(actor: Actor) -> Iterator[SocketSession]:
    session = SocketSession(socketio.Client(reconnection=False))
    session.client.on("events", session.events.append)
    session.client.connect(
        actor.base_url,
        socketio_path="/ws/socket.io",
        auth={"token": actor.token},
        transports=["websocket"],
        wait_timeout=30,
    )
    try:
        yield session
    finally:
        session.client.disconnect()
