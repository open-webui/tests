"""A Socket.IO connection signed in as one account: what a browser tab receives and can send.

`connected(actor)` joins the account's `user:{id}` room the way the web client does, records
every `events` message the server pushes to it, and `call(event, data)` sends an event and
returns once the server's handler has finished. `join_note` and `edit_note` do what the note
editor does with a note's live Yjs document; `note_edit(text)` is the raw update it sends after
typing `text` into an empty note (built with pycrdt, which the backend depends on) and
`note_text(state)` reads back a document state the server sent.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pycrdt
import socketio

from harness.actors import Actor


@dataclass
class SocketSession:
    client: socketio.Client
    events: list[dict] = field(default_factory=list)
    document_states: list[dict] = field(default_factory=list)
    document_updates: list[dict] = field(default_factory=list)

    def call(self, event: str, data: dict) -> None:
        self.client.call(event, data, timeout=30)

    def join_note(self, note_id: str) -> None:
        """Open the note's live document; the server answers with its state."""
        self.call("ydoc:document:join", {"document_id": f"note:{note_id}"})

    def edit_note(self, note_id: str, text: str) -> None:
        """Send the update and content snapshot the editor sends after typing `text`."""
        content = {"md": text, "html": f"<p>{text}</p>", "json": _paragraph_json(text)}
        update = {
            "document_id": f"note:{note_id}",
            "update": note_edit(text),
            "data": {"content": content},
        }
        self.call("ydoc:document:update", update)

    def note_state(self, note_id: str, timeout: float = 30.0) -> str:
        """The note's document, from the first state the server sent this tab for it."""
        return note_text(_first_about(self.document_states, note_id, timeout)["state"])

    def note_update(self, note_id: str, timeout: float = 30.0) -> dict:
        """The first edit to the note that the server passed on from another tab."""
        return _first_about(self.document_updates, note_id, timeout)

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
    session.client.on("ydoc:document:state", session.document_states.append)
    session.client.on("ydoc:document:update", session.document_updates.append)
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


def note_edit(text: str) -> list[int]:
    """The Yjs update of an empty note that gets one paragraph reading `text`."""
    document = pycrdt.Doc()
    fragment = document.get("prosemirror", type=pycrdt.XmlFragment)
    fragment.children.append(pycrdt.XmlElement("paragraph", None, [pycrdt.XmlText(text)]))
    return list(document.get_update())


def note_text(state: list[int]) -> str:
    """The note document a state or update holds, as `<paragraph>...</paragraph>` markup."""
    document = pycrdt.Doc()
    document.apply_update(bytes(state))
    return str(document.get("prosemirror", type=pycrdt.XmlFragment))


def _first_about(arrived: list[dict], note_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for payload in list(arrived):
            if payload.get("document_id") == f"note:{note_id}":
                return payload
        time.sleep(0.05)
    raise AssertionError(f"nothing arrived for note {note_id} within {timeout}s")


def _paragraph_json(text: str) -> dict:
    paragraph = {"type": "paragraph", "content": [{"type": "text", "text": text}]}
    return {"type": "doc", "content": [paragraph]}
