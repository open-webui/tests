"""A voice provider speaking OpenAI's Realtime API over a WebSocket, for realtime voice calls.

With realtime calls on, the browser opens `/api/v1/audio/realtime` on the instance and the
server opens `<base URL>/realtime?model=<voice model>` at the provider with the admin's key,
configures the session and then relays events both ways. `serving_realtime_provider()` yields a
`FakeRealtime` that plays the provider: it opens every connection with `session.created`,
confirms a `session.update` with `session.updated` and records every event it is sent, per
connection, in a `RealtimeCall` (`call.received(kind)`, `call.session`, `call.path`,
`call.headers`).

The fake hears what a test lines up with `fake.hears(transcript)`: once a fifth of a second of
microphone audio arrived it reports speech and then that transcript, the way server-side voice
detection and input transcription do; `fake.mishears()` lines up a turn whose transcription
fails instead. Asked to respond to it, it hands the request to the chat model through the
`generate_chat_completion` function, or with `answers=text` speaks that text itself. Asked to
respond to a function result, it speaks the result's answer; asked for a call status, it speaks
the status sentence; asked to read text (the realtime text-to-speech engine), it speaks that
text. Speaking sends `fake.speech` (a quarter second of silent 24 kHz PCM unless a test sets a
longer one) with its transcript, and `fake.spoken` lists every transcript spoken. A turn lined up
with `gesture=name` (and `answers=text`) plays that gesture through the avatar's `play_animation`
function while speaking the text; the gesture's result lands in `call.gesture_results`, and
asked to go on after it the fake speaks the result's `effect`.

`fake.refuse_handshake = status` turns the next connections away with that HTTP status,
`fake.refuse_session = True` answers the session setup with an error event and `call.drop()`
closes a call from the provider's side. `using_realtime(client, fake)` saves the admin's voice
call settings on the fake (Call mode Realtime) and restores the audio settings on exit through
the config import, as `harness.audio_engine` does. `realtime_call(base_url, token, model_id)`
opens the browser's side of the call against the instance and sends its first-message auth.
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Callable, Iterator

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect
from websockets.sync.server import ServerConnection, serve

from harness.audio_engine import AUDIO_CONFIG, AUDIO_NAMESPACE, CONFIG_IMPORT
from harness.instance import free_port

API_KEY = "sk-realtime"
VOICE_MODEL = "gpt-realtime-harbour"
VOICE = "harbour"
TRANSCRIPTION_MODEL = "transcribe-harbour"
SAMPLE_RATE = 24000
HEARING_BYTES = SAMPLE_RATE * 2 // 5  # a fifth of a second of 16-bit PCM
SPOKEN_PCM = b"\x00\x00" * (SAMPLE_RATE // 4)
# what a provider error carries, which the instance must never pass on
PROVIDER_ERROR_TEXT = "invalid key sk-realtime-secret for instructions"


@dataclass
class Turn:
    transcript: str
    answer: str | None  # None hands the request to the chat model
    fails: bool = False
    gesture: str | None = None
    gestured: bool = False


@dataclass
class RealtimeCall:
    path: str
    headers: dict[str, str]
    connection: ServerConnection = field(repr=False)
    events: list[dict] = field(default_factory=list)
    ended: threading.Event = field(default_factory=threading.Event)
    heard_bytes: int = 0
    inputs: dict[str, Turn] = field(default_factory=dict)
    results: dict[str, str] = field(default_factory=dict)
    gesture_results: list[dict] = field(default_factory=list)

    def received(self, kind: str) -> list[dict]:
        return [event for event in list(self.events) if event.get("type") == kind]

    @property
    def session(self) -> dict:
        updates = self.received("session.update")
        assert updates, "the instance never configured the voice session"
        return updates[0]["session"]

    def send(self, event: dict) -> None:
        self.connection.send(json.dumps(event))

    def drop(self) -> None:
        """Close the call from the provider's side, the way a network drop ends it."""
        self.connection.close(1011, "provider went away")


@dataclass
class FakeRealtime:
    base_url: str
    calls: list[RealtimeCall] = field(default_factory=list)
    turns: deque[Turn] = field(default_factory=deque)
    spoken: list[str] = field(default_factory=list)
    refuse_handshake: int | None = None
    refuse_session: bool = False
    speech: bytes = SPOKEN_PCM
    lock: threading.Lock = field(default_factory=threading.Lock)

    def hears(
        self, transcript: str, answers: str | None = None, gesture: str | None = None
    ) -> None:
        with self.lock:
            self.turns.append(Turn(transcript, answers, gesture=gesture))

    def mishears(self) -> None:
        """The next turn's transcription fails."""
        with self.lock:
            self.turns.append(Turn("", None, fails=True))

    def wait_for(self, condition: Callable[[], bool], what: str, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise AssertionError(f"the voice provider never saw {what}")
            time.sleep(0.05)

    def wait_for_call(self, timeout: float = 20.0) -> RealtimeCall:
        self.wait_for(lambda: bool(self.calls), "a call", timeout)
        return self.calls[-1]

    def _next_turn(self) -> Turn | None:
        with self.lock:
            return self.turns.popleft() if self.turns else None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _speak(fake: FakeRealtime, call: RealtimeCall, metadata: dict, text: str) -> None:
    response_id, item_id = _new_id("resp"), _new_id("item")
    audio = base64.b64encode(fake.speech).decode()
    located = {"response_id": response_id, "item_id": item_id, "content_index": 0}
    call.send({"type": "response.created", "response": {"id": response_id, "metadata": metadata}})
    call.send({"type": "response.output_audio.delta", **located, "delta": audio})
    call.send({"type": "response.output_audio_transcript.delta", **located, "delta": text})
    call.send({"type": "response.output_audio_transcript.done", **located, "transcript": text})
    call.send({"type": "response.output_audio.done", **located})
    with fake.lock:
        fake.spoken.append(text)
    done = {"id": response_id, "status": "completed", "metadata": metadata}
    call.send({"type": "response.done", "response": done})


def _delegate(call: RealtimeCall, metadata: dict, request: str) -> None:
    response_id = _new_id("resp")
    item = {
        "id": _new_id("item"),
        "type": "function_call",
        "status": "completed",
        "name": "generate_chat_completion",
        "call_id": _new_id("call"),
        "arguments": json.dumps({"request": request}),
    }
    call.send({"type": "response.created", "response": {"id": response_id, "metadata": metadata}})
    call.send({"type": "response.output_item.done", "response_id": response_id, "item": item})
    done = {"id": response_id, "status": "completed", "metadata": metadata}
    call.send({"type": "response.done", "response": done})


def _gesture_and_speak(fake: FakeRealtime, call: RealtimeCall, metadata: dict, turn: Turn) -> None:
    """One response that plays `turn.gesture` through `play_animation` while speaking the answer."""
    response_id = _new_id("resp")
    item = {
        "id": _new_id("item"),
        "type": "function_call",
        "status": "completed",
        "name": "play_animation",
        "call_id": _new_id("call"),
        "arguments": json.dumps({"name": turn.gesture}),
    }
    turn.gestured = True
    call.send({"type": "response.created", "response": {"id": response_id, "metadata": metadata}})
    call.send({"type": "response.output_item.done", "response_id": response_id, "item": item})
    item_id = _new_id("item")
    located = {"response_id": response_id, "item_id": item_id, "content_index": 0}
    call.send(
        {
            "type": "response.output_audio.delta",
            **located,
            "delta": base64.b64encode(fake.speech).decode(),
        }
    )
    call.send(
        {"type": "response.output_audio_transcript.done", **located, "transcript": turn.answer}
    )
    call.send({"type": "response.output_audio.done", **located})
    with fake.lock:
        fake.spoken.append(turn.answer)
    done = {"id": response_id, "status": "completed", "metadata": metadata}
    call.send({"type": "response.done", "response": done})


def _hear(fake: FakeRealtime, call: RealtimeCall, audio: str) -> None:
    call.heard_bytes += len(base64.b64decode(audio))
    if call.heard_bytes < HEARING_BYTES:
        return
    turn = fake._next_turn()
    if turn is None:
        return
    call.heard_bytes = 0
    item_id = _new_id("item")
    call.inputs[item_id] = turn
    call.send({"type": "input_audio_buffer.speech_started", "item_id": item_id})
    if turn.fails:
        failed = {"type": "conversation.item.input_audio_transcription.failed", "item_id": item_id}
        call.send({**failed, "content_index": 0, "error": {"message": "inaudible"}})
        return
    call.send(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "content_index": 0,
            "transcript": turn.transcript,
        }
    )


def _respond(fake: FakeRealtime, call: RealtimeCall, response: dict) -> None:
    metadata = response.get("metadata") or {}
    if "input_item_id" in metadata:
        turn = call.inputs[metadata["input_item_id"]]
        if turn.gesture and not turn.gestured:
            _gesture_and_speak(fake, call, metadata, turn)
        elif turn.gesture:
            _speak(fake, call, metadata, call.gesture_results[-1]["effect"])
        elif turn.answer is None:
            _delegate(call, metadata, turn.transcript)
        else:
            _speak(fake, call, metadata, turn.answer)
    elif "call_id" in metadata:
        _speak(fake, call, metadata, call.results[metadata["call_id"]])
    elif response.get("input"):
        _speak(fake, call, metadata, _input_text(response["input"]))
    else:
        _speak(fake, call, metadata, response.get("instructions", ""))


def _input_text(items: list[dict]) -> str:
    """The text a response was asked to read, as text-to-speech sends it."""
    return " ".join(part["text"] for item in items for part in item.get("content", []))


def _answer(fake: FakeRealtime, call: RealtimeCall, event: dict) -> None:
    kind = event.get("type")
    if kind == "session.update":
        if fake.refuse_session:
            error = {"type": "invalid_request_error", "message": PROVIDER_ERROR_TEXT}
            call.send({"type": "error", "error": error})
        else:
            call.send({"type": "session.updated", "session": event["session"]})
    elif kind == "input_audio_buffer.append":
        _hear(fake, call, event["audio"])
    elif kind == "response.create":
        _respond(fake, call, event.get("response") or {})
    elif kind == "conversation.item.create" and event["item"]["type"] == "function_call_output":
        item = event["item"]
        output = json.loads(item["output"])
        if "effect" in output:
            call.gesture_results.append(output)
        else:
            call.results[item["call_id"]] = output.get("answer", "")


@contextlib.contextmanager
def serving_realtime_provider() -> Iterator[FakeRealtime]:
    port = free_port()
    fake = FakeRealtime(base_url=f"http://127.0.0.1:{port}/v1")

    def refuse(connection: ServerConnection, _request):
        if fake.refuse_handshake is not None:
            status = HTTPStatus(fake.refuse_handshake)
            return connection.respond(status, f"{status.phrase}\n")
        return None

    def handle(connection: ServerConnection) -> None:
        request = connection.request
        call = RealtimeCall(
            path=request.path,
            headers={name.lower(): value for name, value in request.headers.raw_items()},
            connection=connection,
        )
        with fake.lock:
            fake.calls.append(call)
        try:
            call.send({"type": "session.created", "session": {"id": _new_id("sess")}})
            for message in connection:
                event = json.loads(message)
                call.events.append(event)
                _answer(fake, call, event)
        except ConnectionClosed:
            pass
        finally:
            call.ended.set()

    server = serve(handle, "127.0.0.1", port, process_request=refuse, compression=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield fake
    finally:
        server.shutdown()


def realtime_settings(fake: FakeRealtime, **overrides) -> dict:
    """The admin's voice call settings on the fake, as the Audio tab saves them."""
    return {
        "ENABLED": True,
        "OPENAI_API_BASE_URL": fake.base_url,
        "OPENAI_API_KEY": API_KEY,
        "MODEL": VOICE_MODEL,
        "VOICE": VOICE,
        "TRANSCRIPTION_MODEL": TRANSCRIPTION_MODEL,
        "REALTIME_CALL_PROMPT_TEMPLATE": None,
        **overrides,
    }


def saveable_stt(stt: dict) -> dict:
    """Speech-to-text settings an offline instance can save: local Whisper moved to OpenAI."""
    # saving local Whisper loads its model, which an offline instance cannot
    return stt if stt["ENGINE"] else {**stt, "ENGINE": "openai"}


def save_realtime(client: httpx.Client, settings: dict) -> dict:
    current = client.get(AUDIO_CONFIG[0])
    current.raise_for_status()
    form = {
        "tts": current.json()["tts"],
        "stt": saveable_stt(current.json()["stt"]),
        "realtime": settings,
    }
    saved = client.post(AUDIO_CONFIG[1], json=form)
    assert saved.status_code == 200, f"saving the voice call settings failed: {saved.text}"
    return saved.json()


@contextlib.contextmanager
def restoring_audio_settings(client: httpx.Client) -> Iterator[None]:
    snapshot = client.get(AUDIO_NAMESPACE)
    snapshot.raise_for_status()
    try:
        yield
    finally:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
        assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"


@contextlib.contextmanager
def using_realtime(client: httpx.Client, fake: FakeRealtime, **overrides) -> Iterator[dict]:
    """Turn realtime calls on with the fake as provider; yields what was saved."""
    with restoring_audio_settings(client):
        yield save_realtime(client, realtime_settings(fake, **overrides))


@contextlib.contextmanager
def realtime_call(
    base_url: str, token: str, model_id: str, chat_id: str | None = None
) -> Iterator[ClientConnection]:
    """The browser's side of a call through the instance, past its first-message auth."""
    url = base_url.replace("http://", "ws://", 1) + "/api/v1/audio/realtime"
    auth = {"type": "auth", "token": token, "model_id": model_id, "chat_id": chat_id}
    with connect(url, proxy=None, open_timeout=10, close_timeout=2) as session:
        session.send(json.dumps(auth))
        yield session


def next_event(session: ClientConnection, kind: str, timeout: float = 10.0) -> dict:
    """The next event of `kind` the instance relays, failing on a `bridge.error` before it."""
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        event = json.loads(session.recv(timeout=remaining))
        if event["type"] == kind:
            return event
        if event["type"] == "bridge.error":
            raise AssertionError(f"the call ended before {kind}: {event['message']}")
    raise AssertionError(f"no {kind} within {timeout} seconds")


def call_error(session: ClientConnection, timeout: float = 10.0) -> str | None:
    """The message of the `bridge.error` that ends the call, or None if it ends without one."""
    deadline = time.monotonic() + timeout
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            event = json.loads(session.recv(timeout=remaining))
            if event["type"] == "bridge.error":
                return event["message"]
    except ConnectionClosed:
        return None
    raise AssertionError(f"the call neither failed nor closed within {timeout} seconds")


def spoken_audio(seconds: float = 0.25) -> str:
    """Microphone audio as the call sends it: base64 16-bit PCM at 24 kHz."""
    return base64.b64encode(b"\x00\x00" * int(SAMPLE_RATE * seconds)).decode()
