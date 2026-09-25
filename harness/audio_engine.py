"""An OpenAI-shaped speech engine played by a `listener`, and the admin audio settings that use it.

`serve_audio_engine(listener)` answers `/audio/speech` with `engine.speech` (MP3 unless a test
swaps it), `/audio/transcriptions` with `engine.transcript`, and `/audio/models` and
`/audio/voices` with what the admin's voice settings list. `using_audio_engine(client, engine)`
points speech-to-text and text-to-speech at it and restores the previous audio settings on exit.
The restore goes through the admin's config import: saving local Whisper back through the audio
settings loads the Whisper model, which an offline instance cannot.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

AUDIO_CONFIG = ("/api/v1/audio/config", "/api/v1/audio/config/update")
AUDIO_NAMESPACE = "/api/v1/configs/namespace/audio"
CONFIG_IMPORT = "/api/v1/configs/import"
TRANSCRIPT = "the quick brown fox"
SPEECH = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" * 64  # an MP3 as far as we care
SPEECH_MODEL = "tts-local"
VOICE = "narrator"


@dataclass
class AudioEngine:
    listener: Listener
    transcript: str = TRANSCRIPT
    speech: bytes = SPEECH
    speech_type: str = "audio/mpeg"
    voices: dict[str, str] = field(default_factory=lambda: {VOICE: "The Narrator"})
    models: list[str] = field(default_factory=lambda: [SPEECH_MODEL])

    @property
    def base_url(self) -> str:
        return self.listener.base_url

    def speak(self, _request: ReceivedRequest) -> Answer:
        return 200, {"Content-Type": self.speech_type}, self.speech

    def transcribe(self, _request: ReceivedRequest) -> Answer:
        return json_answer({"text": self.transcript})

    def speech_requests(self) -> list[dict]:
        return [request.json() for request in self.listener.requests_to("/audio/speech")]

    def transcription_requests(self) -> list[ReceivedRequest]:
        return self.listener.requests_to("/audio/transcriptions")


def serve_audio_engine(listener: Listener) -> AudioEngine:
    engine = AudioEngine(listener)
    listener.route("POST", "/audio/speech", engine.speak)
    listener.route("POST", "/audio/transcriptions", engine.transcribe)
    listener.route(
        "GET",
        "/audio/models",
        lambda _request: json_answer({"models": [{"id": name} for name in engine.models]}),
    )
    listener.route(
        "GET",
        "/audio/voices",
        lambda _request: json_answer(
            {"voices": [{"id": key, "name": name} for key, name in engine.voices.items()]}
        ),
    )
    return engine


@contextmanager
def using_audio_engine(client: httpx.Client, engine: AudioEngine) -> Iterator[dict]:
    """Save audio settings that use `engine` for both directions; yields what was saved."""
    snapshot = client.get(AUDIO_NAMESPACE)
    snapshot.raise_for_status()
    current = client.get(AUDIO_CONFIG[0])
    current.raise_for_status()
    connection = {"OPENAI_API_BASE_URL": engine.base_url, "OPENAI_API_KEY": "sk-audio"}
    settings = {
        "tts": {
            **current.json()["tts"],
            **connection,
            "ENGINE": "openai",
            "MODEL": SPEECH_MODEL,
            "VOICE": VOICE,
        },
        "stt": {**current.json()["stt"], **connection, "ENGINE": "openai", "MODEL": "whisper-1"},
    }
    saved = client.post(AUDIO_CONFIG[1], json=settings)
    assert saved.status_code == 200, f"saving the audio settings failed: {saved.text}"
    try:
        yield saved.json()
    finally:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
        assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"
