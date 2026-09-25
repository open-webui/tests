"""Journey: voice settings, speech and transcription through an OpenAI-compatible audio engine.

The admin points both directions at the engine the way the audio settings page saves them. A
user's voice picker then lists the engine's own voices and models, reading a message aloud sends
the engine the configured model, the chosen voice and the key, and a recording comes back as the
engine's transcript. Putting the settings back leaves local Whisper selected as before.

Discriminates: in a backend copy, `get_available_voices` answering the built-in OpenAI voices
without asking a custom endpoint, `_tts_openai` leaving out the configured model and
`_transcribe_openai` posting to `/transcriptions` each turn one test red.
"""

from __future__ import annotations

import io
import uuid
import wave

import pytest

from harness.audio_engine import (
    AUDIO_CONFIG,
    SPEECH,
    SPEECH_MODEL,
    TRANSCRIPT,
    VOICE,
    serve_audio_engine,
    using_audio_engine,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def engine(admin, listener):
    engine = serve_audio_engine(listener)
    with admin.client() as client, using_audio_engine(client, engine):
        yield engine


def _recording() -> bytes:
    """A tenth of a second of silence, as a WAV file."""
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return recording.getvalue()


def test_the_voice_picker_lists_the_engines_voices_and_models(engine, make_user):
    with make_user().client() as client:
        voices = client.get("/api/v1/audio/voices")
        models = client.get("/api/v1/audio/models")

    assert voices.status_code == 200, voices.text
    assert voices.json()["voices"] == [{"id": VOICE, "name": "The Narrator"}]
    assert models.status_code == 200, models.text
    assert models.json()["models"] == [{"id": SPEECH_MODEL}]
    assert engine.listener.requests_to("/audio/voices"), "the engine was never asked its voices"


def test_a_message_read_aloud_is_the_engines_speech(engine, make_user):
    text = f"read this aloud {uuid.uuid4().hex}"
    with make_user().client() as client:
        spoken = client.post("/api/v1/audio/speech", json={"input": text, "voice": VOICE})

    assert spoken.status_code == 200, spoken.text
    assert spoken.content == SPEECH
    sent = engine.speech_requests()
    assert sent == [{"input": text, "voice": VOICE, "model": SPEECH_MODEL}], sent
    headers = engine.listener.requests_to("/audio/speech")[0].headers
    assert headers["Authorization"] == "Bearer sk-audio"


def test_a_recording_comes_back_as_the_engines_transcript(engine, make_user):
    recording = _recording()
    with make_user().client() as client:
        transcribed = client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("recording.wav", recording, "audio/wav")},
        )

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    sent = engine.transcription_requests()
    assert len(sent) == 1 and recording in sent[0].body
    assert b'name="model"' in sent[0].body and b"whisper-1" in sent[0].body


def test_putting_the_settings_back_keeps_local_whisper(admin, listener):
    engine = serve_audio_engine(listener)
    with admin.client() as client:
        before = client.get(AUDIO_CONFIG[0]).json()
        with using_audio_engine(client, engine) as saved:
            assert saved["stt"]["ENGINE"] == "openai"
        after = client.get(AUDIO_CONFIG[0]).json()

    assert after == before
