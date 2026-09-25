"""Dependency smoke: speech-to-text and text-to-speech through an OpenAI-compatible engine.

python-mimeparse decides which uploads count as audio, aiofiles writes the upload, the speech
the engine returns and the cache that answers a repeated request, and pydub transcodes speech
that does not come back as MP3 (through ffmpeg, so that test skips on a host without it). The
engine is the audio stand-in of `harness/audio_engine.py`, saved into the shared instance's audio
settings for each test.

Discriminates: passes on dev ac00d40e3; in a backend copy, `strict_match_mime_type` taking the
first supported type without `mimeparse.best_match` lets the text upload through, skipping the
cache lookup in `speech` asks the engine twice and dropping the `aiofiles` write of the speech
serves an empty file. The pydub test is unproven on a host without ffmpeg.
"""

from __future__ import annotations

import io
import shutil
import uuid
import wave

import numpy
import pytest
import soundfile

from harness.audio_engine import SPEECH, TRANSCRIPT, serve_audio_engine, using_audio_engine

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]


def _ogg_recording() -> bytes:
    """A tenth of a second of silence, Ogg Vorbis like a browser's recording."""
    recording = io.BytesIO()
    silence = numpy.zeros(1600, dtype="float32")
    soundfile.write(recording, silence, 16000, format="OGG", subtype="VORBIS")
    return recording.getvalue()


def _wav_speech() -> bytes:
    speech = io.BytesIO()
    with wave.open(speech, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 2400)
    return speech.getvalue()


@pytest.fixture
def engine(admin, listener):
    engine = serve_audio_engine(listener)
    with admin.client() as client, using_audio_engine(client, engine):
        yield engine


@pytest.fixture
def speaker(make_user):
    return make_user()


def _transcribe(speaker, filename: str, content: bytes, content_type: str):
    with speaker.client() as client:
        return client.post(
            "/api/v1/audio/transcriptions", files={"file": (filename, content, content_type)}
        )


def _speak(speaker, text: str):
    with speaker.client() as client:
        return client.post("/api/v1/audio/speech", json={"input": text, "voice": "alloy"})


def test_an_ogg_recording_is_written_and_transcribed(speaker, engine):
    recording = _ogg_recording()
    before = len(engine.transcription_requests())

    transcribed = _transcribe(speaker, "recording.ogg", recording, "audio/ogg")

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    sent = engine.transcription_requests()[before:]
    assert len(sent) == 1 and sent[0].headers["Content-Type"].startswith("multipart/form-data")
    assert recording in sent[0].body


def test_a_text_upload_is_not_taken_for_audio(speaker, engine):
    before = len(engine.transcription_requests())

    refused = _transcribe(speaker, "notes.ogg", b"just some text", "text/plain")

    assert refused.status_code == 400, refused.text
    assert len(engine.transcription_requests()) == before


def _speech_requests_for(engine, text: str) -> list:
    requests = engine.listener.requests_to("/audio/speech")
    return [request for request in requests if text.encode() in request.body]


def test_speech_is_asked_for_once_and_then_served_from_the_cache(speaker, engine):
    text = f"read this aloud {uuid.uuid4().hex}"

    first = _speak(speaker, text)
    second = _speak(speaker, text)

    assert first.status_code == 200, first.text
    assert first.content == SPEECH
    assert second.status_code == 200, second.text
    assert second.content == SPEECH
    assert len(_speech_requests_for(engine, text)) == 1, "the cached speech was not reused"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="pydub transcodes through ffmpeg")
def test_wav_speech_is_transcoded_to_mp3(speaker, engine):
    engine.speech, engine.speech_type = _wav_speech(), "audio/wav"

    spoken = _speak(speaker, f"transcode this {uuid.uuid4().hex}")

    assert spoken.status_code == 200, spoken.text
    assert spoken.content[:3] == b"ID3" or spoken.content[:2] in (b"\xff\xfb", b"\xff\xf3")
    assert not spoken.content.startswith(b"RIFF"), "the WAV speech was passed through as is"
