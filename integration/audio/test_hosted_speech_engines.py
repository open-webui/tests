"""Journey: ElevenLabs text-to-speech and Deepgram speech-to-text, at their real host names.

Neither engine's address can be set in the admin's audio settings: Deepgram is always
`api.deepgram.com` and ElevenLabs `api.elevenlabs.io`, so the instance reaches both through the
proxy of `harness.hosted_speech`, which answers for those names. With ElevenLabs picked, the voice
and model lists are ElevenLabs' own, a message read aloud is ElevenLabs' speech in the voice asked
for with the admin's model and key, and a voice ElevenLabs does not list is refused before it is
asked. With Deepgram picked, a recording comes back as Deepgram's transcript, sent with the admin's
key and model and the language asked for, and an error Deepgram gives is passed on. That last
test is red on dev ebc6add67: the transcription keeps only Deepgram's status ("401, message=
'Unauthorized'") and drops the reason Deepgram gives, which the error handler means to show.

Discriminates: the other tests pass on dev ebc6add67; in a backend copy whose `_tts_elevenlabs`
sends no `model_id`, whose ElevenLabs voice list is answered without asking ElevenLabs, or whose
`_transcribe_deepgram` leaves out the language or the model, one test each turns red.
"""

from __future__ import annotations

import io
import uuid
import wave

import pytest

from harness.actors import admin_of, create_user
from harness.audio_engine import AUDIO_CONFIG, SPEECH, TRANSCRIPT
from harness.hosted_speech import (
    ELEVENLABS_MODELS,
    ELEVENLABS_VOICES,
    serving_speech_hosts,
    speech_hosts_env,
)
from harness.listener import json_answer

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

ELEVENLABS_KEY = "xi-key-0123456789"
DEEPGRAM_KEY = "deepgram-key-0123456789"
DEEPGRAM_MODEL = "nova-3"
VOICE = "gull-voice-02"
MODEL = "eleven_multilingual_v2"


def _recording() -> bytes:
    """A tenth of a second of silence, as a WAV file."""
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return recording.getvalue()


@pytest.fixture(scope="module")
def speech_hosts():
    with serving_speech_hosts() as proxy:
        yield proxy


@pytest.fixture(scope="module")
def hosted(instance_with, speech_hosts):
    """An instance whose Deepgram and ElevenLabs calls go through the proxy, both engines picked."""
    instance = instance_with(speech_hosts_env(speech_hosts))
    with admin_of(instance).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        tts = {**current["tts"], "ENGINE": "elevenlabs", "API_KEY": ELEVENLABS_KEY}
        stt = {
            **current["stt"],
            "ENGINE": "deepgram",
            "DEEPGRAM_API_KEY": DEEPGRAM_KEY,
            "MODEL": DEEPGRAM_MODEL,
        }
        saved = client.post(AUDIO_CONFIG[1], json={"tts": {**tts, "MODEL": MODEL}, "stt": stt})
    assert saved.status_code == 200, saved.text
    return instance


@pytest.fixture
def services(speech_hosts):
    """The two engines behind the proxy, with their default answers and nothing recorded."""
    speech_hosts.reset()
    return speech_hosts.listener


def _transcribe(client, language: str | None = None):
    form = {"language": language} if language else None
    return client.post(
        "/api/v1/audio/transcriptions",
        files={"file": ("recording.wav", _recording(), "audio/wav")},
        data=form,
    )


def test_the_voice_picker_lists_the_elevenlabs_voices_and_models(hosted, services):
    with create_user(hosted).client() as client:
        voices = client.get("/api/v1/audio/voices")
        models = client.get("/api/v1/audio/models")

    assert voices.status_code == 200, voices.text
    listed = {voice["id"]: voice["name"] for voice in voices.json()["voices"]}
    assert listed == ELEVENLABS_VOICES
    assert models.status_code == 200, models.text
    assert [model["id"] for model in models.json()["models"]] == list(ELEVENLABS_MODELS)
    asked = services.requests_to("/v1/voices")
    assert asked and asked[0].headers.get("xi-api-key") == ELEVENLABS_KEY


def test_a_message_read_aloud_is_elevenlabs_speech_in_the_voice_asked_for(hosted, services):
    text = f"the gulls are back {uuid.uuid4().hex}"
    with create_user(hosted).client() as client:
        spoken = client.post("/api/v1/audio/speech", json={"input": text, "voice": VOICE})

    assert spoken.status_code == 200, spoken.text
    assert spoken.content == SPEECH
    [request] = services.requests_to(f"/v1/text-to-speech/{VOICE}")
    assert request.headers.get("xi-api-key") == ELEVENLABS_KEY
    assert (request.json()["text"], request.json()["model_id"]) == (text, MODEL)


def test_a_voice_elevenlabs_does_not_list_is_refused_before_asking_it(hosted, services):
    with create_user(hosted).client() as client:
        spoken = client.post(
            "/api/v1/audio/speech", json={"input": f"hello {uuid.uuid4().hex}", "voice": "nobody"}
        )

    assert spoken.status_code == 400, spoken.text
    assert "Invalid voice id" in spoken.text
    assert services.requests_to("/v1/text-to-speech/nobody") == []


def test_a_recording_comes_back_as_the_deepgram_transcript(hosted, services):
    with create_user(hosted).client() as client:
        transcribed = _transcribe(client, language="fr")

    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json()["text"] == TRANSCRIPT
    [request] = services.requests_to("/v1/listen")
    assert request.headers.get("Authorization") == f"Token {DEEPGRAM_KEY}"
    assert f"model={DEEPGRAM_MODEL}" in request.path and "language=fr" in request.path
    assert request.body, "Deepgram never got the recording"


def test_an_error_deepgram_gives_is_passed_on(hosted, services):
    refusal = {"err_code": "INVALID_AUTH", "error": "Invalid credentials."}
    services.route("POST", "/v1/listen", json_answer(refusal, 401))
    with create_user(hosted).client() as client:
        transcribed = _transcribe(client)

    assert transcribed.status_code >= 400
    # the failed response is released before its body is read, so only the status is left
    assert "Invalid credentials." in transcribed.text, (
        f"Deepgram's own error message never reaches the user: {transcribed.text}"
    )
