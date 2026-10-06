"""Journey: reading a message aloud with the OpenAI Realtime text-to-speech engine.

Admin Settings > Audio offers OpenAI Realtime as a text-to-speech engine. Read aloud then opens
a Realtime session at `<base URL>/realtime?model=...` with the key, sets the voice and a renderer
prompt (the admin's Prompt Template, else the built-in one that reads text verbatim), asks for
one audio response reading the text and returns the 24 kHz PCM it got as a WAV file. The voice
picker lists the Realtime voices and models without asking the provider. Same text, voice and
prompt is served from the cache; a provider error is a 502 that never repeats the provider's
words. Here the provider is `harness.realtime_provider`.

Discriminates: in a backend copy, the admin's prompt template ignored for the default turns the
template test red, the speech cache keyed without the prompt turns the cache test red, and the
provider's error passed on turns the error test red.
"""

from __future__ import annotations

import io
import uuid
import wave

import pytest

from harness.audio_engine import AUDIO_CONFIG
from harness.realtime_provider import (
    API_KEY,
    PROVIDER_ERROR_TEXT,
    SPOKEN_PCM,
    VOICE,
    restoring_audio_settings,
    saveable_stt,
    serving_realtime_provider,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SPEECH_MODEL = "gpt-realtime-2.1-mini"
DEFAULT_PROMPT_START = "You are a text-to-speech renderer."
REALTIME_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]


@pytest.fixture
def provider():
    with serving_realtime_provider() as fake:
        yield fake


@pytest.fixture
def save_speech(admin, provider):
    """Saves the realtime text-to-speech engine with the given changes, restored afterwards."""
    with admin.client() as client, restoring_audio_settings(client):

        def save(**changes) -> None:
            current = client.get(AUDIO_CONFIG[0]).json()
            tts = {
                **current["tts"],
                "ENGINE": "openai-realtime",
                "OPENAI_API_BASE_URL": provider.base_url,
                "OPENAI_API_KEY": API_KEY,
                "MODEL": SPEECH_MODEL,
                "VOICE": VOICE,
                "REALTIME_TTS_PROMPT_TEMPLATE": None,
                **changes,
            }
            saved = client.post(
                AUDIO_CONFIG[1], json={"tts": tts, "stt": saveable_stt(current["stt"])}
            )
            assert saved.status_code == 200, saved.text

        yield save


def read_aloud(actor, text: str):
    with actor.client() as client:
        return client.post("/api/v1/audio/speech", json={"input": text})


def test_read_aloud_is_the_providers_speech_as_a_wav(save_speech, provider, make_user):
    save_speech()
    text = f"The lamp is lit {uuid.uuid4().hex[:8]}."

    spoken = read_aloud(make_user(), text)

    assert spoken.status_code == 200, spoken.text
    assert spoken.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(spoken.content)) as recording:
        assert (recording.getframerate(), recording.getnchannels()) == (24000, 1)
        assert recording.readframes(recording.getnframes()) == SPOKEN_PCM
    call = provider.calls[0]
    assert call.path == f"/v1/realtime?model={SPEECH_MODEL}"
    assert call.headers["authorization"] == f"Bearer {API_KEY}"
    assert call.session["audio"]["output"]["voice"] == VOICE
    assert call.session["instructions"].startswith(DEFAULT_PROMPT_START)
    asked = call.received("response.create")[0]["response"]
    assert asked["input"][0]["content"] == [{"type": "input_text", "text": text}]
    assert provider.spoken == [text]


def test_the_admins_prompt_template_is_the_renderer_prompt(save_speech, provider, make_user):
    prompt = "Read the text slowly, like a lighthouse keeper."
    save_speech(REALTIME_TTS_PROMPT_TEMPLATE=prompt)

    spoken = read_aloud(make_user(), f"Mind the rocks {uuid.uuid4().hex[:8]}.")

    assert spoken.status_code == 200, spoken.text
    assert provider.calls[0].session["instructions"] == prompt
    assert provider.calls[0].received("response.create")[0]["response"]["instructions"] == prompt


def test_the_same_text_is_spoken_once_until_the_prompt_changes(save_speech, provider, make_user):
    save_speech()
    reader = make_user()
    text = f"Fog on the bay {uuid.uuid4().hex[:8]}."

    first = read_aloud(reader, text)
    again = read_aloud(reader, text)
    calls_before_the_change = len(provider.calls)
    save_speech(REALTIME_TTS_PROMPT_TEMPLATE="Read it in a whisper.")
    after_the_change = read_aloud(reader, text)

    assert first.status_code == again.status_code == after_the_change.status_code == 200
    assert again.content == first.content
    assert calls_before_the_change == 1
    assert len(provider.calls) == 2, "a new prompt was answered from the old recording"


def test_the_voice_picker_lists_realtime_voices_and_models(save_speech, provider, make_user):
    save_speech()
    with make_user().client() as client:
        voices = client.get("/api/v1/audio/voices")
        models = client.get("/api/v1/audio/models")

    assert [voice["id"] for voice in voices.json()["voices"]] == REALTIME_VOICES
    assert [model["id"] for model in models.json()["models"]] == [
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2.1",
    ]
    assert provider.calls == []


def test_a_provider_error_is_a_502_without_its_words(save_speech, provider, make_user):
    save_speech()
    provider.refuse_session = True

    spoken = read_aloud(make_user(), f"Storm warning {uuid.uuid4().hex[:8]}.")

    assert spoken.status_code == 502
    assert spoken.json()["detail"] == (
        "OpenAI Realtime rejected synthesis. Check the model, voice, and key."
    )
    assert PROVIDER_ERROR_TEXT not in spoken.text


def test_without_a_key_read_aloud_asks_for_one(save_speech, provider, make_user):
    save_speech(OPENAI_API_KEY="")

    spoken = read_aloud(make_user(), f"No key {uuid.uuid4().hex[:8]}.")

    assert spoken.status_code == 400
    assert spoken.json()["detail"] == "Configure an OpenAI Realtime API key."
    assert provider.calls == []


def test_empty_text_is_refused(save_speech, provider, make_user):
    save_speech()

    spoken = read_aloud(make_user(), "   ")

    assert spoken.status_code == 400
    assert spoken.json()["detail"] == "Speech input must be nonempty text."
