"""Journey: Read Aloud with the server's own text-to-speech, on a voice built on disk.

With the `transformers` engine, Read Aloud on an answer asks the server for each sentence, which
speaks it with the SpeechT5 pipeline of `harness.local_voices` and a speaker embedding loaded with
datasets, and the chat's audio element plays what comes back. The speaker the admin names as the
TTS model in the Audio tab is the one that speaks: the answer comes out as that speaker says it
and unlike the speaker named before. Twin, in the browser, of
integration/deps/test_local_text_to_speech.py.

Discriminates: passes on the dev ef67cc3fa build; in a backend copy whose pipeline is built for
text generation the speech request fails and the answer is never played. In a frontend build
whose Audio tab saves an empty TTS model the speaker test fails (the fallback speaker reads it).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.actors import admin_of, create_user
from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.local_voices import local_voices_env, save_tiny_speech, speaker

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The keeper lights the lamp at dusk."
FIRST_SPEAKER = speaker(12)
PLAYED = """() => {
    const audio = document.getElementById('audioElement');
    return audio.src.startsWith('blob:') && audio.played.length > 0;
}"""


@pytest.fixture(scope="module")
def speaking(instance_with, tmp_path_factory):
    instance = instance_with(local_voices_env(save_tiny_speech(tmp_path_factory.mktemp("voice"))))
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with admin_of(instance).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        tts = {**current["tts"], "ENGINE": "transformers", "MODEL": FIRST_SPEAKER}
        stt = {**current["stt"], "ENGINE": "web"}
        client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": stt}).raise_for_status()
    return instance


def _is_speech(response) -> bool:
    return response.url.endswith("/api/v1/audio/speech")


def test_an_answer_is_read_aloud_by_the_local_voice(page_for, speaking):
    owner = create_user(speaking)
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "lamp?"}, {"role": "assistant", "content": ANSWER}],
        )
    page = page_for(owner)
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(ANSWER)).to_be_visible()

    with page.expect_response(_is_speech) as spoken:
        page.get_by_role("button", name="Read Aloud").click()

    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(PLAYED)
    expect(page.get_by_text("Audio playback failed")).to_have_count(0)


def _speak(client, text: str) -> bytes:
    # a body of its own, so the browser's request is never answered from this one's cache
    spoken = client.post("/api/v1/audio/speech", json={"input": text, "voice": "reference"})
    assert spoken.status_code == 200, spoken.text
    return spoken.content


def _set_speaker(instance, name: str) -> None:
    with admin_of(instance).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        tts = {**current["tts"], "MODEL": name}
        client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": current["stt"]}).raise_for_status()


def test_the_speaker_named_in_the_audio_tab_reads_the_answer(page_for, speaking):
    answer = "the ferry leaves the harbour at noon."
    owner = create_user(speaking)
    with owner.client() as client:
        first_speaker_says = _speak(client, answer)
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "ferry?"}, {"role": "assistant", "content": answer}],
        )
    admin_page = page_for(create_user(speaking, role="admin"))
    admin_page.goto("/admin/settings/audio")
    settings = admin_page.get_by_role("dialog")
    model_box = settings.get_by_role("combobox", name="CMU ARCTIC speaker embedding name")
    expect(model_box).to_have_value(FIRST_SPEAKER)
    model_box.fill(speaker(40))
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    try:
        page = page_for(owner)
        page.goto(f"/c/{chat_id}")
        expect(page.get_by_text(answer)).to_be_visible()
        with page.expect_response(_is_speech) as spoken:
            page.get_by_role("button", name="Read Aloud").click()
        assert spoken.value.ok, spoken.value.text()
        page.wait_for_function(PLAYED)
        with owner.client() as client:
            named_speaker_says = _speak(client, answer + " ")
            # the same body again is answered from the cache: the speech the browser played
            replayed = client.post(
                "/api/v1/audio/speech",
                content=spoken.value.request.post_data_buffer,
                headers={"Content-Type": "application/json"},
            )
    finally:
        _set_speaker(speaking, FIRST_SPEAKER)

    assert replayed.status_code == 200, replayed.text
    read_aloud = replayed.content
    assert read_aloud != first_speaker_says, "the answer was read by the speaker named before"
    assert read_aloud == named_speaker_says, "the answer was not read by the speaker named"
