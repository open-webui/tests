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

The answer counts as played once an audio element of the page has played speech fetched by it,
retargeted for da850248d, which gave each chat an audio element of its own.

Both tests run in the main chat and in a side chat (`utils/chat_surface.py`): the saved chat
beside a new chat and dragged into a tab beside a saved one. A side chat answer had no Read Aloud
(open-webui/open-webui#34014, fix PR open-webui/open-webui#34064), so the side chat variants are
red on dev c15f28649.
"""

from __future__ import annotations

import dataclasses

import pytest
from playwright.sync_api import expect

from harness.actors import admin_of, create_user
from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.local_voices import local_voices_env, save_tiny_speech, speaker
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT, ChatOpener
from utils.speech_audio import READ_ALOUD_PLAYED

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The keeper lights the lamp at dusk."
FIRST_SPEAKER = speaker(12)


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


@pytest.fixture
def speaking_chat_for(chat_for, speaking, preserve) -> ChatOpener:
    """`chat_for` on the speaking instance, with side chats allowed there."""
    admin = admin_of(speaking)
    if chat_for.place != MAIN_CHAT:
        preserve("permissions", on=speaking)
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["chat"].update(side_chat=True)
            saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text
    return dataclasses.replace(chat_for, upstream=speaking.upstream, admin=admin)


def _is_speech(response) -> bool:
    return response.url.endswith("/api/v1/audio/speech")


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_an_answer_is_read_aloud_by_the_local_voice(speaking_chat_for, speaking):
    """In a side chat: open-webui/open-webui#34014."""
    owner = create_user(speaking)
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "lamp?"}, {"role": "assistant", "content": ANSWER}],
        )
    chat = speaking_chat_for(owner, chat_id=chat_id)
    page = chat.page
    expect(chat.conversation.get_by_text(ANSWER)).to_be_visible()

    with page.expect_response(_is_speech) as spoken:
        chat.panel.get_by_role("button", name="Read Aloud").click()

    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(READ_ALOUD_PLAYED)
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


@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_the_speaker_named_in_the_audio_tab_reads_the_answer(page_for, speaking_chat_for, speaking):
    """In a side chat: open-webui/open-webui#34014."""
    answer = "the ferry leaves the harbour at noon."
    named_speaker = speaker(40)
    owner = create_user(speaking)
    try:
        with owner.client() as client:
            first_speaker_says = _speak(client, answer)
            _set_speaker(speaking, named_speaker)
            named_speaker_says = _speak(client, answer)
            _set_speaker(speaking, FIRST_SPEAKER)
            chat_id, _ = seed_chat(
                client,
                [{"role": "user", "content": "ferry?"}, {"role": "assistant", "content": answer}],
            )
        admin_page = page_for(create_user(speaking, role="admin"), path="/admin/settings/audio")
        settings = admin_page.get_by_role("dialog")
        model_box = settings.get_by_role("combobox", name="TTS Model")
        expect(model_box).to_have_value(FIRST_SPEAKER)
        model_box.fill(named_speaker)
        settings.get_by_role("button", name="Save", exact=True).click()
        expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

        chat = speaking_chat_for(owner, chat_id=chat_id)
        page = chat.page
        expect(chat.conversation.get_by_text(answer)).to_be_visible()
        with page.expect_response(_is_speech) as spoken:
            chat.panel.get_by_role("button", name="Read Aloud").click()
        assert spoken.value.ok, spoken.value.text()
        page.wait_for_function(READ_ALOUD_PLAYED)
        with owner.client() as client:
            # the same body again is answered from the cache: the speech the browser played
            replayed = client.post(
                "/api/v1/audio/speech",
                content=spoken.value.request.post_data_buffer,
                headers={"Content-Type": "application/json"},
            )
    finally:
        _set_speaker(speaking, FIRST_SPEAKER)

    assert replayed.status_code == 200, replayed.text
    assert first_speaker_says != named_speaker_says, "the two speakers sound alike"
    assert replayed.content == named_speaker_says, "the answer was not read by the speaker named"
