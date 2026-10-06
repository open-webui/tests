"""Journey: dictating into the chat input with local Whisper, on a model built on disk.

The chat input's microphone records the browser's own WebM (Opus) and sends it to the server,
which transcribes it with local Whisper: faster-whisper decodes the recording with PyAV (av)
and runs the tiny model of `harness.local_whisper`, whose transcript is made of its `WORDS`.
The transcript lands in the chat input, unsent, also when the admin picks Whisper (Local) in the
Audio tab and loads the model there with Update model. Twin, in the browser, of the local Whisper
tests in integration/deps/test_audio_stack.py.

Discriminates: passes on the dev ef67cc3fa build; in a backend copy whose `av.open` fails, the
recording is never transcribed and the chat input stays empty, as it does when faster-whisper's
`transcribe` is given `beams` for `beam_size` or a segment is read as `txt` for `text`. In one
whose audio settings update ignores `WHISPER_MODEL` the Audio tab test fails (the model is never
loaded).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness.audio_engine import AUDIO_CONFIG, AUDIO_NAMESPACE, CONFIG_IMPORT
from harness.local_whisper import WORDS, save_tiny_whisper, using_local_whisper
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TRANSCRIBE_TIMEOUT_MS = 30_000
TINY_MODEL_WORDS = re.compile("|".join(WORDS))


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory):
    return save_tiny_whisper(tmp_path_factory.mktemp("whisper"))


@pytest.fixture
def local_whisper(admin, e2e_instance, tiny_whisper):
    with admin.client() as client, using_local_whisper(client, tiny_whisper):
        yield


@pytest.fixture
def audio_restored(admin, e2e_instance):
    """Puts the audio settings back through the config import, as `harness.audio_engine` does."""
    with admin.client() as client:
        snapshot = client.get(AUDIO_NAMESPACE)
        snapshot.raise_for_status()
        current = client.get(AUDIO_CONFIG[0]).json()
        # start from an engine the tab moves away from, so picking local Whisper is a change
        stt = {**current["stt"], "ENGINE": "web"}
        client.post(AUDIO_CONFIG[1], json={"tts": current["tts"], "stt": stt}).raise_for_status()
        yield
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
    assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"


def _is_audio_settings_save(response) -> bool:
    return "/api/v1/audio/config/update" in response.url


def dictate(page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice Input").click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()


def test_dictation_puts_the_local_whisper_transcript_in_the_chat_input(
    voice_page_for, make_user, local_whisper
):
    page = voice_page_for(make_user())

    dictate(page)

    expect(chat_input(page)).to_contain_text(TINY_MODEL_WORDS, timeout=TRANSCRIBE_TIMEOUT_MS)


def test_local_whisper_picked_and_loaded_in_the_audio_tab_transcribes_dictation(
    page_for, voice_page_for, make_user, audio_restored, tiny_whisper
):
    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/settings/audio")
    settings = admin_page.get_by_role("dialog")
    settings.get_by_role("combobox", name="Select an engine").select_option(label="Whisper (Local)")
    settings.get_by_role("textbox", name="Set whisper model").fill(str(tiny_whisper))
    with admin_page.expect_response(_is_audio_settings_save) as saved:
        settings.get_by_role("button", name="Update model").click()
    assert saved.value.ok, saved.value.text()
    expect(settings.get_by_role("button", name="Update model")).to_be_enabled()

    page = voice_page_for(make_user())
    dictate(page)

    expect(chat_input(page)).to_contain_text(TINY_MODEL_WORDS, timeout=TRANSCRIBE_TIMEOUT_MS)
