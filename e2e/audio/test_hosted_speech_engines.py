"""Journey: ElevenLabs and Deepgram set up in the admin's Audio tab, used from the chat.

Both are reached at their real host names through the proxy of `harness.hosted_speech`, on an
instance of their own. The admin picks ElevenLabs for text-to-speech and fills its key, voice and
model; a user's Settings > Audio then offers ElevenLabs' own voices, and Read Aloud speaks a reply
in the voice the user picked, with the admin's model, and plays it. The admin picks Deepgram for
speech-to-text with its key and model; a user who set a Speech-to-Text Language dictates into the
chat input and Deepgram's transcript lands there, asked for in that language. When Deepgram
refuses the recording, the error the user sees should give Deepgram's reason; on dev ebc6add67 it
gives only Deepgram's status ("401, message='Unauthorized'"), so that test is red
(open-webui/open-webui#32009). Twin, in the
browser, of integration/audio/test_hosted_speech_engines.py.

Discriminates: passes on the dev ebc6add67 build; in a backend copy whose `_tts_elevenlabs` sends
the admin's voice in place of the one asked for the ElevenLabs test fails, and in one whose
`_transcribe_deepgram` leaves out the language the Deepgram test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import admin_of, create_user
from harness.audio_engine import AUDIO_CONFIG, TRANSCRIPT
from harness.chat_history import seed_chat
from harness.hosted_speech import serving_speech_hosts, speech_hosts_env
from harness.listener import json_answer
from utils.chat_ui import chat_input, conversation
from utils.speech_audio import silent_wav

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ELEVENLABS_KEY = "xi-key-0123456789"
DEEPGRAM_KEY = "deepgram-key-0123456789"
ADMIN_VOICE = "harbour-voice-01"
OWN_VOICE = "gull-voice-02"
MODEL = "eleven_multilingual_v2"
PLAYED = """() => {
    const audio = document.getElementById('audioElement');
    return audio.src.startsWith('blob:') && audio.played.length > 0;
}"""


@pytest.fixture(scope="module")
def speech_hosts():
    with serving_speech_hosts() as proxy:
        yield proxy


@pytest.fixture(scope="module")
def hosted(instance_with, speech_hosts):
    instance = instance_with(speech_hosts_env(speech_hosts))
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return instance


@pytest.fixture
def services(hosted, speech_hosts):
    """Both engines answering afresh, and the admin's audio settings on neither of them."""
    speech_hosts.reset()
    speech_hosts.listener.route(
        "POST", "/v1/text-to-speech/*", (200, {"Content-Type": "audio/mpeg"}, silent_wav(0.5))
    )
    with admin_of(hosted).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        tts = {**current["tts"], "ENGINE": "", "API_KEY": "", "VOICE": "", "MODEL": ""}
        stt = {**current["stt"], "ENGINE": "web", "DEEPGRAM_API_KEY": "", "MODEL": ""}
        saved = client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": stt})
    assert saved.status_code == 200, saved.text
    return speech_hosts.listener


def audio_settings(page: Page) -> Locator:
    """Admin Settings > Audio, freshly loaded from the server."""
    page.goto("/admin/settings/audio")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("combobox", name="Select a mode", exact=True)).to_be_visible()
    return settings


def save(page: Page, locator: Locator) -> None:
    locator.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def users_audio_tab(page: Page) -> Locator:
    page.goto("/?settings=audio")
    tab = page.locator("#tab-audio")
    expect(tab.get_by_role("switch", name="Auto-Playback Response")).to_be_visible()
    return tab


def dictate(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice Input").click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()


def test_elevenlabs_from_the_audio_tab_reads_a_reply_in_the_users_own_voice(
    page_for, hosted, services
):
    admin_page = page_for(create_user(hosted, role="admin"))
    settings = audio_settings(admin_page)
    settings.get_by_role("combobox", name="Select a mode", exact=True).select_option(
        label="ElevenLabs"
    )
    settings.get_by_role("textbox", name="API Key").fill(ELEVENLABS_KEY)
    settings.get_by_role("combobox", name="Select a voice").fill(ADMIN_VOICE)
    settings.get_by_role("combobox", name="Select a model").fill(MODEL)
    save(admin_page, settings)

    owner = create_user(hosted)
    page = page_for(owner)
    tab = users_audio_tab(page)
    voice_box = tab.get_by_role("combobox", name="Voice")
    expect(voice_box).to_have_value(ADMIN_VOICE)
    expect(tab.locator(f"#voice-list option[value='{OWN_VOICE}']")).to_have_count(1)
    voice_box.fill(OWN_VOICE)
    save(page, tab)

    answer = f"The gulls are back on the pier {uuid.uuid4().hex[:6]}."
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "gulls?"}, {"role": "assistant", "content": answer}],
        )
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(answer)).to_be_visible()
    page.get_by_role("button", name="Read Aloud").click()

    page.wait_for_function(PLAYED)
    [request] = services.requests_to(f"/v1/text-to-speech/{OWN_VOICE}")
    assert request.headers.get("xi-api-key") == ELEVENLABS_KEY
    assert (request.json()["text"], request.json()["model_id"]) == (answer, MODEL)


def test_deepgram_from_the_audio_tab_transcribes_dictation_in_the_users_language(
    page_for, voice_page_for, hosted, services
):
    admin_page = page_for(create_user(hosted, role="admin"))
    settings = audio_settings(admin_page)
    settings.get_by_role("combobox", name="Select an engine").select_option(label="Deepgram")
    settings.get_by_role("textbox", name="API Key").fill(DEEPGRAM_KEY)
    settings.get_by_placeholder("Select a model (optional)").fill("nova-3")
    save(admin_page, settings)

    page = voice_page_for(create_user(hosted))
    tab = users_audio_tab(page)
    tab.get_by_role("textbox", name="Speech-to-Text Language").fill("de")
    save(page, tab)
    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    expect(conversation(page).locator(".chat-user")).to_have_count(0)
    [request] = services.requests_to("/v1/listen")
    assert request.headers.get("Authorization") == f"Token {DEEPGRAM_KEY}"
    assert "model=nova-3" in request.path and "language=de" in request.path, request.path


def test_a_dictation_deepgram_refuses_shows_deepgrams_reason(voice_page_for, hosted, services):
    with admin_of(hosted).client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        stt = {**current["stt"], "ENGINE": "deepgram", "DEEPGRAM_API_KEY": DEEPGRAM_KEY}
        client.post(AUDIO_CONFIG[1], json={"tts": current["tts"], "stt": stt}).raise_for_status()
    refusal = {"err_code": "INVALID_AUTH", "error": "Invalid credentials."}
    services.route("POST", "/v1/listen", json_answer(refusal, 401))
    page = voice_page_for(create_user(hosted))

    dictate(page)

    expect(
        page.get_by_text("Invalid credentials.", exact=False),
        "the error shown gives Deepgram's status but drops the reason Deepgram gave "
        "(open-webui/open-webui#32009)",
    ).to_be_visible()
    expect(chat_input(page)).to_have_text("")
