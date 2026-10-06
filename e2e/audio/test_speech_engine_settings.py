"""Journey: the admin sets up a speech engine in the Audio tab, a user hears and dictates with it.

Every test has the admin pick the engine in the admin Audio tab, saved with Save, and then a
regular user does the flow on their own page. Azure AI Speech and MistralAI each read a reply aloud
(Azure takes SSML with the voice and output format, Mistral takes JSON and answers base64 audio)
and each transcribe a dictation (Azure with its locales, Mistral through its transcription route),
all on a local stand-in. On the OpenAI-compatible stand-in: the Response Splitting select decides
whether a reply of two paragraphs with two sentences each is sent as four, two or one speech
request; a user's Speech-to-Text Language reaches the engine with the recording; the admin's
speech-to-text model is the one the engine is asked for; a voice the user picked is dropped when
the admin changes the default voice; and a user who picks the Web API as their own Speech-to-Text
Engine dictates in the browser without the engine being called.

Discriminates: passes on the dev ebc6add67 build; in a frontend copy, the admin tab leaving the
Azure Endpoint URL out of its save turns both Azure tests red, leaving out the Mistral API Base
URL turns both Mistral tests red, leaving out the speech-to-text model turns the model test red,
a response message that ignores the split setting turns the paragraphs and none cases red, the
user's Settings dropping the language turns the language test red, a voice check that always takes
the user's saved voice turns the default voice test red, and dictation that ignores the user's own
Web API engine turns the Web API test red.
"""

from __future__ import annotations

import base64
import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.audio_engine import (
    AUDIO_NAMESPACE,
    CONFIG_IMPORT,
    TRANSCRIPT,
    VOICE,
    serve_audio_engine,
)
from harness.chat_history import seed_chat
from harness.listener import Listener, json_answer
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

AUDIO_CONFIG = ("/api/v1/audio/config", "/api/v1/audio/config/update")
# nothing listens here: a setting the admin tab fails to save sends the call to this dead end
DEAD_END = "http://127.0.0.1:9"
AZURE_KEY = "azure-key-0123456789"
AZURE_VOICE = "de-AT-IngridNeural"
AZURE_FORMAT = "audio-16khz-32kbitrate-mono-mp3"
MISTRAL_KEY = "mistral-key-0123456789"
MISTRAL_VOICE = "storyteller-voice"
PLAYED = """() => {
    const audio = document.getElementById('audioElement');
    return audio.src.startsWith('blob:') && audio.played.length > 0;
}"""
COUNT_SPEECH = """() => {
    window.speechCalls = 0;
    const fetchOriginal = window.fetch;
    window.fetch = (...args) => {
        const url = String(args[0]?.url ?? args[0]);
        if (url.endsWith('/api/v1/audio/speech')) window.speechCalls += 1;
        return fetchOriginal(...args);
    };
}"""
# a recognizer that reports one final phrase 200 ms after it starts
FAKE_RECOGNIZER = """() => {
    class FakeSpeechRecognition {
        start() {
            setTimeout(() => {
                const result = [{transcript: 'said by the browser', confidence: 0.9}];
                result.isFinal = true;
                this.onresult?.({resultIndex: 0, results: [result]});
            }, 200);
        }
        stop() {
            setTimeout(() => this.onend?.(), 0);
        }
    }
    window.SpeechRecognition = FakeSpeechRecognition;
    window.webkitSpeechRecognition = FakeSpeechRecognition;
}"""


def _silent_mp3(frames: int = 20) -> bytes:
    """Half a second of silence as MPEG-1 layer 3 frames; Chromium refuses the harness stub."""
    header = b"\xff\xfb\x90\xc4"
    return b"".join(header + b"\x00" * (417 - 4) for _ in range(frames))


@pytest.fixture
def audio_restored(admin):
    """Puts the audio settings back through the config import, as `harness.audio_engine` does."""
    with admin.client() as client:
        snapshot = client.get(AUDIO_NAMESPACE)
        snapshot.raise_for_status()
    yield
    with admin.client() as client:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
    assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"


@pytest.fixture
def dead_end_urls(admin, audio_restored):
    """Every outside speech endpoint points at a dead end until the admin tab says otherwise."""
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0])
        current.raise_for_status()
        tts, stt = current.json()["tts"], current.json()["stt"]
        tts |= {"AZURE_SPEECH_BASE_URL": DEAD_END, "MISTRAL_API_BASE_URL": DEAD_END}
        # saving local Whisper would load a model, which an offline instance cannot
        stt |= {"ENGINE": "web", "AZURE_BASE_URL": DEAD_END, "MISTRAL_API_BASE_URL": DEAD_END}
        saved = client.post(AUDIO_CONFIG[1], json={"tts": tts, "stt": stt})
    assert saved.status_code == 200, f"saving the audio settings failed: {saved.text}"


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    return page_for(make_user(role="admin"))


def open_admin_audio(page: Page) -> Locator:
    page.goto("/admin/settings/audio")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("combobox", name="Select an engine")).to_be_visible()
    return settings


def save_admin_audio(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def pick_engines(settings: Locator, speech_to_text: str, text_to_speech: str) -> None:
    """The speech-to-text engine first: saving local Whisper would load a model, offline."""
    settings.get_by_role("combobox", name="Select an engine").select_option(label=speech_to_text)
    settings.get_by_role("combobox", name="Select a mode", exact=True).select_option(
        label=text_to_speech
    )


def _is_speech(response) -> bool:
    return response.url.endswith("/api/v1/audio/speech")


def _is_settings_save(response) -> bool:
    return "/user/settings/update" in response.url


def audio_tab(page: Page) -> Locator:
    """The user's Settings > Audio tab, freshly loaded."""
    page.goto("/?settings=audio")
    tab = page.locator("#tab-audio")
    expect(tab.get_by_role("switch", name="Auto-Playback Response")).to_be_visible()
    return tab


def save_user_audio(page: Page, tab: Locator) -> None:
    with page.expect_response(_is_settings_save):
        tab.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def open_answer(page: Page, actor, answer: str) -> None:
    """Opens a seeded chat whose last reply is `answer`; its last paragraph shows when it is up."""
    with actor.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "story?"}, {"role": "assistant", "content": answer}],
        )
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(answer.split("\n\n")[-1])).to_be_visible()


def read_aloud(page: Page, actor, answer: str) -> None:
    open_answer(page, actor, answer)
    with page.expect_response(_is_speech) as spoken:
        page.get_by_role("button", name="Read Aloud").click()
    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(PLAYED)


def dictate(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice Input").click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()


def unique_line() -> str:
    # the server caches speech by its text, so every answer is a test's own
    return f"The keeper lights lamp {uuid.uuid4().hex[:6]}."


def serve_azure(listener: Listener) -> None:
    listener.route(
        "POST", "/cognitiveservices/v1", (200, {"Content-Type": "audio/mpeg"}, _silent_mp3())
    )
    listener.route(
        "POST",
        "/speechtotext/transcriptions:transcribe",
        json_answer({"combinedPhrases": [{"text": TRANSCRIPT}]}),
    )


def serve_mistral(listener: Listener) -> None:
    speech = {"audio_data": base64.b64encode(_silent_mp3()).decode()}
    listener.route("POST", "/audio/speech", json_answer(speech))
    listener.route("POST", "/audio/transcriptions", json_answer({"text": TRANSCRIPT}))


def form_field(body: bytes, name: str) -> bytes | None:
    found = re.search(rb'name="%s"\r\n\r\n(.*?)\r\n--' % name.encode(), body, re.S)
    return found.group(1) if found else None


def test_an_azure_voice_reads_a_reply_aloud_through_the_admin_tab(
    admin_page, page_for, make_user, listener, dead_end_urls
):
    serve_azure(listener)
    settings = open_admin_audio(admin_page)
    pick_engines(settings, "Web API", "Azure AI Speech")
    settings.get_by_role("textbox", name="API Key", exact=True).fill(AZURE_KEY)
    settings.get_by_role("textbox", name="leave blank for eastus").fill("westus")
    settings.get_by_role("textbox", name="commercial endpoint").fill(listener.base_url)
    settings.get_by_role("combobox", name="Select a voice").fill(AZURE_VOICE)
    settings.get_by_role("textbox", name="Select an output format").fill(AZURE_FORMAT)
    save_admin_audio(admin_page, settings)

    reader = make_user()
    answer = unique_line()
    read_aloud(page_for(reader), reader, answer)

    [speech] = listener.requests_to("/cognitiveservices/v1")
    ssml = speech.body.decode()
    assert f'<voice name="{AZURE_VOICE}">{answer}</voice>' in ssml, ssml
    assert 'xml:lang="de-AT"' in ssml, ssml
    assert speech.headers["X-Microsoft-OutputFormat"] == AZURE_FORMAT
    assert speech.headers["Ocp-Apim-Subscription-Key"] == AZURE_KEY


def test_azure_transcribes_a_dictation_through_the_admin_tab(
    admin_page, voice_page_for, make_user, listener, dead_end_urls
):
    serve_azure(listener)
    settings = open_admin_audio(admin_page)
    settings.get_by_role("combobox", name="Select an engine").select_option(label="Azure AI Speech")
    settings.get_by_role("textbox", name="API Key", exact=True).fill(AZURE_KEY)
    settings.get_by_role("textbox", name="leave blank for eastus").fill("westus")
    settings.get_by_role("textbox", name="leave blank for auto-detect").fill("en-US,fr-FR")
    settings.get_by_role("textbox", name="commercial endpoint").fill(listener.base_url)
    save_admin_audio(admin_page, settings)

    page = voice_page_for(make_user())
    page.goto("/")
    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    [sent] = listener.requests_to("/speechtotext/transcriptions:transcribe")
    assert sent.headers["Ocp-Apim-Subscription-Key"] == AZURE_KEY
    definition = form_field(sent.body, "definition")
    assert definition, "the recording went to Azure without a definition"
    assert json.loads(definition)["locales"] == ["en-US", "fr-FR"]


def test_a_mistral_voice_reads_a_reply_aloud_through_the_admin_tab(
    admin_page, page_for, make_user, listener, dead_end_urls
):
    serve_mistral(listener)
    settings = open_admin_audio(admin_page)
    pick_engines(settings, "Web API", "MistralAI")
    settings.get_by_role("textbox", name="API Base URL").fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key", exact=True).fill(MISTRAL_KEY)
    settings.get_by_role("combobox", name="Select a voice").fill(MISTRAL_VOICE)
    save_admin_audio(admin_page, settings)

    reader = make_user()
    answer = unique_line()
    read_aloud(page_for(reader), reader, answer)

    [speech] = listener.requests_to("/audio/speech")
    assert speech.headers["Authorization"] == f"Bearer {MISTRAL_KEY}"
    sent = speech.json()
    assert (sent["input"], sent["voice_id"]) == (answer, MISTRAL_VOICE), sent


def test_mistral_transcribes_a_dictation_through_the_admin_tab(
    admin_page, voice_page_for, make_user, listener, dead_end_urls
):
    serve_mistral(listener)
    settings = open_admin_audio(admin_page)
    settings.get_by_role("combobox", name="Select an engine").select_option(label="MistralAI")
    settings.get_by_role("textbox", name="API Base URL").fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key", exact=True).fill(MISTRAL_KEY)
    settings.get_by_role("textbox", name="voxtral-mini-latest").fill("voxtral-test-model")
    save_admin_audio(admin_page, settings)

    page = voice_page_for(make_user())
    page.goto("/")
    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    [sent] = listener.requests_to("/audio/transcriptions")
    assert sent.headers["Authorization"] == f"Bearer {MISTRAL_KEY}"
    assert form_field(sent.body, "model") == b"voxtral-test-model"


@pytest.mark.parametrize(
    ("split", "label", "parts"),
    [("punctuation", "Punctuation", 4), ("paragraphs", "Paragraphs", 2), ("none", "None", 1)],
)
def test_response_splitting_decides_how_many_speech_requests_a_reply_makes(
    split, label, parts, admin_page, page_for, make_user, speech_engine
):
    speech_engine.speech = _silent_mp3()
    tag = uuid.uuid4().hex[:6]
    sentences = [
        f"The {word} keeper lights the harbour lamp at dusk {tag}."
        for word in ("first", "second", "third", "fourth")
    ]
    answer = f"{sentences[0]} {sentences[1]}\n\n{sentences[2]} {sentences[3]}"
    expected = {
        4: sentences,
        2: [f"{sentences[0]} {sentences[1]}", f"{sentences[2]} {sentences[3]}"],
        1: [f"{sentences[0]} {sentences[1]}\n{sentences[2]} {sentences[3]}"],
    }[parts]
    settings = open_admin_audio(admin_page)
    splitting = settings.get_by_role("combobox").filter(has_text="Paragraphs")
    splitting.select_option(label=label)
    save_admin_audio(admin_page, settings)

    reader = make_user()
    page = page_for(reader)
    page.add_init_script(f"({COUNT_SPEECH})()")
    open_answer(page, reader, answer)
    page.get_by_role("button", name="Read Aloud").click()
    page.wait_for_function(f"() => window.speechCalls >= {parts}")
    page.wait_for_function(PLAYED)

    sent = [request["input"] for request in speech_engine.speech_requests()]
    assert sent == expected, f"{split} splitting sent {sent}"


def test_the_users_speech_to_text_language_goes_to_the_engine_with_the_recording(
    voice_page_for, make_user, speech_engine
):
    page = voice_page_for(make_user())
    tab = audio_tab(page)
    tab.get_by_role("textbox", name="Speech-to-Text Language").fill("fr")
    save_user_audio(page, tab)
    page.goto("/")

    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    [sent] = speech_engine.transcription_requests()
    assert form_field(sent.body, "language") == b"fr"


def test_the_admins_speech_to_text_model_is_the_one_the_engine_is_asked_for(
    admin_page, voice_page_for, make_user, listener, audio_restored
):
    engine = serve_audio_engine(listener)
    settings = open_admin_audio(admin_page)
    settings.get_by_role("combobox", name="Select an engine").select_option(label="OpenAI")
    settings.get_by_role("textbox", name="API Base URL").fill(engine.base_url)
    settings.get_by_role("textbox", name="API Key", exact=True).fill("sk-audio")
    settings.get_by_role("combobox", name="Select a model").fill("whisper-test-model")
    save_admin_audio(admin_page, settings)

    page = voice_page_for(make_user())
    page.goto("/")
    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    [sent] = engine.transcription_requests()
    assert form_field(sent.body, "model") == b"whisper-test-model"


def test_a_new_admin_default_voice_replaces_the_voice_a_user_picked(
    admin_page, page_for, make_user, speech_engine
):
    speech_engine.speech = _silent_mp3()
    speech_engine.voices |= {"storyteller": "The Storyteller", "newscaster": "The Newscaster"}
    owner = make_user()
    page = page_for(owner)
    tab = audio_tab(page)
    tab.get_by_role("combobox", name="Voice").fill("storyteller")
    save_user_audio(page, tab)
    first = unique_line()
    read_aloud(page, owner, first)

    settings = open_admin_audio(admin_page)
    settings.get_by_role("combobox", name="Select a voice").fill("newscaster")
    save_admin_audio(admin_page, settings)
    second = unique_line()
    read_aloud(page, owner, second)

    voices = {request["input"]: request["voice"] for request in speech_engine.speech_requests()}
    assert voices[first] == "storyteller", "the user's own voice was not used"
    assert voices[second] == "newscaster", (
        f"the reply was read in {voices[second]!r} after the admin changed the default voice"
    )
    assert VOICE not in voices.values()


def test_a_users_web_api_engine_dictates_without_calling_the_server_engine(
    voice_page_for, make_user, speech_engine
):
    page = voice_page_for(make_user())
    page.add_init_script(f"({FAKE_RECOGNIZER})()")
    tab = audio_tab(page)
    tab.get_by_role("combobox", name="Speech-to-Text Engine").select_option(label="Web API")
    save_user_audio(page, tab)
    page.goto("/")

    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice Input").click()

    expect(chat_input(page)).to_contain_text("said by the browser")
    assert speech_engine.transcription_requests() == []
