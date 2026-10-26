"""Journey: a user's own audio settings and the admin's audio tab, on an OpenAI-shaped engine.

The admin's Audio tab points speech and transcription at `harness.audio_engine`, which is what
e2e/admin/test_admin_settings_tabs.py saves through the tab itself. On top of that a user's
Settings > Audio tab: the voice a user picks is the one sent when a reply is read aloud, and the
audio plays; Auto-Playback reads a fresh reply with no tap; dictation from the chat input sends
the recording to the engine and its transcript lands in the chat input, unsent; Instant
Auto-Send sends that transcript as the message. On the admin side, moving either engine to the
browser's Web API hides that engine's controls, and once saved the user's tab and the voice call
button follow.

Since 78613d86f a voice call speaks with a user's Kokoro.js (Browser) voice when the admin set no
server speech engine; before it the call fell back to the browser's own speechSynthesis. The
Kokoro model is a download the suite cannot make, so the page's Kokoro worker is replaced by one
that answers each sentence with silence, and speechSynthesis records what it is asked to say.

Discriminates: passes on the dev 176d31d1d build; in a frontend copy, a read aloud that ignores
the user's saved voice turns the voice test red, a finished reply that never presses its speak
button turns the auto-playback test red, a recording that is never transcribed turns the
dictation test red, a transcript that is not submitted turns the auto-send test red, and the
speech-to-text controls or the chat's call button staying on the page for the Web API turn the
hide test red. The Kokoro
call test passes on dev c31423955 and turns red in a build of it whose call plays its audio only
for a server speech engine, as before 78613d86f (the reply goes to speechSynthesis, no Kokoro
audio plays).
"""

from __future__ import annotations

import base64
import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.audio_engine import AUDIO_CONFIG, SPEECH_MODEL, TRANSCRIPT, VOICE, AudioEngine
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.speech_audio import silent_wav
from utils.voice_call import TURN_TIMEOUT_MS, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OWN_VOICE = "storyteller"
# the page drops a clip's source once it ends, so playback is recorded as it starts
RECORD_PLAYBACK = """
window.playedSources = [];
document.addEventListener('playing', (event) => window.playedSources.push(event.target.src), true);
"""
PLAYED = "() => window.playedSources.some((source) => source.startsWith('blob:'))"


def _silent_mp3(frames: int = 20) -> bytes:
    """Half a second of silence as MPEG-1 layer 3 frames; Chromium refuses the harness stub."""
    header = b"\xff\xfb\x90\xc4"
    return b"".join(header + b"\x00" * (417 - 4) for _ in range(frames))


@pytest.fixture
def playable_speech(speech_engine) -> AudioEngine:
    speech_engine.speech = _silent_mp3()
    return speech_engine


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


def save(page: Page, tab: Locator) -> None:
    with page.expect_response(_is_settings_save):
        tab.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def turn_on(page: Page, tab: Locator, setting: str) -> None:
    """Flip a switch that saves on its own."""
    switch = tab.get_by_role("switch", name=setting)
    with page.expect_response(_is_settings_save):
        switch.click()
    expect(switch).to_be_checked()


def open_answer(page: Page, actor, answer: str) -> None:
    with actor.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "story?"}, {"role": "assistant", "content": answer}],
        )
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(answer)).to_be_visible()


def dictate(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice Input").click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()


def spoken_with(engine: AudioEngine, text: str) -> dict:
    requests = [request for request in engine.speech_requests() if request["input"] == text]
    assert requests, f"{text!r} was never sent to the engine, only {engine.speech_requests()}"
    return requests[-1]


def test_a_users_own_voice_is_the_one_a_reply_is_read_in(page_for, make_user, playable_speech):
    playable_speech.voices[OWN_VOICE] = "The Storyteller"
    owner = make_user()
    page = page_for(owner)
    page.add_init_script(RECORD_PLAYBACK)
    tab = audio_tab(page)
    voice_box = tab.get_by_role("combobox", name="Voice")
    expect(voice_box).to_have_value(VOICE)
    voice_box.fill(OWN_VOICE)
    save(page, tab)
    expect(audio_tab(page).get_by_role("combobox", name="Voice")).to_have_value(OWN_VOICE)

    # the server caches speech by its text, so the answer is this test's own
    answer = f"The keeper lights lamp {uuid.uuid4().hex[:6]}."
    open_answer(page, owner, answer)
    with page.expect_response(_is_speech) as spoken:
        page.get_by_role("button", name="Read Aloud").click()

    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(PLAYED)
    request = spoken_with(playable_speech, answer)
    assert (request["voice"], request["model"]) == (OWN_VOICE, SPEECH_MODEL), request


def test_auto_playback_reads_a_new_reply_without_a_tap(
    page_for, make_user, playable_speech, upstream
):
    page = page_for(make_user())
    page.add_init_script(RECORD_PLAYBACK)
    turn_on(page, audio_tab(page), "Auto-Playback Response")

    question = "what does the keeper do at dusk?"
    answer = f"The keeper lights lamp {uuid.uuid4().hex[:6]}."
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    page.goto("/")
    with page.expect_response(_is_speech) as spoken:
        send(page, question)
        expect_reply(page, answer)

    assert spoken.value.ok, spoken.value.text()
    page.wait_for_function(PLAYED)
    request = spoken_with(playable_speech, answer)
    assert (request["voice"], request["model"]) == (VOICE, SPEECH_MODEL), request


def test_dictation_puts_the_engines_transcript_in_the_chat_input(
    voice_page_for, make_user, speech_engine
):
    page = voice_page_for(make_user())
    dictate(page)

    expect(chat_input(page)).to_contain_text(TRANSCRIPT)
    expect(conversation(page).locator(".chat-user")).to_have_count(0)
    sent = speech_engine.transcription_requests()
    assert len(sent) == 1, f"expected one recording, the engine got {len(sent)}"
    assert b'name="model"' in sent[0].body and b"whisper-1" in sent[0].body


def test_instant_auto_send_sends_the_transcript_as_the_message(
    voice_page_for, make_user, speech_engine, upstream
):
    page = voice_page_for(make_user())
    turn_on(page, audio_tab(page), "Instant Auto-Send After Voice Transcription")
    upstream.queue(reply.text("Foxes noted.", match=reply.answering(TRANSCRIPT)))
    page.goto("/")
    dictate(page)

    expect(conversation(page).locator(".chat-user")).to_contain_text(TRANSCRIPT)
    expect_reply(page, "Foxes noted.")
    expect(chat_input(page)).to_have_text("")


def test_the_web_api_engines_hide_their_controls_for_the_admin_and_the_user(
    page_for, make_user, speech_engine
):
    admin_page = page_for(make_user(role="admin"), path="/admin/settings/audio")
    settings = admin_page.get_by_role("dialog")
    tts_engine = settings.get_by_role("combobox", name="Text-to-Speech Engine")
    stt_engine = settings.get_by_role("combobox", name="Speech-to-Text Engine")
    expect(tts_engine).to_have_value("openai")
    expect(stt_engine).to_have_value("openai")
    expect(settings.get_by_role("textbox", name="API Base URL")).to_have_count(2)
    expect(settings.get_by_role("combobox", name="Select a voice")).to_be_visible()
    expect(settings.get_by_text("Supported MIME Types")).to_be_visible()

    tts_engine.select_option(label="Web API")
    expect(settings.get_by_role("textbox", name="API Base URL")).to_have_count(1)
    expect(settings.get_by_role("combobox", name="Select a voice")).to_have_count(0)
    expect(settings.get_by_role("combobox", name="Select a model")).to_have_count(1)

    stt_engine.select_option(label="Web API")
    expect(settings.get_by_role("textbox", name="API Base URL")).to_have_count(0)
    expect(settings.get_by_role("textbox", name="API Key")).to_have_count(0)
    expect(settings.get_by_role("combobox", name="Select a model")).to_have_count(0)
    expect(settings.get_by_text("Supported MIME Types")).to_have_count(0)
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    page = page_for(make_user())
    tab = audio_tab(page)
    expect(tab.get_by_role("combobox", name="Text-to-Speech Engine")).to_be_visible()
    expect(tab.get_by_role("combobox", name="Speech-to-Text Engine")).to_have_count(0)
    expect(tab.get_by_role("textbox", name="Speech-to-Text Language")).to_have_count(0)
    expect(tab.get_by_text("Allow non-local voices")).to_be_visible()
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    # a call needs server speech-to-text, so the Web STT engine hides the call button (#32970)
    expect(page.get_by_role("button", name="Send message")).to_be_visible()
    expect(page.get_by_role("button", name="Voice mode")).to_have_count(0)


# Kokoro.js runs in a worker that downloads its model; this one answers at once with silence
FAKE_KOKORO = """(speech) => {
    window.kokoro = { asked: [], urls: [], played: [], synthesized: [] };
    const RealWorker = window.Worker;
    const wav = Uint8Array.from(atob(speech), (character) => character.charCodeAt(0));
    class FakeKokoro extends EventTarget {
        answer(data) {
            const event = new MessageEvent('message', { data });
            setTimeout(() => {
                this.onmessage?.(event);
                this.dispatchEvent(event);
            });
        }
        postMessage(message) {
            if (message.type === 'init') this.answer({ status: 'init:complete' });
            if (message.type !== 'generate') return;
            window.kokoro.asked.push(message.payload.text);
            const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
            window.kokoro.urls.push(url);
            this.answer({ status: 'generate:complete', audioUrl: url });
        }
        terminate() {}
    }
    window.Worker = function (url, options) {
        return String(url).includes('kokoro') ? new FakeKokoro() : new RealWorker(url, options);
    };
    speechSynthesis.speak = (utterance) => window.kokoro.synthesized.push(utterance.text);
    document.addEventListener(
        'playing',
        (event) => {
            const source = event.target.src;
            if (window.kokoro.urls.includes(source)) window.kokoro.played.push(source);
        },
        true
    );
}"""


def test_a_call_speaks_with_the_users_kokoro_voice_when_the_server_has_no_speech_engine(
    voice_page_for, make_user, admin, speech_engine, upstream
):
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        current["tts"]["ENGINE"] = ""
        saved = client.post(AUDIO_CONFIG[1], json=current)
    assert saved.status_code == 200, saved.text
    caller = make_user()
    with caller.client() as client:
        kokoro = {"audio": {"tts": {"engine": "browser-kokoro"}}}
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": kokoro})
    assert saved.status_code == 200, saved.text
    speech_engine.transcript = f"tell me about the lamp {uuid.uuid4().hex[:6]}"
    answer = f"The keeper lights lamp {uuid.uuid4().hex[:6]}."
    upstream.queue(reply.text(answer, match=reply.answering(speech_engine.transcript)))
    page = voice_page_for(caller)
    speech = base64.b64encode(silent_wav(1)).decode()
    page.add_init_script(f"({FAKE_KOKORO})({json.dumps(speech)})")
    page.reload()

    start_call(page)

    expect_reply(page, answer)
    page.wait_for_function("() => window.kokoro.played.length > 0", timeout=TURN_TIMEOUT_MS)
    spoken = page.evaluate("() => window.kokoro")
    assert spoken["asked"] == [answer], spoken
    assert spoken["synthesized"] == [], "the browser's own voice spoke the reply"
