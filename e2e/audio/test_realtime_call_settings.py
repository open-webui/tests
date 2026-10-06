"""Journey: the Voice calls and OpenAI Realtime speech settings under Admin Settings > Audio.

Voice calls has a Call mode: Standard keeps the provider fields hidden and calls on the classic
transcribe-and-speak loop; Realtime shows the provider's base URL, API key, voice model, voice,
input transcription model and prompt template, with the defaults filled in. Once saved they come
back on the next visit and a user's call opens the provider session with exactly them. Switching
back to Standard hides them again and a call no longer reaches the provider. The Text-to-Speech
engine OpenAI Realtime fills in its own voice and model, offers a Prompt Template in place of
Additional Parameters, and a reply read aloud is rendered with that prompt. Here the provider is
`harness.realtime_provider`.

Discriminates: passes on the dev ebc6add67 build; in a frontend copy, the audio settings saved
without the voice call section turn the Realtime save test red, and saved without the speech
prompt template turn the speech engine test red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.realtime_provider import (
    API_KEY,
    TRANSCRIPTION_MODEL,
    VOICE_MODEL,
    restoring_audio_settings,
    serving_realtime_provider,
    using_realtime,
)
from utils.chat_ui import conversation, last_reply
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def provider():
    with serving_realtime_provider() as fake:
        yield fake


@pytest.fixture
def audio_restored(admin):
    with admin.client() as client, restoring_audio_settings(client):
        yield


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's own settings."""
    return page_for(make_user(role="admin"))


def open_audio_tab(page: Page) -> Locator:
    page.goto("/admin/settings/audio")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    expect(settings.get_by_role("heading", name="Voice calls")).to_be_visible()
    return settings


def section(settings: Locator, title: str) -> Locator:
    return settings.locator("section").filter(has=settings.page.get_by_role("heading", name=title))


def field(scope: Locator, label: str) -> Locator:
    """The input under a field label; these labels are not tied to their inputs."""
    labelled = scope.get_by_text(label, exact=True).locator("xpath=following-sibling::div[1]")
    return labelled.locator("input, textarea")


def is_audio_save(response) -> bool:
    return response.url.endswith("/api/v1/audio/config/update")


def save(page: Page, settings: Locator) -> None:
    # changing the speech engine saves on its own, so wait for this save's answer
    with page.expect_response(is_audio_save) as saved:
        settings.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()


def use_openai_transcription(settings: Locator, base_url: str) -> None:
    # local Whisper would load a model on save, which an offline instance cannot
    settings.get_by_role("combobox", name="Select an engine").select_option(label="OpenAI")
    transcription = section(settings, "Speech-to-Text")
    transcription.get_by_role("textbox", name="API Base URL").fill(base_url)
    transcription.get_by_role("textbox", name="API Key").fill(API_KEY)


def test_realtime_call_mode_saves_its_provider_and_a_users_call_uses_it(
    admin_page, audio_restored, provider, voice_page_for, make_user
):
    voice, prompt = f"lighthouse{uuid.uuid4().hex[:4]}", "Answer like a ferry captain."
    settings = open_audio_tab(admin_page)
    calls = section(settings, "Voice calls")
    expect(calls.get_by_role("combobox", name="Call mode")).to_have_value("false")
    expect(calls.get_by_text("Voice Model", exact=True)).to_have_count(0)
    use_openai_transcription(settings, provider.base_url)

    calls.get_by_role("combobox", name="Call mode").select_option(label="Realtime")

    expect(field(calls, "OpenAI API Base URL")).to_have_value("https://api.openai.com/v1")
    expect(field(calls, "Voice Model")).to_have_value("gpt-realtime-2.1-mini")
    expect(field(calls, "Voice")).to_have_value("marin")
    expect(field(calls, "Input Transcription Model")).to_have_value("gpt-transcribe")
    field(calls, "OpenAI API Base URL").fill(provider.base_url)
    calls.get_by_role("textbox", name="API Key").fill(API_KEY)
    field(calls, "Voice Model").fill(VOICE_MODEL)
    field(calls, "Voice").fill(voice)
    field(calls, "Input Transcription Model").fill(TRANSCRIPTION_MODEL)
    field(calls, "Prompt Template").fill(prompt)
    save(admin_page, settings)

    calls = section(open_audio_tab(admin_page), "Voice calls")
    expect(calls.get_by_role("combobox", name="Call mode")).to_have_value("true")
    expect(field(calls, "OpenAI API Base URL")).to_have_value(provider.base_url)
    expect(field(calls, "Voice")).to_have_value(voice)
    expect(field(calls, "Prompt Template")).to_have_value(prompt)

    page = voice_page_for(make_user())
    start_call(page)
    expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    call = provider.wait_for_call()
    assert call.path == f"/v1/realtime?model={VOICE_MODEL}"
    assert call.headers["authorization"] == f"Bearer {API_KEY}"
    assert call.session["instructions"] == prompt
    assert call.session["audio"]["output"]["voice"] == voice
    assert call.session["audio"]["input"]["transcription"] == {"model": TRANSCRIPTION_MODEL}


def test_standard_call_mode_hides_the_provider_and_calls_skip_it(
    admin_page, admin, provider, speech_engine, voice_page_for, make_user
):
    with admin.client() as client, using_realtime(client, provider):
        settings = open_audio_tab(admin_page)
        calls = section(settings, "Voice calls")
        expect(calls.get_by_role("combobox", name="Call mode")).to_have_value("true")
        expect(field(calls, "Voice Model")).to_have_value(VOICE_MODEL)

        calls.get_by_role("combobox", name="Call mode").select_option(label="Standard")

        expect(calls.get_by_text("Voice Model", exact=True)).to_have_count(0)
        save(admin_page, settings)
        assert client.get(AUDIO_CONFIG[0]).json()["realtime"]["ENABLED"] is False

        page = voice_page_for(make_user())
        start_call(page)
        expect(call_status(page, "Listening...")).to_be_visible(timeout=TURN_TIMEOUT_MS)
        assert not provider.calls, "a standard call reached the realtime provider"


def test_the_realtime_speech_engine_reads_a_reply_with_the_admins_prompt(
    admin_page, audio_restored, provider, page_for, make_user
):
    prompt = f"Read it like a harbour bell, take {uuid.uuid4().hex[:4]}."
    settings = open_audio_tab(admin_page)
    use_openai_transcription(settings, provider.base_url)
    speech = section(settings, "Text-to-Speech")

    speech.get_by_role("combobox", name="Select a mode", exact=True).select_option(
        label="OpenAI Realtime"
    )

    expect(speech.get_by_role("combobox", name="Select a voice")).to_have_value("marin")
    expect(speech.get_by_role("combobox", name="Select a model")).to_have_value(
        "gpt-realtime-2.1-mini"
    )
    expect(speech.get_by_text("Additional Parameters", exact=True)).to_have_count(0)
    speech.get_by_role("textbox", name="API Base URL").fill(provider.base_url)
    speech.get_by_role("textbox", name="API Key").fill(API_KEY)
    field(speech, "Prompt Template").fill(prompt)
    save(admin_page, settings)

    speech = section(open_audio_tab(admin_page), "Text-to-Speech")
    expect(speech.get_by_role("combobox", name="Select a mode", exact=True)).to_have_value(
        "openai-realtime"
    )
    expect(field(speech, "Prompt Template")).to_have_value(prompt)

    reader = make_user()
    answer = f"The ferry leaves at noon from pier {uuid.uuid4().hex[:4]}."
    with reader.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "ferry?"}, {"role": "assistant", "content": answer}],
        )
    page = page_for(reader)
    page.goto(f"/c/{chat_id}")
    expect(last_reply(page)).to_contain_text(answer)
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Read Aloud").last.click()

    provider.wait_for(lambda: answer in provider.spoken, "the reply read aloud")
    assert provider.calls[-1].session["instructions"] == prompt
