"""Regression: Voice mode opened from the URL skipped what the Voice mode button sets up and checks.

A chat opened with `?call=true` (and the desktop app's call shortcut) starts Voice mode without
the Voice mode button, through a path of its own.

Issue open-webui/open-webui#31825, fix 79baaca3b (PR open-webui/open-webui#31826): that path
applied none of the button's checks, so a user without the Allow Call permission, a chat with
two models or an instance on the Web API speech-to-text engine got a call anyway. It now applies
the same three checks, with the button's messages.

Issue open-webui/open-webui#31828, fix 51da3eefc (PR open-webui/open-webui#31829): with Kokoro.js
(Browser) as the speech engine only the button set up the Kokoro voice, so a call from the URL
showed replies and never spoke them. The call now sets the voice up itself when it opens. The
Kokoro model is a download the suite cannot make, so here the model request is answered with a
404: the call asks for the model, shows the loading error and stays open. A spoken reply is not
covered.

Red on dev ebc6add67 since 093bfce2b (open-webui/open-webui#31993): a new chat opened with
`?call=true` shows the side panel without the call. Opening the call now sets the panel's
switches at once, where it waited a moment before, and the call does not stay open. Both tests
that expect a call are red; they pass on that build with the wait put back.

Discriminates: passes on the dev b859124f9 build, fails on that build with 79baaca3b and
51da3eefc reverted (each refused case opens a call without its message, and the Kokoro call
never asks for the model).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, Route, expect

from harness.actors import Actor
from harness.audio_engine import AUDIO_CONFIG
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

KOKORO_MODEL = re.compile(r"Kokoro-82M")
MODEL_LOAD_ERROR = re.compile(r"Could not locate file: .*Kokoro-82M")
NO_CALL_FROM_URL = "?call=true opened the side panel without the call (open-webui/open-webui#31993)"
# a call opened in error shows at once, so this is long enough to see none open
NO_CALL_WAIT_MS = 1500


def end_call_button(page: Page):
    return page.get_by_role("button", name="End call")


def open_from_url(page: Page, query: str) -> None:
    page.goto(f"/?{query}")
    expect(chat_input(page)).to_be_visible()


def expect_no_call(page: Page) -> None:
    page.wait_for_timeout(NO_CALL_WAIT_MS)
    refused_call = end_call_button(page)
    expect(refused_call, "#31825: the URL opened a call the button refuses").to_have_count(0)


@pytest.fixture
def second_model(upstream, admin):
    model_id = f"second-{uuid.uuid4().hex[:8]}"
    upstream.models.append(model_id)
    with admin.client() as client:
        listed = client.get("/api/models", params={"refresh": "true"})
        assert model_id in [entry["id"] for entry in listed.json()["data"]], listed.text
        grant = {"principal_type": "user", "principal_id": "*", "permission": "read"}
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={"id": model_id, "name": model_id, "access_grants": [grant]},
        )
        assert shared.status_code == 200, shared.text
    yield model_id
    upstream.models.remove(model_id)
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})
        client.get("/api/models", params={"refresh": "true"})


@pytest.fixture
def no_call_permission(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["call"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def web_speech_to_text(admin, speech_engine):
    # the speech engine fixture restores the audio settings afterwards
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        current["stt"]["ENGINE"] = "web"
        saved = client.post(AUDIO_CONFIG[1], json=current)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def kokoro_caller(make_user) -> Actor:
    account = make_user()
    kokoro = {"audio": {"tts": {"engine": "browser-kokoro"}}}
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": kokoro})
    assert saved.status_code == 200, saved.text
    return account


def test_voice_mode_from_the_url_opens_the_call(voice_page_for, make_user, speech_engine):
    page = voice_page_for(make_user())

    open_from_url(page, "call=true")

    expect(end_call_button(page), NO_CALL_FROM_URL).to_be_visible()


def test_without_allow_call_the_url_opens_no_call(
    voice_page_for, make_user, speech_engine, no_call_permission
):
    page = voice_page_for(make_user())

    open_from_url(page, "call=true")

    expect(page.get_by_role("button", name="Voice mode")).to_have_count(0)
    expect_no_call(page)


def test_two_models_from_the_url_open_no_call(
    voice_page_for, make_user, speech_engine, second_model
):
    page = voice_page_for(make_user())

    open_from_url(page, f"models={MOCK_MODEL_ID},{second_model}&call=true")

    expect(page.get_by_text("Select only one model to call"), "#31825").to_be_visible()
    expect_no_call(page)


def test_the_web_speech_engine_opens_no_call_from_the_url(
    voice_page_for, make_user, web_speech_to_text
):
    page = voice_page_for(make_user())

    open_from_url(page, "call=true")

    refusal = page.get_by_text("Call feature is not supported when using Web STT engine")
    expect(refusal, "#31825").to_be_visible()
    expect_no_call(page)


def test_a_kokoro_call_from_the_url_sets_up_the_voice(voice_page_for, kokoro_caller, speech_engine):
    page = voice_page_for(kokoro_caller)
    asked_for_model: list[str] = []

    def refuse_download(route: Route) -> None:
        asked_for_model.append(route.request.url)
        route.fulfill(status=404, body="not here")

    page.context.route(KOKORO_MODEL, refuse_download)
    open_from_url(page, "call=true")

    expect(end_call_button(page), NO_CALL_FROM_URL).to_be_visible()
    expect(page.get_by_text(MODEL_LOAD_ERROR), "#31828: no Kokoro setup was tried").to_be_visible()
    assert asked_for_model, "#31828: the call never set up the Kokoro voice"
