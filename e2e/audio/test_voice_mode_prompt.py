"""Journey: the Voice Mode Prompt of Admin Settings > Interface instructs the model during a call.

With Voice Mode Prompt on and a Prompt Template of the admin's own, the request a person's voice
call sends the model carries that template in place of the built-in voice instructions, while a
typed chat of the same person carries neither. Switched off and saved, a call sends the model no
voice instructions at all.

Discriminates: passes on dev ebc6add67; in a frontend build whose Interface form sends the stored
task settings back in place of the edited ones, both tests fail (the call still carries the
built-in instructions).
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.cached_chat import ask
from utils.voice_call import start_call, wait_for_transcriptions

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

BUILT_IN_MARKER = "Everything you say will be spoken aloud."
ADMIN_TEMPLATE = "You are the lighthouse keeper on the radio. Answer in one breath."


@pytest.fixture
def tasks_restored(preserve):
    preserve("tasks")


def open_interface_settings(page: Page) -> Locator:
    page.goto("/admin/settings/interface")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Voice Mode Prompt")).to_be_visible()
    return settings


def set_voice_mode_prompt(settings: Locator, turn_on: bool) -> None:
    switch = settings.get_by_role("switch", name="Voice Mode Prompt")
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def save(settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(settings.page.get_by_text("Settings saved successfully!")).to_be_visible()


def first_spoken_request(upstream, typed_requests: int, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        requests = [body for body in upstream.chat_requests() if body.get("stream")]
        if len(requests) > typed_requests:
            return requests[typed_requests]
        time.sleep(0.2)
    raise AssertionError("the call never reached the model")


def test_the_admins_voice_template_instructs_the_model_during_a_call_only(
    page_for, voice_page_for, admin, make_user, speech_engine, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_voice_mode_prompt(settings, turn_on=True)
    template = settings.get_by_text("Prompt Template", exact=True)
    template.locator("xpath=following-sibling::div//textarea").fill(ADMIN_TEMPLATE)
    save(settings)
    page = voice_page_for(make_user())
    ask(page, upstream, "is the lamp lit?", reply.text("It is lit."))

    start_call(page)
    wait_for_transcriptions(speech_engine, 1, timeout=60.0)

    typed, spoken = upstream.chat_requests()[0], first_spoken_request(upstream, 1)
    assert ADMIN_TEMPLATE in str(spoken["messages"])
    assert BUILT_IN_MARKER not in str(spoken["messages"])
    assert ADMIN_TEMPLATE not in str(typed["messages"])


def test_a_switched_off_voice_mode_prompt_sends_no_voice_instructions(
    page_for, voice_page_for, admin, make_user, speech_engine, upstream, tasks_restored
):
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        client.post(
            "/api/v1/tasks/config/update",
            json={**current, "ENABLE_VOICE_MODE_PROMPT": True, "VOICE_MODE_PROMPT_TEMPLATE": ""},
        ).raise_for_status()
    settings = open_interface_settings(page_for(admin))
    set_voice_mode_prompt(settings, turn_on=False)
    save(settings)
    page = voice_page_for(make_user())
    page.goto("/")

    start_call(page)
    wait_for_transcriptions(speech_engine, 1, timeout=60.0)

    spoken = first_spoken_request(upstream, 0)
    assert BUILT_IN_MARKER not in str(spoken["messages"])
    assert spoken["messages"][-1]["role"] == "user"
