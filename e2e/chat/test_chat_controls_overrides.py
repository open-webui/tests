"""A chat's own Controls lost to the account's settings: system prompt and streaming.

* open-webui/open-webui#30333, fix `f2702e1f0`: a chat's system prompt, typed into Controls and
  then cleared, stayed an empty string, which `??` does not skip, so the chat was sent an empty
  system prompt where it should fall back to the account's personal one.
* open-webui/open-webui#31361, fix `2c922bb53` (PR open-webui/open-webui#31362): the account-wide
  Stream Chat Response setting was read before the chat's own, so a chat's Controls setting was
  ignored whenever the account had one.

`test_a_chat_can_switch_streaming_off_over_the_account_setting` failed on dev 9bbb95048 in CI with
the reply left blank and passed 3 of 3 locally: since de73bb830 the reply in a new chat sometimes
stays blank until a reload although the server saved it whole (open-webui/open-webui#32091).

Discriminates: passes on the efe63bd34 build; with `f2702e1f0` reverted the cleared-prompt test
fails (the provider gets an empty system prompt), with `2c922bb53` reverted both streaming
overrides fail (the provider gets the account's choice).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PERSONAL_PROMPT = "You answer like a pirate."
CHAT_PROMPT = "You answer like a butler."


def open_with(page_for, make_user, **ui) -> Page:
    account = make_user()
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui})
    saved.raise_for_status()
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    return page


def controls(page: Page) -> Locator:
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    prompt = page.get_by_role("textbox", name="Enter system prompt")
    expect(prompt).to_be_visible()
    return prompt


def stream_toggle(page: Page) -> Locator:
    label = page.get_by_text("Stream Chat Response", exact=True)
    return label.locator("xpath=following-sibling::button")


def ask(page: Page, upstream, question: str) -> dict:
    upstream.queue(reply.text("Aye.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Aye.")
    return [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]


def system_prompts(body: dict) -> list[str]:
    return [entry["content"] for entry in body["messages"] if entry["role"] == "system"]


# --------------------------------------------------------------------------- system prompt


def test_a_cleared_chat_system_prompt_falls_back_to_the_personal_one(page_for, make_user, upstream):
    page = open_with(page_for, make_user, system=PERSONAL_PROMPT)
    prompt = controls(page)
    prompt.fill(CHAT_PROMPT)
    prompt.fill("")

    sent = ask(page, upstream, "Where is the treasure?")

    assert system_prompts(sent) == [PERSONAL_PROMPT]


def test_a_chat_system_prompt_replaces_the_personal_one(page_for, make_user, upstream):
    page = open_with(page_for, make_user, system=PERSONAL_PROMPT)
    controls(page).fill(CHAT_PROMPT)

    sent = ask(page, upstream, "Is dinner ready?")

    assert system_prompts(sent) == [CHAT_PROMPT]


def test_without_any_system_prompt_none_is_sent(page_for, make_user, upstream):
    page = open_with(page_for, make_user)
    prompt = controls(page)
    prompt.fill(CHAT_PROMPT)
    prompt.fill("")

    sent = ask(page, upstream, "Anything at all?")

    assert [text for text in system_prompts(sent) if text.strip()] == []


# --------------------------------------------------------------------------- streaming


def test_a_chat_can_switch_streaming_on_over_the_account_setting(page_for, make_user, upstream):
    page = open_with(page_for, make_user, params={"stream_response": False})
    controls(page)
    toggle = stream_toggle(page)
    toggle.click()
    expect(toggle).to_have_text("On")

    sent = ask(page, upstream, "Stream this one")

    assert sent["stream"] is True


def test_a_chat_can_switch_streaming_off_over_the_account_setting(page_for, make_user, upstream):
    page = open_with(page_for, make_user, params={"stream_response": True})
    controls(page)
    toggle = stream_toggle(page)
    toggle.click()
    toggle.click()
    expect(toggle).to_have_text("Off")

    sent = ask(page, upstream, "Do not stream this one")

    assert sent.get("stream") is not True


def test_the_account_setting_applies_when_the_chat_has_none(page_for, make_user, upstream):
    page = open_with(page_for, make_user, params={"stream_response": False})
    controls(page)
    expect(stream_toggle(page)).to_have_text("Default")

    sent = ask(page, upstream, "Whatever the account says")

    assert sent.get("stream") is not True
