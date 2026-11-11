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

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`). A system prompt
typed into a side chat's Controls was dropped (open-webui/open-webui#34027, fix PR
open-webui/open-webui#34064), which the test for it pins in every way a side chat starts and every
state it can be in; there a side chat opened with `/side` adds its side conversation instruction,
so it checks that the typed prompt is sent in place of the one the chat had. For it the dev
c15f28649 build is the mutant. On that build a New Chat tab beside a chat also got the side
conversation instruction (open-webui/open-webui#34029), which turns the other system prompt tests
red there, and its Controls had no working streaming choice (no filed issue). Every variant passes
on dev e791cbdde.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_controls import control_button, open_chat_controls, open_section
from utils.chat_surface import EVERYWHERE, ChatOpener, ChatSurface
from utils.chat_ui import chat_input, close_menu, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PERSONAL_PROMPT = "You answer like a pirate."
CHAT_PROMPT = "You answer like a butler."


def open_with(chat_for: ChatOpener, make_user, **ui) -> ChatSurface:
    account = make_user()
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui})
    saved.raise_for_status()
    chat = chat_for(account)
    expect(chat_input(chat)).to_be_visible()
    return chat


def controls(chat: ChatSurface) -> Locator:
    prompt = open_chat_controls(chat).get_by_role("textbox", name="System Prompt", exact=True)
    expect(prompt).to_be_visible()
    return prompt


def stream_toggle(chat: ChatSurface) -> Locator:
    panel = open_section(chat.page.get_by_role("menu"), "Advanced Params")
    return control_button(panel, "Stream Chat Response")


def ask(chat: ChatSurface, upstream, question: str) -> dict:
    """Closes the open Controls, then sends `question` and returns what the provider got."""
    upstream.queue(reply.text("Aye.", match=reply.answering(question)))
    close_menu(chat)
    send(chat, question)
    expect_reply(chat, "Aye.")
    return [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]


def system_prompts(body: dict) -> list[str]:
    return [entry["content"] for entry in body["messages"] if entry["role"] == "system"]


# --------------------------------------------------------------------------- system prompt


def test_a_cleared_chat_system_prompt_falls_back_to_the_personal_one(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34029, the side conversation instruction."""
    chat = open_with(chat_for, make_user, system=PERSONAL_PROMPT)
    prompt = controls(chat)
    prompt.fill(CHAT_PROMPT)
    prompt.fill("")

    sent = ask(chat, upstream, "Where is the treasure?")

    assert system_prompts(sent) == [PERSONAL_PROMPT]


def test_a_chat_system_prompt_replaces_the_personal_one(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34029, the side conversation instruction."""
    chat = open_with(chat_for, make_user, system=PERSONAL_PROMPT)
    controls(chat).fill(CHAT_PROMPT)

    sent = ask(chat, upstream, "Is dinner ready?")

    assert system_prompts(sent) == [CHAT_PROMPT]


def test_without_any_system_prompt_none_is_sent(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34029, the side conversation instruction."""
    chat = open_with(chat_for, make_user)
    prompt = controls(chat)
    prompt.fill(CHAT_PROMPT)
    prompt.fill("")

    sent = ask(chat, upstream, "Anything at all?")

    assert [text for text in system_prompts(sent) if text.strip()] == []


# one model only: the chat's Controls hold one system prompt for every model it asks
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_a_system_prompt_typed_into_the_chats_controls_is_sent(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34027."""
    chat = open_with(chat_for, make_user, system=PERSONAL_PROMPT)
    controls(chat).fill(CHAT_PROMPT)

    sent = system_prompts(ask(chat, upstream, "Shall I pour the tea?"))

    assert any(CHAT_PROMPT in text for text in sent), (
        f"the {chat.place} was not sent the system prompt typed into its Controls: {sent}"
    )
    assert not any(PERSONAL_PROMPT in text for text in sent), (
        f"the {chat.place} kept the system prompt it had over the one typed in: {sent}"
    )


# ------------------------------------------------------------------------- streaming


def test_a_chat_can_switch_streaming_on_over_the_account_setting(chat_for, make_user, upstream):
    """Red in a side chat on dev c15f28649: no Stream Chat Response control there."""
    chat = open_with(chat_for, make_user, params={"stream_response": False})
    controls(chat)
    toggle = stream_toggle(chat)
    toggle.click()
    expect(toggle).to_have_text("On")

    sent = ask(chat, upstream, "Stream this one")

    assert sent["stream"] is True


def test_a_chat_can_switch_streaming_off_over_the_account_setting(chat_for, make_user, upstream):
    """Red in a side chat on dev c15f28649: no Stream Chat Response control there."""
    chat = open_with(chat_for, make_user, params={"stream_response": True})
    controls(chat)
    toggle = stream_toggle(chat)
    toggle.click()
    toggle.click()
    expect(toggle).to_have_text("Off")

    sent = ask(chat, upstream, "Do not stream this one")

    assert sent.get("stream") is not True


def test_the_account_setting_applies_when_the_chat_has_none(chat_for, make_user, upstream):
    """Red in a side chat on dev c15f28649: no Stream Chat Response control there."""
    chat = open_with(chat_for, make_user, params={"stream_response": False})
    controls(chat)
    expect(stream_toggle(chat)).to_have_text("Default")

    sent = ask(chat, upstream, "Whatever the account says")

    assert sent.get("stream") is not True
