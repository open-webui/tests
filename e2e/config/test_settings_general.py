"""Journey: Settings > General, the theme, the language, the system prompt and the parameters.

A user picks the Dark theme and the page turns dark, picks Deutsch and the app speaks German, and
both stay after a reload. A personal system prompt and a custom temperature saved on the tab are
sent with every chat and are still in the tab after a reload. A second account in a browser of its
own meets none of it: the default theme and language, and a chat sent without the prompt or the
temperature. Stream Chat Response switched off asks the model for the whole reply at once, which
still shows, and max_tokens, a stop sequence and the reasoning effort saved there reach the model.

Discriminates: passes on dev 176d31d1d; in a frontend copy, dropping the class the theme picker
adds to the page turns the theme test red, not passing the chosen language to i18next turns the
language test red, and saving General without its `system` field turns the system prompt test red.
In a frontend build of dev 30f3f6a8f whose chat request leaves out the account's saved parameters
(stream, params and stop), the temperature, stream and parameter tests fail.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DARK = re.compile(r"(^|\s)dark(\s|$)")
SYSTEM_PROMPT = "Answer every question as a lighthouse keeper would."


def general_tab(page: Page) -> Locator:
    page.goto("/?settings=general")
    tab = page.locator("#tab-general")
    expect(tab.get_by_role("combobox").first).to_be_visible()
    return tab


def save(page: Page, tab: Locator) -> None:
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        tab.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Settings saved successfully!")).to_be_visible()


def ask(page: Page, upstream, question: str) -> dict:
    upstream.queue(reply.text("Aye, keeper here.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "Aye, keeper here.")
    return [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]


def system_prompts(body: dict) -> list[str]:
    return [entry["content"] for entry in body["messages"] if entry["role"] == "system"]


def test_the_dark_theme_is_applied_and_kept_after_a_reload(page_for, make_user):
    page = page_for(make_user())
    tab = general_tab(page)
    expect(page.locator("html")).not_to_have_class(DARK)

    tab.get_by_role("combobox", name="Theme").select_option("dark")
    expect(page.locator("html")).to_have_class(DARK)

    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(page.locator("html")).to_have_class(DARK)
    expect(general_tab(page).get_by_role("combobox", name="Theme")).to_have_value("dark")

    other = page_for(make_user())
    expect(chat_input(other)).to_be_visible()
    expect(other.locator("html")).not_to_have_class(DARK)


def test_a_chosen_language_relabels_the_app_and_is_kept_after_a_reload(page_for, make_user):
    page = page_for(make_user())
    tab = general_tab(page)

    tab.get_by_role("combobox", name="Language").select_option("de-DE")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", name="Allgemein", exact=True)).to_be_visible()

    page.reload()
    expect(page.get_by_role("navigation").get_by_label("Benutzermenü")).to_be_visible()
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")

    other = page_for(make_user())
    expect(
        other.get_by_role("navigation", name="Chat history").get_by_label("User menu")
    ).to_be_visible()


def test_the_system_prompt_and_temperature_reach_the_model(page_for, make_user, upstream):
    page = page_for(make_user())
    tab = general_tab(page)
    tab.get_by_role("textbox").fill(SYSTEM_PROMPT)
    tab.get_by_role("button", name="Show").click()
    tab.get_by_text("Temperature", exact=True).locator("xpath=following-sibling::button").click()
    tab.get_by_role("spinbutton", name="Temperature").fill("0.35")
    save(page, tab)

    sent = ask(page, upstream, "Is the lamp lit tonight?")
    assert system_prompts(sent) == [SYSTEM_PROMPT]
    assert sent["temperature"] == 0.35

    page.reload()
    tab = general_tab(page)
    expect(tab.get_by_role("textbox")).to_have_value(SYSTEM_PROMPT)
    tab.get_by_role("button", name="Show").click()
    expect(tab.get_by_role("spinbutton", name="Temperature")).to_have_value("0.35")

    other = page_for(make_user())
    sent_for_other = ask(other, upstream, "Is the harbour busy?")
    assert system_prompts(sent_for_other) == []
    assert "temperature" not in sent_for_other


def parameter_button(tab: Locator, label: str) -> Locator:
    return tab.get_by_text(label, exact=True).locator("xpath=following-sibling::button")


def test_stream_chat_response_off_asks_for_the_whole_reply_at_once(page_for, make_user, upstream):
    page = page_for(make_user())
    tab = general_tab(page)
    tab.get_by_role("button", name="Show").click()
    stream = parameter_button(tab, "Stream Chat Response")
    stream.click()
    stream.click()
    expect(stream).to_have_text("Off")
    save(page, tab)

    sent = ask(page, upstream, "Is the fog lifting?")

    assert sent["stream"] is False


def test_max_tokens_stop_and_reasoning_effort_reach_the_model(page_for, make_user, upstream):
    page = page_for(make_user())
    tab = general_tab(page)
    tab.get_by_role("button", name="Show").click()
    for label in ("max_tokens", "Stop Sequence", "Reasoning Effort"):
        parameter_button(tab, label).click()
    tab.get_by_role("spinbutton", name="max_tokens").fill("64")
    tab.get_by_role("textbox", name="Stop Sequence").fill("OVER")
    tab.get_by_role("textbox", name="Reasoning Effort").fill("low")
    save(page, tab)

    sent = ask(page, upstream, "How far is the next buoy?")

    assert sent["max_tokens"] == 64
    assert sent["stop"] == ["OVER"]
    assert sent["reasoning_effort"] == "low"
