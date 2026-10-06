"""Journey: the generation tasks an admin sets in Admin Settings > Interface shape the next chat.

Title Generation switched on there, with a prompt of the admin's own, sends that prompt with the
first question to the model and titles the chat with its answer; switched off, the chat keeps
its first message as the title and the model is never asked. Follow Up Generation switched on
puts the model's suggested questions under the reply, and pressing one asks it. Tags Generation
switched on files the chat under the model's tags, which its menu lists. Autocomplete
Generation switched on lets a person who turns Prompt Autocompletion on in their own settings see
the model's continuation of what they type, and Tab takes it into the message.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Interface form sends the stored
task settings back in place of the edited ones, every test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
TITLE_PROMPT = "Write a newspaper headline for this exchange: {{MESSAGES:END:2}}"
HEADLINE_MARKER = "Write a newspaper headline"
FOLLOW_UP_MARKER = "Suggest 3-5 relevant follow-up questions"
AUTOCOMPLETE_MARKER = "You are an autocompletion system"
TAGS_MARKER = "Generate 1-3 broad tags"
QUESTION = "When does the ferry to Hallstatt leave?"
ANSWER = "The first ferry leaves at seven."


@pytest.fixture
def tasks_restored(preserve):
    preserve("tasks", CHAT_CONFIG)


def open_interface_settings(page: Page) -> Locator:
    page.goto("/admin/settings/interface")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Title Generation")).to_be_visible()
    return settings


def set_switch(settings: Locator, name: str, turn_on: bool) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def prompt_field(settings: Locator, label: str) -> Locator:
    return settings.get_by_text(label, exact=True).locator("xpath=following-sibling::div//textarea")


def save(settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(settings.page.get_by_text("Settings saved successfully!")).to_be_visible()


def set_title_generation(admin_client, turn_on: bool) -> None:
    current = admin_client.get("/api/v1/tasks/config").json()
    saved = admin_client.post(
        "/api/v1/tasks/config/update", json={**current, "ENABLE_TITLE_GENERATION": turn_on}
    )
    saved.raise_for_status()


def sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def requests_with(upstream, marker: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if marker in str(body.get("messages"))]


def test_a_switched_on_title_generation_titles_the_chat_with_the_admins_prompt(
    page_for, admin, make_user, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Title Generation", turn_on=True)
    prompt_field(settings, "Title Generation Prompt").fill(TITLE_PROMPT)
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"title": "Early Ferry Confirmed"}', match=reply.answering(HEADLINE_MARKER)),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
    )

    send(page, QUESTION)

    expect_reply(page, ANSWER)
    expect(sidebar(page).get_by_text("Early Ferry Confirmed")).to_be_visible()
    [title_request] = requests_with(upstream, HEADLINE_MARKER)
    assert QUESTION in str(title_request["messages"])


def test_a_switched_off_title_generation_keeps_the_question_as_the_title(
    page_for, admin, make_user, upstream, tasks_restored
):
    with admin.client() as client:
        set_title_generation(client, turn_on=True)
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Title Generation", turn_on=False)
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"title": "Early Ferry Confirmed"}', match=reply.answering("title")),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
    )

    send(page, QUESTION)

    expect_reply(page, ANSWER)
    expect(sidebar(page).get_by_text(QUESTION)).to_be_visible()
    assert [body for body in upstream.chat_requests() if not body.get("stream")] == []


def test_a_switched_on_follow_up_generation_offers_questions_that_ask_when_pressed(
    page_for, admin, make_user, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Follow Up Generation", turn_on=True)
    save(settings)
    page = page_for(make_user())
    follow_up = "How long does the crossing take?"
    upstream.queue(
        reply.text(f'{{"follow_ups": ["{follow_up}"]}}', match=reply.answering(FOLLOW_UP_MARKER)),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
        reply.text("About twenty minutes.", match=reply.answering(follow_up)),
    )

    send(page, QUESTION)
    expect_reply(page, ANSWER)
    page.get_by_role("button", name=f"Follow up: {follow_up}").click()

    expect_reply(page, "About twenty minutes.")
    assert len(requests_with(upstream, FOLLOW_UP_MARKER)) >= 1


def test_a_switched_on_autocomplete_suggests_a_continuation_that_tab_accepts(
    page_for, admin, make_user, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Autocomplete Generation", turn_on=True)
    save(settings)
    page = page_for(make_user())
    page.goto("/?settings=interface")
    personal = page.locator("#tab-interface").get_by_role("switch", name="Prompt Autocompletion")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        personal.click()
    upstream.queue(
        reply.text('{"text": " early in the autumn."}', match=reply.answering(AUTOCOMPLETE_MARKER))
    )

    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("The best time to visit Hallstatt is")
    expect(page.locator("[data-suggestion]")).to_have_attribute(
        "data-suggestion", " early in the autumn."
    )
    page.keyboard.press("Tab")

    expect(chat_input(page)).to_have_text(
        "The best time to visit Hallstatt is early in the autumn."
    )
    assert "Hallstatt" in str(requests_with(upstream, AUTOCOMPLETE_MARKER)[-1]["messages"])


def test_a_switched_on_tags_generation_files_the_chat_under_the_models_tags(
    page_for, admin, make_user, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Tags Generation", turn_on=True)
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"tags": ["Ferries", "Lake travel"]}', match=reply.answering(TAGS_MARKER)),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
    )

    send(page, QUESTION)
    expect_reply(page, ANSWER)
    page.get_by_role("button", name="Chat actions").first.click()

    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Ferries", exact=True)).to_be_visible()
    expect(menu.get_by_text("Lake travel", exact=True)).to_be_visible()
