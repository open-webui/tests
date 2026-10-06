"""Journey: the generation tasks an admin sets in Admin Settings > Interface shape the next chat.

A model picked as the External Task Model is the one that writes the title, its own system prompt
included, and the Task Model Parameters reach that title request while the chat itself keeps its
own. Title Generation switched on there, with a prompt of the admin's own, sends that prompt
with the first question to the model and titles the chat with its answer; switched off, the chat
keeps its first message as the title and the model is never asked. Follow Up Generation switched on
asks the model with the admin's follow-up prompt, puts its suggested questions under the reply and
pressing one asks it. Tags Generation switched on asks with the admin's tags prompt and files the
chat under the model's tags, which its menu lists. Autocomplete Generation switched on lets a
person who turns Prompt Autocompletion on in their own settings see the model's continuation of
what they type, and Tab takes it into the message. Retrieval Query Generation switched on asks the
model, with the admin's Query Generation Prompt, what to search a file attached to the chat for:
the status under the reply shows its queries and they are what the file is searched with; switched
off, the file is searched with the question as typed and no queries show. Web Search Query
Generation and the image and tool calling prompts only act under legacy function calling, which
this suite leaves out.

Discriminates: passes on dev ebc6add67; in a frontend build whose Interface form sends the stored
task settings back in place of the edited ones, every test fails. In a backend copy, with the task
parameters left out of task requests the parameters test fails, with the generated queries thrown
away the switched on retrieval test fails (no queries show) and with the retrieval switch ignored
the switched off one fails (the model's queries show).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.cached_chat import attach
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
TITLE_PROMPT = "Write a newspaper headline for this exchange: {{MESSAGES:END:2}}"
HEADLINE_MARKER = "Write a newspaper headline"
FOLLOW_UP_MARKER = "Offer the traveller next questions"
FOLLOW_UP_PROMPT = f"{FOLLOW_UP_MARKER} as JSON follow_ups: {{{{MESSAGES:END:2}}}}"
AUTOCOMPLETE_MARKER = "You are an autocompletion system"
TAGS_MARKER = "File this trip under travel tags"
TAGS_PROMPT = f"{TAGS_MARKER} as JSON tags: {{{{MESSAGES:END:2}}}}"
QUESTION = "When does the ferry to Hallstatt leave?"
ANSWER = "The first ferry leaves at seven."
SHELF_QUESTION = "Which shelf holds the atlas?"
SHELF_NOTES = "The atlas is kept on the top shelf of the map room."
QUERY_MARKER = "List the library searches for this request"


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
    prompt_field(settings, "Follow Up Generation Prompt").fill(FOLLOW_UP_PROMPT)
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
    prompt_field(settings, "Tags Generation Prompt").fill(TAGS_PROMPT)
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


@pytest.fixture
def scribe(admin):
    """A public preset with a system prompt of its own, to pick as the task model."""
    form = {
        "id": f"scribe-{uuid.uuid4().hex[:8]}",
        "name": f"Scribe {uuid.uuid4().hex[:6]}",
        "base_model_id": "mock-model",
        "meta": {},
        "params": {"system": "You are the harbour scribe."},
        "access_grants": [{"principal_type": "user", "principal_id": "*", "permission": "read"}],
    }
    with admin.client() as client:
        client.post("/api/v1/models/create", json=form).raise_for_status()
    yield form
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": form["id"]})


def test_the_external_task_model_writes_the_title(
    page_for, admin, make_user, upstream, tasks_restored, scribe
):
    with admin.client() as client:
        set_title_generation(client, turn_on=True)
    settings = open_interface_settings(page_for(admin))
    task_model = settings.get_by_text("External Task Model", exact=True)
    task_model.locator("xpath=following-sibling::div//select").select_option(label=scribe["name"])
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"title": "Scribe Title"}', match=reply.answering("Generate a concise")),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
    )

    send(page, QUESTION)
    expect_reply(page, ANSWER)
    expect(sidebar(page).get_by_text("Scribe Title")).to_be_visible()

    [title_request] = requests_with(upstream, "Generate a concise")
    assert "You are the harbour scribe." in str(title_request["messages"])
    [chat_request] = [body for body in upstream.chat_requests() if body.get("stream")]
    assert "harbour scribe" not in str(chat_request["messages"])


def test_task_model_parameters_reach_the_title_request_and_not_the_chat(
    page_for, admin, make_user, upstream, tasks_restored
):
    with admin.client() as client:
        set_title_generation(client, turn_on=True)
    settings = open_interface_settings(page_for(admin))
    settings.get_by_role("button", name="Task Model Parameters").click()
    temperature = settings.get_by_text("Temperature", exact=True)
    temperature.locator("xpath=ancestor::div[.//button][1]").get_by_role(
        "button", name="Default"
    ).click()
    settings.get_by_role("spinbutton", name="Temperature").fill("0.15")
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"title": "Cool Ferry Title"}', match=reply.answering("Generate a concise")),
        reply.text(ANSWER, match=reply.answering(QUESTION)),
    )

    send(page, QUESTION)
    expect_reply(page, ANSWER)
    expect(sidebar(page).get_by_text("Cool Ferry Title")).to_be_visible()

    [title_request] = requests_with(upstream, "Generate a concise")
    assert title_request.get("temperature") == 0.15
    [chat_request] = [body for body in upstream.chat_requests() if body.get("stream")]
    assert chat_request.get("temperature") != 0.15


def embedded_texts(upstream) -> str:
    return str([entry.body for entry in upstream.requests_to("/embeddings")])


def ask_about_the_upload(page: Page) -> None:
    attach(page, "library.txt", SHELF_NOTES)
    expect(page.get_by_role("button", name="library.txt")).to_be_visible()
    send(page, SHELF_QUESTION)
    expect_reply(page, "The top shelf.")
    page.get_by_role("button", name="Toggle status history").first.click()


def test_a_switched_on_retrieval_query_generation_searches_the_upload_with_the_models_queries(
    page_for, admin, make_user, upstream, tasks_restored
):
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Retrieval Query Generation", turn_on=True)
    prompt_field(settings, "Query Generation Prompt").fill(
        f"{QUERY_MARKER}: {{{{MESSAGES:END:2}}}}"
    )
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text('{"queries": ["map room atlas shelf"]}', match=reply.answering(QUERY_MARKER)),
        reply.text("The top shelf.", match=reply.answering(SHELF_QUESTION)),
    )

    ask_about_the_upload(page)

    expect(conversation(page).get_by_text("map room atlas shelf")).to_be_visible()
    [query_request] = requests_with(upstream, QUERY_MARKER)
    assert SHELF_QUESTION in str(query_request["messages"])
    assert "map room atlas shelf" in embedded_texts(upstream)


def test_a_switched_off_retrieval_query_generation_searches_the_upload_with_the_question(
    page_for, admin, make_user, upstream, tasks_restored
):
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        client.post(
            "/api/v1/tasks/config/update",
            json={**current, "ENABLE_RETRIEVAL_QUERY_GENERATION": True},
        ).raise_for_status()
    settings = open_interface_settings(page_for(admin))
    set_switch(settings, "Retrieval Query Generation", turn_on=False)
    save(settings)
    page = page_for(make_user())
    upstream.queue(
        reply.text(
            '{"queries": ["map room atlas shelf"]}', match=reply.answering("search queries")
        ),
        reply.text("The top shelf.", match=reply.answering(SHELF_QUESTION)),
    )

    ask_about_the_upload(page)

    expect(conversation(page).get_by_text("Querying")).to_be_visible()
    expect(conversation(page).get_by_text("map room atlas shelf")).to_have_count(0)
    assert [body for body in upstream.chat_requests() if not body.get("stream")] == []
    assert SHELF_QUESTION in embedded_texts(upstream)
