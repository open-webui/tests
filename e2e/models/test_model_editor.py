"""Journey: what a model's settings in the workspace editor change in the chat on that model.

A fresh admin edits a preset on the scripted model in the workspace's model editor, saves it and
opens a chat on it. Unticking File Upload leaves the chat's upload marked as unsupported, and
unticking Web Search takes the Web Search toggle out of the chat. Web Search ticked as a default
feature starts a new chat with it on. Advanced params reach the provider with the chat, a tag
becomes a filter of the model selector and a prompt suggestion shows on the new chat and sends
itself. A tool ticked for the model is offered to the provider, a filter changes the message
before it is sent and an action shows under the reply and runs. A builtin tool category unticked
for the model, or the Builtin Tools capability as a whole, leaves those tools out of the chat. A
model made in the editor is private to its maker until the editor shares it with a group, whose
members then see it. The system prompt is covered by e2e/workspace/test_workspace_presets.py.

Discriminates: passes on the dev 176d31d1d build; on a build of it whose editor saves without
the capabilities, default features, tags, prompt suggestions, tools, filters, actions and params,
and starts a new model shared with everyone, every test fails. With only the access grants left
out of the save, the group member never sees the model. In a backend copy that offers every
builtin tool whatever the model says, both builtin tool tests fail.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOCKER_TOOL = """class Tools:
    def lookup_locker(self, number: int) -> str:
        \"\"\"Look up who holds a locker.

        :param number: the locker number
        \"\"\"
        return f"Locker {number} belongs to Ada."
"""

FILTER_SOURCE = """class Filter:
    def inlet(self, body: dict, __user__=None) -> dict:
        body["messages"][-1]["content"] += " (checked at the harbour gate)"
        return body
"""

ACTION_SOURCE = """class Action:
    async def action(self, body: dict, __user__=None, __event_emitter__=None):
        await __event_emitter__(
            {"type": "notification", "data": {"type": "success", "content": "Logged at the desk"}}
        )
"""


def unique(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def preset(builder) -> dict:
    """A preset on the scripted model that the builder made over the API, every account reads."""
    model = {"id": f"editor-{uuid.uuid4().hex[:8]}", "name": unique("Harbour guide")}
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    return model


@pytest.fixture
def web_search_on(admin, preserve, listener) -> None:
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


def open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def save(editor: Locator) -> None:
    editor.get_by_role("button", name="Save & Update").click()
    expect(editor.page).to_have_url(re.compile(r"/workspace/models/?$"))


def section(editor: Locator, title: str) -> Locator:
    """The innermost part of the editor under `title` that holds checkboxes."""
    titled = editor.locator("div").filter(has=editor.page.get_by_text(title, exact=True))
    return titled.filter(has=editor.page.get_by_role("checkbox")).last


def set_checkbox(editor: Locator, title: str, label: str, checked: bool) -> None:
    checkbox = section(editor, title).get_by_role("checkbox", name=label, exact=True)
    if (checkbox.get_attribute("aria-checked") == "true") != checked:
        checkbox.click()
    expect(checkbox).to_have_attribute("aria-checked", str(checked).lower())


def header_row(editor: Locator, label: str) -> Locator:
    """The row holding the setting labelled `label` and its switch."""
    return editor.get_by_text(label, exact=True).locator("xpath=..")


def pick(editor: Locator, kind: str, name: str) -> None:
    """Tick `name` in the editor's picker of that kind: Tool, Filter or Action."""
    editor.get_by_text(f"Select {kind}", exact=True).click()
    editor.page.get_by_placeholder(f"Search {kind.lower()}s").fill(name)
    editor.page.get_by_role("button", name=name).click()
    editor.page.keyboard.press("Escape")
    expect(editor.get_by_role("checkbox", name=name)).to_be_checked()


def open_chat_on(page: Page, model: dict) -> None:
    page.goto(f"/?model={model['id']}")
    expect(page.get_by_role("button", name=f"Selected model: {model['name']}")).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def open_menu(page: Page, button_name: str) -> Locator:
    page.get_by_role("button", name=button_name, exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def web_search_toggle(page: Page) -> Locator:
    return page.get_by_role("menu").get_by_role("button", name="Web Search")


def model_option(page: Page, name: str) -> Locator:
    return page.get_by_role("listbox", name="Available models").get_by_role(
        "option", name=f"Select {name} model"
    )


def sent_request(page: Page, upstream, question: str) -> dict:
    """Send `question` in the open chat; the request the provider got for it."""
    upstream.queue(reply.text("noted", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "noted")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def test_unticking_file_upload_leaves_the_upload_unsupported(page_for, builder, preset):
    page = page_for(builder)
    open_chat_on(page, preset)
    with page.expect_file_chooser():
        open_menu(page, "More").get_by_role("button", name="Upload Files").click()

    editor = open_editor(page, preset)
    set_checkbox(editor, "Capabilities", "File Upload", False)
    save(editor)

    open_chat_on(page, preset)
    menu = open_menu(page, "More")
    unsupported = tooltip_button(menu, "Model(s) do not support file upload")
    expect(unsupported).to_have_text("Upload Files")


def test_unticking_web_search_takes_the_toggle_out_of_the_chat(
    page_for, builder, preset, web_search_on
):
    page = page_for(builder)
    open_chat_on(page, preset)
    open_menu(page, "Integrations")
    expect(web_search_toggle(page)).to_be_visible()

    editor = open_editor(page, preset)
    set_checkbox(editor, "Capabilities", "Web Search", False)
    save(editor)

    open_chat_on(page, preset)
    integrations = page.get_by_role("button", name="Integrations", exact=True)
    if integrations.count():
        open_menu(page, "Integrations")
    expect(web_search_toggle(page)).to_have_count(0)


def test_web_search_as_a_default_feature_is_on_in_a_new_chat(
    page_for, builder, preset, upstream, web_search_on
):
    page = page_for(builder)
    editor = open_editor(page, preset)
    set_checkbox(editor, "Default Features", "Web Search", True)
    save(editor)

    open_chat_on(page, preset)
    open_menu(page, "Integrations")
    expect(web_search_toggle(page)).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("Escape")

    def is_chat_request(request) -> bool:
        return request.method == "POST" and request.url.endswith("/api/chat/completions")

    upstream.queue(reply.text("searched", match=reply.answering("any news?")))
    with page.expect_request(is_chat_request) as sent:
        send(page, "any news?")
    assert (sent.value.post_data_json.get("features") or {}).get("web_search") is True


TIME_TOOLS = {"get_current_timestamp", "calculate_timestamp"}


def offered_tool_names(page: Page, upstream, question: str) -> set[str]:
    return {tool["function"]["name"] for tool in sent_request(page, upstream, question)["tools"]}


def test_an_unticked_builtin_tool_is_left_out_of_the_chat(page_for, builder, preset, upstream):
    page = page_for(builder)
    open_chat_on(page, preset)
    before = offered_tool_names(page, upstream, "what time is it?")
    assert TIME_TOOLS | {"ask_user"} <= before, before

    editor = open_editor(page, preset)
    set_checkbox(editor, "Builtin Tools", "Time & Calculation", False)
    save(editor)

    open_chat_on(page, preset)
    after = offered_tool_names(page, upstream, "what time is it now?")
    assert not TIME_TOOLS & after, after
    assert "ask_user" in after


def test_unticking_the_builtin_tools_capability_offers_none_of_them(
    page_for, builder, preset, upstream
):
    page = page_for(builder)
    editor = open_editor(page, preset)
    set_checkbox(editor, "Capabilities", "Builtin Tools", False)
    save(editor)

    open_chat_on(page, preset)
    request = sent_request(page, upstream, "what time is it?")
    offered = {tool["function"]["name"] for tool in request.get("tools") or []}
    assert not (TIME_TOOLS | {"ask_user"}) & offered, offered


def test_advanced_params_reach_the_provider(page_for, builder, preset, upstream):
    page = page_for(builder)
    editor = open_editor(page, preset)
    header_row(editor, "Advanced Params").get_by_role("button", name="Show").click()
    for label in ("Temperature", "Seed"):
        header_row(editor, label).get_by_role("button", name="Default").click()
    editor.get_by_role("spinbutton", name="Temperature").fill("0.3")
    editor.get_by_role("spinbutton", name="Seed").fill("4242")
    save(editor)

    open_chat_on(page, preset)
    request = sent_request(page, upstream, "how warm is the water?")

    assert request["temperature"] == 0.3
    assert request["seed"] == 4242


def test_a_tag_filters_the_model_selector(page_for, builder, preset):
    tag = f"harbour{uuid.uuid4().hex[:6]}"
    page = page_for(builder)
    editor = open_editor(page, preset)
    editor.get_by_placeholder("Add a tag...").fill(tag)
    editor.get_by_placeholder("Add a tag...").press("Enter")
    expect(editor.get_by_text(tag, exact=True)).to_be_visible()
    save(editor)

    open_chat_on(page, preset)
    page.get_by_role("button", name=f"Selected model: {preset['name']}").click()
    expect(model_option(page, MOCK_MODEL_ID)).to_be_visible()
    page.get_by_role("button", name="All", exact=True).click()
    page.get_by_role("button", name=tag, exact=True).click()

    expect(model_option(page, preset["name"])).to_be_visible()
    expect(model_option(page, MOCK_MODEL_ID)).to_have_count(0)


def test_a_prompt_suggestion_shows_on_the_new_chat_and_sends(page_for, builder, preset, upstream):
    title, content = unique("Tide times"), unique("When is high tide in the harbour today?")
    page = page_for(builder)
    editor = open_editor(page, preset)
    prompts = editor.locator("section").filter(has=page.get_by_text("Prompts", exact=True))
    prompts.get_by_role("button", name="Default").click()
    prompts.get_by_role("textbox", name="Title", exact=True).fill(title)
    prompts.get_by_role("textbox", name="Content").fill(content)
    save(editor)

    open_chat_on(page, preset)
    upstream.queue(reply.text("At noon.", match=reply.answering(content)))
    page.get_by_role("listitem").filter(has_text=title).click()

    expect_reply(page, "At noon.")
    sent = next(filter(reply.answering(content), upstream.chat_requests()))
    assert sent["messages"][-1] == {"role": "user", "content": content}


def test_a_tool_ticked_in_the_editor_is_offered_with_the_chat(
    page_for, admin, builder, preset, upstream
):
    tool_name = unique("Locker desk")
    with python_tool(admin, LOCKER_TOOL, name=tool_name):
        page = page_for(builder)
        editor = open_editor(page, preset)
        pick(editor, "Tool", tool_name)
        save(editor)

        open_chat_on(page, preset)
        request = sent_request(page, upstream, "who holds locker 7?")

    offered = {tool["function"]["name"] for tool in request.get("tools") or []}
    assert "lookup_locker" in offered, f"the model's tool was not offered: {sorted(offered)}"


def test_a_filter_ticked_in_the_editor_runs_on_the_chat(page_for, admin, builder, preset, upstream):
    with installed_function(admin, FILTER_SOURCE) as filter_id:
        page = page_for(builder)
        editor = open_editor(page, preset)
        pick(editor, "Filter", filter_id)
        save(editor)

        open_chat_on(page, preset)
        request = sent_request(page, upstream, "may I moor here?")

    assert request["messages"][-1]["content"] == "may I moor here? (checked at the harbour gate)"


def test_an_action_ticked_in_the_editor_shows_under_the_reply_and_runs(
    page_for, admin, builder, preset, upstream
):
    with installed_function(admin, ACTION_SOURCE) as action_id:
        page = page_for(builder)
        editor = open_editor(page, preset)
        pick(editor, "Action", action_id)
        save(editor)

        open_chat_on(page, preset)
        sent_request(page, upstream, "log my arrival")
        page.get_by_role("button", name=action_id, exact=True).click()

        expect(page.get_by_text("Logged at the desk")).to_be_visible()


@pytest.fixture
def crew(admin, make_user):
    """A member of a fresh group and a user outside it; the group is deleted afterwards."""
    member, stranger = make_user(), make_user()
    name = unique("Harbour crew")
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [member.id]}
        )
        assert added.status_code == 200, added.text
    yield name, member, stranger
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def sees_model(account, model_id: str) -> bool:
    with account.client() as client:
        listed = client.get("/api/models")
    listed.raise_for_status()
    return model_id in {model["id"] for model in listed.json()["data"]}


def test_a_new_model_is_private_until_shared_with_a_group(page_for, builder, crew):
    group_name, member, stranger = crew
    model = {"id": f"editor-{uuid.uuid4().hex[:8]}", "name": unique("Crew guide")}
    page = page_for(builder)
    page.goto("/workspace/models/create")
    editor = page.get_by_role("main")
    editor.get_by_placeholder("Model Name").fill(model["name"])
    editor.get_by_placeholder("Model ID").fill(model["id"])
    editor.get_by_role("button", name="Select a base model (e.g. llama3, gpt-4o)").click()
    page.get_by_role("textbox", name="Search In Models").fill(MOCK_MODEL_ID)
    model_option(page, MOCK_MODEL_ID).click()
    editor.get_by_role("button", name="Save & Create").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))
    assert not sees_model(member, model["id"]), "a new model was visible to another user"

    editor = open_editor(page, model)
    editor.get_by_role("button", name="Access", exact=True).click()
    access = page.get_by_role("dialog").filter(has_text="Access Control")
    access.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(group_name)
    picker.get_by_role("button", name=re.compile(re.escape(group_name))).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(access.get_by_text(group_name)).to_be_visible()
    page.keyboard.press("Escape")
    save(editor)

    member_page = page_for(member)
    member_page.get_by_role("button", name=re.compile("^Selected model")).click()
    member_page.get_by_role("textbox", name="Search In Models").fill(model["name"])
    expect(model_option(member_page, model["name"])).to_be_visible()
    assert not sees_model(stranger, model["id"]), "sharing with a group showed it to everyone"
