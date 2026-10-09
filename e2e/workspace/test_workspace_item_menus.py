"""Journey: the "More" menu on a workspace list row clones, hides and deletes what it lists.

A fresh admin owns a model, a prompt, a tool and a skill. Clone from a row's menu opens the
creation editor filled in from the original, and saving it stores a copy that the chat offers
under its new name or command; a skill's Clone stores the copy at once and opens it. A model
hidden from its menu leaves the chat's model selector and returns when shown again. Delete,
after its confirmation, removes the model, prompt or tool from the list and from the chat's
selector, slash menu or integrations menu. In the prompt editor's version picker an older version
can be deleted, and the production one offers no delete.

Discriminates: passes on dev 30f3f6a8f. In a backend copy where the model, prompt, tool or prompt
history delete route answers true without deleting, the matching delete test goes red, and where
the model update ignores `hidden` the hide test goes red; in a frontend build whose clone handlers
change the copied system prompt, content or instructions, each clone test goes red. Retargeted for
9bbb95048, where a skill's Clone saves the copy at once and opens it: the skill clone test passes
on dev 178de3666 and goes red in a backend copy whose clone drops the instructions. Retargeted
for 37138282f, where prompt versions are picked from a menu: the prompt history test passes there.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.model_selector import model_options

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'


@pytest.fixture
def builder(make_user):
    """A fresh admin; what it owns in the workspace is deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _create(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _fetch(account: Actor, path: str) -> dict | list:
    with account.client() as client:
        fetched = client.get(path)
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def _new_model(account: Actor, system: str = "") -> dict:
    suffix = uuid.uuid4().hex[:8]
    model = {"id": f"menu-{suffix}", "name": f"Menu model {suffix}"}
    form = {**model, "base_model_id": MOCK_MODEL_ID, "meta": {}, "params": {"system": system}}
    _create(account, "/api/v1/models/create", form)
    return model


def _new_tool(account: Actor) -> dict:
    suffix = uuid.uuid4().hex[:8]
    tool = {"id": f"menu_tool_{suffix}", "name": f"Menu tool {suffix}"}
    _create(
        account,
        "/api/v1/tools/create",
        {**tool, "content": TOOL_SOURCE, "meta": {"description": "pings"}},
    )
    return tool


def _row(page: Page, section: str, name: str) -> Locator:
    page.goto(f"/workspace/{section}")
    page.get_by_role("textbox", name=f"Search {section.title()}").fill(name)
    row = page.get_by_role("main").get_by_role("button").filter(has_text=name)
    expect(row).to_be_visible()
    return row


def _menu_item(page: Page, row: Locator, trigger: str, item: str) -> None:
    row.hover()
    row.get_by_role("button", name=trigger).last.click()
    page.get_by_role("button", name=item, exact=True).click()


def _confirm(page: Page) -> None:
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()


def _model_menu_item(page: Page, model: dict, item: str) -> None:
    page.goto("/workspace/models")
    page.get_by_role("textbox", name="Search Models").fill(model["name"])
    expect(page.get_by_role("link", name=model["name"], exact=True)).to_be_visible()
    page.get_by_label(f"More: {model['name']}").click()
    page.get_by_role("button", name=item, exact=True).click()


def _model_in_selector(page: Page, name: str) -> Locator:
    """The selector's options for `name` in a fresh chat, once the model list has loaded."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(model_options(page, MOCK_MODEL_ID)).to_have_count(1)
    return model_options(page, name)


def _slash_menu(page: Page, typed: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


def _integration_tools(page: Page) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    return page.get_by_role("button")


def test_a_cloned_model_is_a_second_model_with_the_same_system_prompt(page_for, builder):
    original = _new_model(builder, system="Answer like a pirate.")
    page = page_for(builder)

    _model_menu_item(page, original, "Clone")
    page.get_by_role("button", name="Save & Create").click()
    expect(page).to_have_url(re.compile(r"/workspace/models$"))

    clone_name = f"{original['name']} (Clone)"
    stored = _fetch(builder, f"/api/v1/models/model?id={original['id']}-clone")
    assert stored["name"] == clone_name
    assert stored["params"]["system"] == "Answer like a pirate."
    expect(_model_in_selector(page, clone_name)).to_have_count(1)
    expect(model_options(page, original["name"])).to_have_count(1)


def test_a_hidden_model_leaves_the_selector_until_shown_again(page_for, builder):
    model = _new_model(builder)
    page = page_for(builder)
    expect(_model_in_selector(page, model["name"])).to_have_count(1)

    _model_menu_item(page, model, "Hide Model")
    expect(page.get_by_text(f"Model {model['id']} is now hidden")).to_be_visible()

    assert _fetch(builder, f"/api/v1/models/model?id={model['id']}")["meta"]["hidden"] is True
    expect(_model_in_selector(page, model["name"])).to_have_count(0)

    _model_menu_item(page, model, "Show Model")
    expect(page.get_by_text(f"Model {model['id']} is now visible")).to_be_visible()

    expect(_model_in_selector(page, model["name"])).to_have_count(1)


def test_a_deleted_model_leaves_the_list_and_the_selector(page_for, builder):
    kept, deleted = _new_model(builder), _new_model(builder)
    page = page_for(builder)

    _model_menu_item(page, deleted, "Delete")
    _confirm(page)
    expect(page.get_by_text(f"Deleted {deleted['id']}")).to_be_visible()

    page.reload()
    page.get_by_role("textbox", name="Search Models").fill("Menu model")
    expect(page.get_by_role("link", name=kept["name"], exact=True)).to_be_visible()
    expect(page.get_by_role("link", name=deleted["name"], exact=True)).to_have_count(0)
    expect(_model_in_selector(page, kept["name"])).to_have_count(1)
    expect(model_options(page, deleted["name"])).to_have_count(0)


def test_a_cloned_prompt_is_saved_with_its_own_command_and_offered_by_the_slash_menu(
    page_for, builder
):
    command = f"memo{uuid.uuid4().hex[:8]}"
    name = f"Prompt {command}"
    _create(
        builder,
        "/api/v1/prompts/create",
        {"command": command, "name": name, "content": "Write a short memo."},
    )
    page = page_for(builder)

    _menu_item(page, _row(page, "prompts", name), "Prompt Menu", "Clone")
    page.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_text("Prompt created successfully")).to_be_visible()

    clone_command = f"{command}-clone"
    copies = [p for p in _fetch(builder, "/api/v1/prompts/") if p["command"] == clone_command]
    assert [(copy["name"], copy["content"]) for copy in copies] == [
        (f"{name} (Clone)", "Write a short memo.")
    ]
    menu = _slash_menu(page, clone_command)
    expect(menu.get_by_role("button", name=clone_command)).to_be_visible()


def test_a_deleted_prompt_leaves_the_list_and_the_slash_menu(page_for, builder):
    prefix = f"note{uuid.uuid4().hex[:6]}"
    kept, deleted = f"{prefix}kept", f"{prefix}gone"
    for command in (kept, deleted):
        form = {"command": command, "name": f"Prompt {command}", "content": "Write a note."}
        _create(builder, "/api/v1/prompts/create", form)
    page = page_for(builder)

    _menu_item(page, _row(page, "prompts", f"Prompt {deleted}"), "Prompt Menu", "Delete")
    _confirm(page)
    expect(page.get_by_text(f"Deleted {deleted}")).to_be_visible()

    page.reload()
    page.get_by_role("textbox", name="Search Prompts").fill(prefix)
    expect(page.get_by_role("main").get_by_role("button").filter(has_text=kept)).to_be_visible()
    expect(page.get_by_role("main").get_by_role("button").filter(has_text=deleted)).to_have_count(0)
    menu = _slash_menu(page, prefix)
    expect(menu.get_by_role("button", name=kept)).to_be_visible()
    expect(menu.get_by_role("button", name=deleted)).to_have_count(0)


def test_a_version_in_the_prompt_history_can_be_deleted_but_not_the_live_one(page_for, builder):
    command = f"brief{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": f"Prompt {command}", "content": "Summarise in three lines."}
    prompt = _create(builder, "/api/v1/prompts/create", {**form, "commit_message": "First draft"})
    _create(
        builder,
        f"/api/v1/prompts/id/{prompt['id']}/update",
        {**form, "content": "Summarise in one line.", "commit_message": "Shorter"},
    )
    page = page_for(builder)
    page.goto(f"/workspace/prompts/{prompt['id']}")
    expect(page.get_by_role("textbox", name="Prompt Content")).to_have_value(
        "Summarise in one line."
    )
    picker = page.get_by_label("Select version", exact=True)
    expect(picker).to_have_text("Production")

    page.get_by_label("More Options").click()
    page.get_by_text("Delete", exact=True).hover()
    expect(page.get_by_text("Cannot delete the production version")).to_be_visible()
    expect(page.get_by_role("button", name="Delete", exact=True)).to_have_count(0)
    page.keyboard.press("Escape")

    picker.click()
    page.get_by_role("menuitemradio", name="First draft").click()
    expect(picker).to_have_text("First draft")
    page.get_by_label("More Options").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("dialog", name="Delete Version").get_by_role("button", name="Delete").click()
    expect(page.get_by_text("Version deleted")).to_be_visible()

    page.reload()
    picker.click()
    expect(page.get_by_role("menuitemradio", name="Production")).to_be_visible()
    expect(page.get_by_role("menuitemradio", name="First draft")).to_have_count(0)
    history = _fetch(builder, f"/api/v1/prompts/id/{prompt['id']}/history")
    assert [entry["commit_message"] for entry in history] == ["Shorter"]


def test_a_cloned_tool_is_saved_under_a_new_id_with_the_same_code(page_for, builder):
    tool = _new_tool(builder)
    page = page_for(builder)

    _menu_item(page, _row(page, "tools", tool["name"]), "Tool Menu", "Clone")
    page.get_by_role("main").get_by_role("button", name="Save & Create").click()
    _confirm(page)
    expect(page).to_have_url(re.compile(r"/workspace/tools$"))

    stored = _fetch(builder, f"/api/v1/tools/id/{tool['id']}_clone")
    assert stored["name"] == f"{tool['name']} (Clone)"
    assert stored["content"] == TOOL_SOURCE
    assert _fetch(builder, f"/api/v1/tools/id/{tool['id']}")["content"] == TOOL_SOURCE


def test_a_deleted_tool_leaves_the_list_and_the_integrations_menu(page_for, builder):
    kept, deleted = _new_tool(builder), _new_tool(builder)
    page = page_for(builder)

    _menu_item(page, _row(page, "tools", deleted["name"]), "Tool Menu", "Delete")
    _confirm(page)
    expect(page.get_by_text("Tool deleted successfully")).to_be_visible()

    page.reload()
    page.get_by_role("textbox", name="Search Tools").fill("Menu tool")
    expect(
        page.get_by_role("main").get_by_role("button").filter(has_text=kept["name"])
    ).to_be_visible()
    expect(
        page.get_by_role("main").get_by_role("button").filter(has_text=deleted["name"])
    ).to_have_count(0)
    tools = _integration_tools(page)
    expect(tools.filter(has_text=kept["name"])).to_be_visible()
    expect(tools.filter(has_text=deleted["name"])).to_have_count(0)


def test_a_cloned_skill_is_saved_with_the_same_instructions(page_for, builder):
    suffix = uuid.uuid4().hex[:8]
    skill = {"id": f"menu-skill-{suffix}", "name": f"Menu skill {suffix}"}
    form = {**skill, "content": "Answer in exactly three bullet points.", "meta": {}}
    _create(builder, "/api/v1/skills/create", {**form, "is_active": True})
    page = page_for(builder)

    _menu_item(page, _row(page, "skills", skill["name"]), "Skill Menu", "Clone")
    # a skill's clone is saved at once under a short random suffix and opened in the editor
    expect(page).to_have_url(re.compile(rf"/workspace/skills/edit\?id={skill['id']}-\w+$"))

    clone_id = page.url.split("id=")[1]
    stored = _fetch(builder, f"/api/v1/skills/id/{clone_id}")
    assert stored["name"] == f"{skill['name']} ({clone_id.rsplit('-', 1)[1]})"
    assert stored["content"] == "Answer in exactly three bullet points."
