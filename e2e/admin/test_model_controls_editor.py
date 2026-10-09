"""Journey: an admin builds the model controls in Admin Settings > Models, users get them in chat.

Opening a model in Admin Settings > Models shows a Model controls section. Add control opens a
form for the name, the default option, a description and the options, each with its own custom
parameters (a new option starts with one blank parameter row, and Add parameter stays off while
a row is blank); Apply stays off while the name or an option name is blank. Saving the model
stores the controls, and a user's chat on the model then sends the default option's parameters,
with string values read as JSON. Editing a control keeps the keys of its options when they are
renamed, Remove takes a control away from the model and from the chat input, and the arrow keys
on a control's handle reorder the controls, in the store and in the chat input's menu. The
workspace model editor of the same model shows no Model controls section and saves the controls
it was given unchanged. Admin Settings opens a preset in the workspace editor, so the form is
tested on a model the scripted provider serves and the workspace editor on a preset.

Discriminates: passes on the dev 93fc3fcb7 build. In a frontend build whose Apply drops the
options' parameters and makes new option keys from the labels, whose Remove and reorder handles
do nothing, whose Apply is never disabled and whose workspace editor shows the section too, every
test fails. The workspace editor test was retargeted for 16849284f, whose editor saves only a
change: it renames the preset first, passes on dev 206bf9723 and fails in a build whose workspace
editor drops the controls on save.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.model_controls import LENGTH, THINKING, preset_with_controls
from harness.python_tools import EVERYONE_READS
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUICK_AND_DEEP = {
    "thinking": {
        "label": "Thinking",
        "default": "quick",
        "options": {
            "quick": {"label": "Quick", "params": {"reasoning_effort": "low"}},
            "deep": {"label": "Deep", "params": {"reasoning_effort": "high"}},
        },
    }
}


@pytest.fixture
def builder(make_user) -> Actor:
    return make_user(role="admin")


@pytest.fixture
def served_model(upstream, builder) -> Iterator[dict]:
    """A model the scripted provider serves, with settings saved for it and no controls."""
    suffix = uuid.uuid4().hex[:8]
    model = {"id": f"controls-base-{suffix}", "name": f"Lagoon base {suffix}"}
    upstream.models.append(model["id"])
    with builder.client() as client:
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": None,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield model
        client.post("/api/v1/models/model/delete", json={"id": model["id"]})


def store_controls(builder: Actor, model: dict, controls: dict) -> None:
    with builder.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
        stored["params"] = {**stored["params"], "model_controls": controls}
        updated = client.post(
            "/api/v1/models/model/update", params={"id": model["id"]}, json=stored
        )
    assert updated.status_code == 200, updated.text


def stored_controls(builder: Actor, model: dict) -> dict:
    with builder.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model["id"]})
    assert stored.status_code == 200, stored.text
    return stored.json()["params"].get("model_controls") or {}


def open_admin_editor(page: Page, model: dict) -> Locator:
    page.goto("/admin/settings/models")
    settings = page.get_by_role("dialog")
    settings.get_by_role("textbox", name="Search Models").fill(model["name"])
    row = settings.locator("#model-list > div").filter(has_text=model["name"])
    expect(row).to_have_count(1)
    row.get_by_role("button", name=model["name"]).first.click()
    expect(settings.get_by_text("Model controls", exact=True)).to_be_visible()
    return settings


def save_model(editor: Locator) -> None:
    saved = editor.page.expect_response(
        lambda response: (
            response.request.method == "POST" and "/models/model/update" in response.url
        )
    )
    with saved as response:
        editor.get_by_role("button", name="Save & Update").click()
    assert response.value.ok, response.value.status
    expect(editor.get_by_role("button", name="Save & Update")).to_have_count(0)


def control_form(page: Page) -> Locator:
    return page.get_by_role("dialog").filter(has=page.get_by_role("textbox", name="Control name"))


def option_block(form: Locator, index: int) -> Locator:
    """The part of the form for the option at `index`: its name, handle and parameters."""
    return form.get_by_role("textbox", name="Option name").nth(index).locator("xpath=../..")


def fill_parameters(block: Locator, parameters: dict[str, str]) -> None:
    """A new option starts with one blank parameter row, which takes the first parameter."""
    for index, (name, value) in enumerate(parameters.items()):
        if index > 0:
            block.get_by_role("button", name="Add Custom Parameter").click()
        block.get_by_role("textbox", name="Custom Parameter Name").last.fill(name)
        block.get_by_role("textbox", name="Custom Parameter Value").last.fill(value)
        block.get_by_role("textbox", name="Custom Parameter Value").last.blur()


def fill_option(form: Locator, index: int, label: str, parameters: dict[str, str]) -> None:
    if index > 0:
        form.get_by_role("button", name="Add option").click()
    form.get_by_role("textbox", name="Option name").nth(index).fill(label)
    fill_parameters(option_block(form, index), parameters)


def apply_control(form: Locator) -> None:
    form.get_by_role("button", name="Apply").click()
    expect(form).to_have_count(0)


def asked_in_chat(page: Page, upstream, model: dict) -> dict:
    """Send a message on the model; returns the provider request that carried it."""
    prompt = f"how warm is the lagoon? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Warm enough.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Warm enough.")
    return next(filter(reply.answering(prompt), upstream.chat_requests()))


def open_chat(page: Page, model: dict) -> None:
    page.goto(f"/?models={model['id']}")
    expect(page.get_by_role("button", name=f"Selected model: {model['name']}")).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def listed_controls(page: Page) -> list[str]:
    """The control names the chat input's Model controls menu lists, each with its option."""
    page.get_by_label("Model controls").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button").first).to_be_visible()
    return [text.replace("\n", " ") for text in menu.get_by_role("button").all_inner_texts()]


def test_an_added_control_is_stored_as_entered_and_its_default_is_sent(
    page_for, builder, make_user, served_model, upstream
):
    editor = open_admin_editor(page_for(builder), served_model)
    editor.get_by_role("button", name="Add control").click()
    form = control_form(editor.page)
    form.get_by_role("textbox", name="Control name").fill("Thinking")
    form.get_by_role("textbox", name="Description").fill("How long the model thinks")
    fill_option(form, 0, "Fast", {"reasoning_effort": "low", "temperature": "0.3"})
    fill_option(form, 1, "High", {"reasoning_effort": "high", "thinking": '{"type": "enabled"}'})
    form.get_by_role("combobox", name="Default option").select_option(label="High")
    apply_control(form)
    expect(editor.get_by_role("button", name="Edit Thinking")).to_contain_text(
        "Default option: High"
    )
    save_model(editor)

    assert stored_controls(builder, served_model) == {
        "thinking": {
            "label": "Thinking",
            "description": "How long the model thinks",
            "default": "high",
            "options": {
                "fast": {
                    "label": "Fast",
                    "params": {"reasoning_effort": "low", "temperature": "0.3"},
                },
                "high": {
                    "label": "High",
                    "params": {"reasoning_effort": "high", "thinking": '{"type": "enabled"}'},
                },
            },
        }
    }
    page = page_for(make_user())
    open_chat(page, served_model)
    sent = asked_in_chat(page, upstream, served_model)
    assert sent["reasoning_effort"] == "high"
    assert sent["thinking"] == {"type": "enabled"}
    assert "temperature" not in sent


def test_an_edited_control_keeps_its_option_keys_and_sends_the_new_default(
    page_for, builder, make_user, served_model, upstream
):
    store_controls(builder, served_model, QUICK_AND_DEEP)
    editor = open_admin_editor(page_for(builder), served_model)
    editor.get_by_role("button", name="Edit Thinking").click()
    form = control_form(editor.page)
    form.get_by_role("textbox", name="Option name").nth(1).fill("Deeper")
    value = option_block(form, 1).get_by_role("textbox", name="Custom Parameter Value")
    value.fill("medium")
    form.get_by_role("combobox", name="Default option").select_option(label="Deeper")
    apply_control(form)
    expect(editor.get_by_role("button", name="Edit Thinking")).to_contain_text(
        "Default option: Deeper"
    )
    save_model(editor)

    assert stored_controls(builder, served_model) == {
        "thinking": {
            "label": "Thinking",
            "default": "deep",
            "options": {
                "quick": {"label": "Quick", "params": {"reasoning_effort": "low"}},
                "deep": {"label": "Deeper", "params": {"reasoning_effort": "medium"}},
            },
        }
    }
    page = page_for(make_user())
    open_chat(page, served_model)
    assert asked_in_chat(page, upstream, served_model)["reasoning_effort"] == "medium"


def test_a_removed_control_leaves_the_model_and_the_chat_input(
    page_for, builder, make_user, served_model
):
    store_controls(builder, served_model, {"thinking": THINKING})
    user_page = page_for(make_user())
    open_chat(user_page, served_model)
    expect(user_page.get_by_label("Model controls")).to_be_visible()
    editor = open_admin_editor(page_for(builder), served_model)
    editor.get_by_role("button", name="Remove Thinking").click()
    expect(editor.get_by_role("button", name="Edit Thinking")).to_have_count(0)
    save_model(editor)

    assert stored_controls(builder, served_model) == {}
    open_chat(user_page, served_model)
    expect(user_page.get_by_label("Model controls")).to_have_count(0)


def test_controls_reordered_with_the_keyboard_are_stored_and_listed_in_that_order(
    page_for, builder, make_user, served_model
):
    store_controls(builder, served_model, {"thinking": THINKING, "length": LENGTH})
    editor = open_admin_editor(page_for(builder), served_model)
    handle = editor.get_by_role("button", name="Reorder Thinking")
    handle.focus()
    handle.press("ArrowDown")
    rows = editor.get_by_role("button", name=re.compile("^Edit "))
    expect(rows.first).to_have_attribute("aria-label", "Edit Answer length")
    save_model(editor)

    assert list(stored_controls(builder, served_model)) == ["length", "thinking"]
    page = page_for(make_user())
    open_chat(page, served_model)
    assert listed_controls(page) == ["Answer length Default", "Thinking Medium"]


def test_apply_stays_off_while_the_control_name_or_an_option_name_is_blank(
    page_for, builder, served_model
):
    editor = open_admin_editor(page_for(builder), served_model)
    editor.get_by_role("button", name="Add control").click()
    form = control_form(editor.page)
    apply = form.get_by_role("button", name="Apply")
    name = form.get_by_role("textbox", name="Control name")
    option = form.get_by_role("textbox", name="Option name")
    expect(apply).to_be_disabled()

    name.fill("Thinking")
    expect(apply).to_be_disabled()
    option.fill("Fast")
    expect(apply).to_be_enabled()

    name.fill("   ")
    expect(apply).to_be_disabled()
    name.fill("Thinking")
    option.fill("")
    expect(apply).to_be_disabled()
    option.fill("Fast")
    form.get_by_role("button", name="Add option").click()
    expect(apply).to_be_disabled()
    option.nth(1).fill("Slow")
    expect(apply).to_be_enabled()


def test_the_workspace_editor_shows_no_controls_and_saves_them_unchanged(page_for, builder):
    controls = {"thinking": THINKING, "length": LENGTH}
    with preset_with_controls(builder, controls) as preset:
        page = page_for(builder)
        page.goto(f"/workspace/models/edit?id={preset['id']}")
        editor = page.get_by_role("main")
        expect(editor.get_by_placeholder("Model Name")).to_have_value(preset["name"])
        expect(editor.get_by_text("Advanced Params")).to_be_visible()
        expect(editor.get_by_text("Model controls", exact=True)).to_have_count(0)
        expect(editor.get_by_role("button", name="Add control")).to_have_count(0)
        # the editor saves only a change, so rename the preset
        editor.get_by_placeholder("Model Name").fill(f"{preset['name']} renamed")
        editor.get_by_role("button", name="Save & Update").click()
        expect(page).to_have_url(re.compile(r"/workspace/models/?$"))

        assert stored_controls(builder, preset) == controls
