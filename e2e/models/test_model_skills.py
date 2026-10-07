"""Journey: skills attached to a model in the editor, and what a chat on that model does with them.

A fresh admin attaches a skill in the model editor's Skills section and saves. A chat on that
model then carries the skill's instructions in the system prompt without any mention, and a chat
on the plain scripted model carries none. Detaching the skill in the editor and saving stops it
reaching the next chat, and a skill switched off after it was attached is not applied. A model
shared with everyone that carries a skill the chatting account may not read gives that account
none of its instructions, while a skill shared with the account does reach it. Where the model
uses native function calling with builtin tools, the system prompt lists the attached skill by
name and description only, the provider is offered the skill viewer, and when the model calls it
the chat shows the call and the skill's instructions reach the provider in the tool result.

Discriminates: passes on dev ebc6add67; in a backend copy whose chat middleware ignores the skills
a chat sends along for its model, the attach, detach, switched-off and unreadable-skill tests
fail; in one that no longer filters skills by what the chatting account may read, the
unreadable-skill test fails; in one that no longer skips switched-off skills, the switched-off
test fails; in one whose skill viewer answers an error or whose system prompt leaves out the
skill list, the native test fails.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NO_BUILTIN_TOOLS = {"builtin_tools": False}


def unique(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models and skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for skill in client.get("/api/v1/skills/list").json().get("items", []):
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


@pytest.fixture
def make_skill(builder) -> Callable[..., dict]:
    """`make_skill(instructions, grants=[])` is a skill the builder made, with a unique name."""

    def create(instructions: str, grants: list[dict] | None = None) -> dict:
        skill = {"id": f"dock-{uuid.uuid4().hex[:8]}", "name": unique("Dock manners")}
        with builder.client() as client:
            created = client.post(
                "/api/v1/skills/create",
                json={
                    **skill,
                    "description": f"How to behave at the dock {skill['id']}",
                    "content": instructions,
                    "meta": {},
                    "access_grants": grants or [],
                },
            )
        assert created.status_code == 200, created.text
        return skill

    return create


@pytest.fixture
def make_model(builder) -> Callable[..., dict]:
    """`make_model(**meta)` is a preset on the scripted model that every account reads."""

    def create(params: dict | None = None, capabilities: dict | None = None) -> dict:
        model = {"id": f"skilled-{uuid.uuid4().hex[:8]}", "name": unique("Dockmaster")}
        with builder.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    **model,
                    "base_model_id": MOCK_MODEL_ID,
                    "meta": {"capabilities": capabilities or {}},
                    "params": params or {},
                    "access_grants": [EVERYONE_READS],
                },
            )
        assert created.status_code == 200, created.text
        return model

    return create


def open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def save(editor: Locator) -> None:
    editor.get_by_role("button", name="Save & Update").click()
    expect(editor.page).to_have_url(re.compile(r"/workspace/models/?$"))


def attach_in_editor(page: Page, model: dict, skill: dict) -> None:
    """Tick the skill in the model editor's Skills section and save."""
    editor = open_editor(page, model)
    editor.get_by_text("Select Skill", exact=True).click()
    page.get_by_placeholder("Search skills").fill(skill["name"])
    page.get_by_role("button", name=skill["name"], exact=True).click()
    page.keyboard.press("Escape")
    expect(editor.get_by_role("checkbox", name=skill["name"])).to_be_checked()
    save(editor)


def detach_in_editor(page: Page, model: dict, skill: dict) -> None:
    editor = open_editor(page, model)
    checkbox = editor.get_by_role("checkbox", name=skill["name"])
    expect(checkbox).to_be_checked()
    checkbox.click()
    expect(checkbox).to_have_count(0)
    save(editor)


def open_chat_on(page: Page, model_id: str, name: str) -> None:
    page.goto(f"/?model={model_id}")
    expect(page.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def system_prompt_of(page: Page, upstream, question: str) -> str:
    """Send `question` in the open chat; the system prompt the provider got with it."""
    upstream.queue(reply.text("noted", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "noted")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return system_text(request)


def test_a_skill_attached_in_the_editor_reaches_a_chat_on_that_model(
    page_for, builder, make_model, make_skill, upstream
):
    instructions = unique("Tie off the bow line first.")
    model, skill = make_model(capabilities=NO_BUILTIN_TOOLS), make_skill(instructions)
    page = page_for(builder)
    attach_in_editor(page, model, skill)

    open_chat_on(page, model["id"], model["name"])
    attached = system_prompt_of(page, upstream, "how do I moor?")
    open_chat_on(page, MOCK_MODEL_ID, MOCK_MODEL_ID)
    plain = system_prompt_of(page, upstream, "how do I moor here?")

    assert f'<skill name="{skill["name"]}">' in attached, attached
    assert instructions in attached, attached
    assert instructions not in plain, plain


def test_a_skill_detached_in_the_editor_stops_reaching_the_chat(
    page_for, builder, make_model, make_skill, upstream
):
    instructions = unique("Coil the stern line twice.")
    model, skill = make_model(capabilities=NO_BUILTIN_TOOLS), make_skill(instructions)
    page = page_for(builder)
    attach_in_editor(page, model, skill)
    open_chat_on(page, model["id"], model["name"])
    assert instructions in system_prompt_of(page, upstream, "how do I moor?")

    detach_in_editor(page, model, skill)

    open_chat_on(page, model["id"], model["name"])
    after = system_prompt_of(page, upstream, "and how do I leave?")
    assert instructions not in after, after


def test_a_switched_off_attached_skill_is_not_applied(
    page_for, builder, make_model, make_skill, upstream
):
    on_text, off_text = unique("Fenders go out first."), unique("Radio the harbour office.")
    model = make_model(capabilities=NO_BUILTIN_TOOLS)
    kept, switched_off = make_skill(on_text), make_skill(off_text)
    page = page_for(builder)
    attach_in_editor(page, model, kept)
    attach_in_editor(page, model, switched_off)
    with builder.client() as client:
        toggled = client.post(f"/api/v1/skills/id/{switched_off['id']}/toggle")
    assert toggled.status_code == 200 and toggled.json()["is_active"] is False, toggled.text

    open_chat_on(page, model["id"], model["name"])
    prompt = system_prompt_of(page, upstream, "how do I moor?")

    assert on_text in prompt, prompt
    assert off_text not in prompt, prompt


def test_a_skill_the_chatting_account_may_not_read_is_not_applied(
    page_for, builder, make_user, make_model, make_skill, upstream
):
    private_text, shared_text = unique("Private harbour code."), unique("Shared berth rules.")
    model = make_model(capabilities=NO_BUILTIN_TOOLS)
    private, shared = make_skill(private_text), make_skill(shared_text, [EVERYONE_READS])
    admin_page = page_for(builder)
    attach_in_editor(admin_page, model, private)
    attach_in_editor(admin_page, model, shared)

    page = page_for(make_user())
    open_chat_on(page, model["id"], model["name"])
    prompt = system_prompt_of(page, upstream, "how do I moor?")

    assert shared_text in prompt, prompt
    assert private_text not in prompt, prompt


def test_native_function_calling_lists_the_skill_and_the_viewer_loads_it(
    page_for, builder, make_model, make_skill, upstream
):
    instructions = unique("Raise the green flag at dusk.")
    model = make_model(params={"function_calling": "native"})
    skill = make_skill(instructions)
    page = page_for(builder)
    attach_in_editor(page, model, skill)
    open_chat_on(page, model["id"], model["name"])
    question = "what does the dock expect?"

    upstream.queue(
        reply.tool_call("view_skill", {"id": skill["id"]}, match=reply.answering(question)),
        reply.text("Flags at dusk.", match=reply.answering(question)),
    )
    send(page, question)
    expect_reply(page, "Flags at dusk.")

    answered = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    first, last = answered[0], answered[-1]
    manifest = system_text(first)
    offered = {tool["function"]["name"] for tool in first.get("tools") or []}
    tool_results = json.dumps([entry for entry in last["messages"] if entry["role"] == "tool"])
    assert "<available_skills>" in manifest, manifest
    assert skill["name"] in manifest and f"How to behave at the dock {skill['id']}" in manifest
    assert instructions not in manifest, manifest
    assert "view_skill" in offered, sorted(offered)
    assert instructions in tool_results, tool_results
    expect(
        last_reply(page).get_by_role("button", name="View Result from view_skill")
    ).to_be_visible()
