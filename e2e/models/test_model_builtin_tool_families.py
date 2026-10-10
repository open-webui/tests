"""Journey: each Builtin Tools category unticked in the model editor leaves exactly its tools out.

A fresh admin's preset on the scripted model has every capability on but File Context (so a file
attached in the chat is read through the file tools), Web Search and Image Generation as default
features, and the admin has channels, user webhooks, web search and image generation on, so a chat
on it offers every builtin tool category the instance has. The admin opens a chat on the preset
with a file attached and notes what the model is offered, unticks one category under Builtin Tools
in the model editor and asks again: that category's tools are gone and every other tool is still
offered. The Time & Calculation category and the Builtin Tools capability as a whole are covered in
e2e/models/test_model_editor.py, Sub-agents in e2e/admin/test_subagents_settings.py; the Code
Interpreter is legacy and left out.

Discriminates: passes on the dev ebc6add67 build. In a backend copy whose builtin tools ignore the
model's unticked categories every case fails (the category's tools are still offered); in a
frontend build whose editor saves the Builtin Tools categories as they were loaded every case
fails as well.
Retargeted for 8d0ff76f2, whose Builtin Tools categories are switches in a section that opens on a
click: passes on dev 76ad6f97c (3 of 3), and in a build of it whose editor saves the model's
settings as they were loaded every case fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.channel_chat import serve_openai_images
from harness.image_engines import IMAGES_CONFIG, save_image_settings
from harness.prompt_caching import ADMIN_CONFIG, CAPABILITIES
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.cached_chat import attach
from utils.model_editor import (
    offered_tool_names,
    open_chat_on,
    open_editor,
    save,
    sent_request,
    set_checkbox,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# editor label, and every tool that category offers on this setup
FAMILIES = {
    "ask user": ("Ask User", {"ask_user"}),
    "memory": (
        "Memory",
        {
            "search_memories",
            "list_memory_paths",
            "read_memory_path",
            "list_memories",
            "update_memory",
            "add_memory",
            "replace_memory_content",
            "delete_memory",
        },
    ),
    "chat history": ("Chat History", {"search_chats", "view_chat"}),
    "notes": ("Notes", {"search_notes", "view_note", "write_note", "replace_note_content"}),
    "knowledge": (
        "Knowledge Base",
        {
            "list_knowledge_bases",
            "search_knowledge_bases",
            "query_knowledge_bases",
            "grep_knowledge_files",
            "search_knowledge_files",
            "query_knowledge_files",
            "view_knowledge_file",
        },
    ),
    "files": ("Files", {"list_chat_files", "query_chat_files", "grep_chat_files", "view_file"}),
    "channels": (
        "Channels",
        {
            "search_channels",
            "search_channel_messages",
            "view_channel_thread",
            "view_channel_message",
        },
    ),
    "notifications": ("Notifications", {"notify"}),
    "web search": ("Web Search", {"search_web", "fetch_url"}),
    "image generation": ("Image Generation", {"generate_image", "edit_image"}),
    "tasks": ("Task Management", {"create_tasks", "update_task"}),
    "automations": (
        "Automations",
        {
            "create_automation",
            "update_automation",
            "list_automations",
            "toggle_automation",
            "delete_automation",
        },
    ),
    "calendar": (
        "Calendar",
        {
            "search_calendar_events",
            "create_calendar_event",
            "update_calendar_event",
            "delete_calendar_event",
        },
    ),
}
HARBOUR_NOTES = ("harbour.txt", "Berth 3 is reserved for the pilot boat.")


@pytest.fixture
def every_category_on(admin, preserve, listener) -> None:
    """Channels, user webhooks, web search and image generation on for the instance."""
    preserve("admin_config", RETRIEVAL_CONFIG, IMAGES_CONFIG)
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        enabled = {**current, "ENABLE_CHANNELS": True, "ENABLE_USER_WEBHOOKS": True}
        saved = client.post(ADMIN_CONFIG, json=enabled)
        assert saved.status_code == 200, saved.text
        save_web_settings(client, **serve_search_results(listener, []))
        save_image_settings(client, **serve_openai_images(listener))


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
def preset(builder, every_category_on) -> dict:
    """Every capability on but File Context, with Web Search and Image Generation by default."""
    model = {
        "id": f"toolbox-{uuid.uuid4().hex[:8]}",
        "name": f"Harbour toolbox {uuid.uuid4().hex[:6]}",
    }
    meta = {
        "capabilities": CAPABILITIES,
        "defaultFeatureIds": ["web_search", "image_generation"],
    }
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": meta,
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    return model


def offered_with_a_file(page: Page, upstream, model: dict, question: str) -> set[str]:
    open_chat_on(page, model)
    attach(page, *HARBOUR_NOTES)
    expect(page.locator("form").get_by_text(HARBOUR_NOTES[0])).to_be_visible()
    return offered_tool_names(sent_request(page, upstream, question))


@pytest.mark.parametrize("family", FAMILIES)
def test_an_unticked_category_leaves_only_its_tools_out(
    page_for, builder, preset, upstream, family
):
    label, tools = FAMILIES[family]
    page = page_for(builder)
    before = offered_with_a_file(page, upstream, preset, "which berth is free?")
    assert tools <= before, f"not offered while ticked: {sorted(tools - before)}"

    editor = open_editor(page, preset)
    set_checkbox(editor, "Builtin Tools", label, False)
    save(editor)

    after = offered_with_a_file(page, upstream, preset, "which berth is free now?")
    assert not tools & after, f"still offered once unticked: {sorted(tools & after)}"
    lost, gained = before - tools - after, after - before
    assert not lost and not gained, (
        f"other tools changed: lost {sorted(lost)}, gained {sorted(gained)}"
    )
