"""Journey: a Python tool written in the workspace editor is called from a chat.

A fresh admin writes a tool in the workspace's Tools editor, turns it on for a chat under
Integrations, and asks a question the scripted model answers by calling the tool (native
function calling, the default). The tool runs on the server, the model is sent its result, and
the reply shows the call; opening it shows what the tool returned.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `/api/v1/tools/create` storing
the editor's source without its `specs` the model is never offered the tool.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOCKER_TOOL = """class Tools:
    def lookup_locker(self, number: int) -> str:
        \"\"\"Look up who holds a locker.

        :param number: the locker number
        \"\"\"
        return f"Locker {number} belongs to Ada, code {number * 111}."
"""


@pytest.fixture
def toolsmith(make_user):
    """A fresh admin; the tools it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")


def test_a_tool_from_the_editor_is_called_and_its_result_shows(page_for, toolsmith, upstream):
    name = f"Lockers {uuid.uuid4().hex[:6]}"
    page = page_for(toolsmith)
    page.goto("/workspace/tools/create")
    editor = page.get_by_role("main")
    editor.get_by_role("textbox", name="Tool Name").fill(name)
    editor.get_by_role("textbox", name="Tool Description").fill("who holds which locker")
    editor.locator(".cm-content").click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(LOCKER_TOOL)
    editor.get_by_role("button", name="Save & Create").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()
    expect(page).to_have_url(re.compile(r"/workspace/tools$"))

    question = "who holds locker 7?"
    upstream.queue(
        reply.tool_call("lookup_locker", {"number": 7}, match=reply.answering(question)),
        reply.text("Ada holds locker 7.", match=reply.answering(question)),
    )
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    page.get_by_role("button", name=name).click()
    page.keyboard.press("Escape")
    send(page, question)
    expect_reply(page, "Ada holds locker 7.")

    offered = next(filter(reply.answering(question), upstream.chat_requests()))
    assert "lookup_locker" in [tool["function"]["name"] for tool in offered.get("tools", [])]
    tool_results = [
        entry["content"]
        for entry in upstream.chat_requests()[-1]["messages"]
        if entry["role"] == "tool"
    ]
    assert tool_results == ["Locker 7 belongs to Ada, code 777."], tool_results

    conversation(page).get_by_text("View Result from lookup_locker").click()
    expect(conversation(page).get_by_text("Locker 7 belongs to Ada, code 777.")).to_be_visible()
