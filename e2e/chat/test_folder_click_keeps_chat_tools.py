"""Regression: clicking a folder in the sidebar gave the open chat the folder model's tools.

Issue open-webui/open-webui#30226, fix 15f350b41 (PR open-webui/open-webui#30376). A folder can
name default models for the chats started in it. Clicking a folder's name while a chat was open
switched that chat to the folder's model for a moment before the folder page opened, which reset
the chat's tools to that model's defaults and saved them as the chat's draft. Back in the chat,
the next message went out with the folder model's tools. The folder's models now only apply to a
chat that has no messages yet. Since dev da850248d a folder click adds no browser history entry,
so the tests go back to the chat through its row in the sidebar.

Discriminates: passes on the dev efe63bd34 build, fails on that build with 15f350b41 reverted
(the next message carries the folder model's tool).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, send
from utils.sidebar_chats import chat_row
from utils.sidebar_folders import folder_link

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER = "Research"
TOOL_SOURCE = '''
class Tools:
    def lookup(self, term: str) -> str:
        """Look a term up."""
        return term
'''


def _tool_model(admin, tool_id: str, prefix: str):
    """A model every account may use whose default tool is `tool_id`."""
    model_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {"toolIds": [tool_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def folder_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Folder lookup") as tool_id:
        yield tool_id


@pytest.fixture
def chat_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Chat lookup") as tool_id:
        yield tool_id


@pytest.fixture
def folder_model(admin, folder_tool):
    yield from _tool_model(admin, folder_tool, "folder-model")


@pytest.fixture
def chat_model(admin, chat_tool):
    yield from _tool_model(admin, chat_tool, "chat-model")


@pytest.fixture
def owner(make_user, folder_model):
    """An account with a folder whose chats default to `folder_model`."""
    account = make_user()
    with account.client() as client:
        created = client.post(
            "/api/v1/folders/", json={"name": FOLDER, "data": {"model_ids": [folder_model]}}
        )
        assert created.status_code == 200, created.text
    return account


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    return sidebar


def visit_folder_and_return(page: Page, title: str, last_reply: str) -> None:
    """Click the folder's name, then go back to the chat titled `title` from the sidebar."""
    sidebar = open_sidebar(page)
    chat_url = page.url
    folder_link(sidebar, FOLDER).click()
    expect(page).to_have_url(re.compile(r"/folders/"))
    page.wait_for_timeout(1000)  # a person's glance at the folder; the draft saves after 500 ms
    chat_row(sidebar, title).click()
    expect(page).to_have_url(chat_url)
    # the URL changes before the chat replaces the folder page, whose input would start a new chat
    expect_reply(page, last_reply)
    expect(chat_input(page)).to_be_visible()


def sent_payload(page: Page, text: str) -> dict:
    """Send `text` in the open chat; returns what the page posted to the server."""
    with page.expect_request(
        lambda request: request.url.endswith("/api/chat/completions") and request.method == "POST"
    ) as posted:
        send(page, text)
    return json.loads(posted.value.post_data or "{}")


def test_an_open_chat_keeps_its_own_tools_after_a_folder_click(
    page_for, owner, upstream, chat_model, chat_tool
):
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}],
            model=chat_model,
        )
    page = page_for(owner, path=f"/c/{chat_id}")
    expect(page.get_by_text("hi there")).to_be_visible()

    visit_folder_and_return(page, "Seeded chat", "hi there")
    upstream.queue(reply.text("still here", match=reply.answering("and now?")))
    payload = sent_payload(page, "and now?")

    assert payload["model"] == chat_model
    assert payload.get("tool_ids") == [chat_tool]
    expect_reply(page, "still here")


def test_a_chat_started_on_the_home_page_keeps_its_tools(page_for, owner, upstream, folder_tool):
    page = page_for(owner)
    upstream.queue(reply.text("first answer", match=reply.answering("first question")))
    send(page, "first question")
    expect_reply(page, "first answer")
    expect(page).to_have_url(re.compile(r"/c/"))

    visit_folder_and_return(page, "first question", "first answer")
    upstream.queue(reply.text("second answer", match=reply.answering("second question")))
    payload = sent_payload(page, "second question")

    assert payload["model"] == MOCK_MODEL_ID
    assert folder_tool not in (payload.get("tool_ids") or [])


def test_a_new_chat_in_the_folder_still_uses_the_folder_model_and_its_tools(
    page_for, owner, upstream, folder_model, folder_tool
):
    page = page_for(owner)
    folder_link(open_sidebar(page), FOLDER).click()
    expect(page).to_have_url(re.compile(r"/folders/"))

    upstream.queue(reply.text("in the folder", match=reply.answering("new in folder")))
    payload = sent_payload(page, "new in folder")

    assert payload["model"] == folder_model
    assert folder_tool in (payload.get("tool_ids") or [])
