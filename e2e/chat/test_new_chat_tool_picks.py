"""Regression: a tool picked in a new chat was dropped after leaving the chat and coming back.

Issue open-webui/open-webui#32084. A user opens a new chat, picks a tool from the More menu's
integrations and sends the first message. Opening another chat from the sidebar and going back, or
reloading the page, used to drop the tool: the chat fell back to the model's default tools, so the
next message went out without it. Picks made later in an existing chat were always kept. The menu's
Tools entry counts the picks.

Discriminates: both tests fail on the dev ee1d9255f build (the next message carries no picked
tool) and pass with the picks saved for the chat once its first message creates it.

A tool picked in an opened saved chat stays after a reload, in the main chat and dragged into a
tab beside a new chat or a saved one (`utils/chat_surface.py`). A chat dragged into a tab dropped
what was picked there (open-webui/open-webui#34034, fix PR open-webui/open-webui#34064): those tab
variants are red on dev c15f28649.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.python_tools import python_tool
from utils.cached_chat import turn_on_tool
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT
from utils.chat_ui import chat_input, expect_picked, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PUFFIN_TOOL = '''
class Tools:
    def count_puffins(self, cliff: str) -> str:
        """Count the puffins nesting on a cliff."""
        return "forty"
'''


@pytest.fixture
def puffin_tool(admin) -> str:
    name = f"Puffin counter {uuid.uuid4().hex[:8]}"
    with python_tool(admin, PUFFIN_TOOL, name=name):
        yield name


def pick_tool(page: Page, name: str) -> None:
    turn_on_tool(page, name)
    expect_picked(page, "Tools", 1)


def ask(page: Page, upstream, question: str) -> dict:
    """Send `question` and return the request the model got for it."""
    upstream.queue(reply.text(f"Answer to {question}", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, f"Answer to {question}")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def offered_tools(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def start_chat_with_tool(page: Page, upstream, tool: str) -> str:
    """A new chat whose first message went out with `tool` picked; returns the chat's id."""
    page.goto("/")
    pick_tool(page, tool)
    first = ask(page, upstream, f"first question {uuid.uuid4().hex[:6]}")
    assert "count_puffins" in offered_tools(first)
    page.wait_for_url(re.compile(r"/c/"))
    return page.url.rsplit("/c/", 1)[1]


def open_from_sidebar(page: Page, chat_id: str) -> None:
    if page.get_by_role("button", name="Open Sidebar", exact=True).is_visible():
        page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.locator(f'a[href="/c/{chat_id}"]').click()
    expect(page).to_have_url(re.compile(f"/c/{chat_id}$"))
    expect(chat_input(page)).to_be_visible()


def test_a_tool_picked_in_a_new_chat_stays_after_switching_chats(
    page_for, user, upstream, puffin_tool
):
    page = page_for(user)
    other_question = f"other chat {uuid.uuid4().hex[:6]}"
    ask(page, upstream, other_question)
    page.wait_for_url(re.compile(r"/c/"))
    other_chat = page.url.rsplit("/c/", 1)[1]

    picked_chat = start_chat_with_tool(page, upstream, puffin_tool)
    open_from_sidebar(page, other_chat)
    open_from_sidebar(page, picked_chat)

    expect_picked(page, "Tools", 1)
    follow_up = ask(page, upstream, f"follow-up {uuid.uuid4().hex[:6]}")
    assert "count_puffins" in offered_tools(follow_up)


def test_a_tool_picked_in_a_new_chat_stays_after_a_reload(page_for, user, upstream, puffin_tool):
    page = page_for(user)
    start_chat_with_tool(page, upstream, puffin_tool)

    page.reload()

    expect_picked(page, "Tools", 1)
    follow_up = ask(page, upstream, f"follow-up {uuid.uuid4().hex[:6]}")
    assert "count_puffins" in offered_tools(follow_up)


def _saved_chat(account) -> str:
    """A saved chat of `account` with one answered question and no tool picked; its id."""
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "How many puffins nest here?"},
                {"role": "assistant", "content": "Quite a few."},
            ],
        )
    return chat_id


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_tool_picked_in_an_opened_saved_chat_stays_after_a_reload(
    chat_for, make_user, upstream, puffin_tool
):
    """In a tab: open-webui/open-webui#34034."""
    account = make_user()
    chat = chat_for(account, chat_id=_saved_chat(account))
    pick_tool(chat, puffin_tool)
    first = ask(chat, upstream, f"with the tool {uuid.uuid4().hex[:6]}")
    assert "count_puffins" in offered_tools(first), f"the {chat.place} sent without the tool"

    chat = chat.reload()

    expect_picked(chat, "Tools", 1)
    follow_up = ask(chat, upstream, f"after the reload {uuid.uuid4().hex[:6]}")
    assert "count_puffins" in offered_tools(follow_up), "the tool picked in a tab was dropped"
