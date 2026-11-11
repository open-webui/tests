"""Journey: a question waiting when the page reloads comes back, and its answer finishes the reply.

An Ask User question and a tool approval in a saved chat are stored with the chat, so the
reloaded page shows the card again and answering it carries the reply on. A tool's own input
dialog lives only on the socket: since 0.12.4 (c596d40bd) the server waits up to a minute for the
reloaded page to open the chat again and asks it there.

Twin of integration/chat/test_questions_survive_reconnect.py.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), reloaded with the
side tab selected, and the input dialog test in every way a side chat starts and every state it
can be in but a temporary one, which is gone after a reload. The reload opens a side chat as the
page's own chat, so the question comes back there; every variant passes on dev c15f28649 and on dev
e791cbdde.

Discriminates: passes on the dev 948432b43 build. With the resume turned off in a backend copy
(no call may wait for a reconnect) the input dialog test fails (no dialog after the reload, and
the tool gets "Client session disconnected."). With the waiting call no longer saved with the
chat the Ask User and approval tests fail (never stored as waiting; with that wait skipped, the
card is gone after the reload).
"""

from __future__ import annotations

import json
import re
import time
import uuid

import httpx
import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.python_tools import python_tool
from utils.cached_chat import turn_on_tool
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE, ChatSurface
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    send,
    set_tool_approval_mode,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REJECTED = "Error: tool call rejected by user."

PLAN_QUESTION = {
    "id": "plan",
    "header": "Plan",
    "question": "Which plan fits you best?",
    "options": [
        {"label": "Basic", "description": "The cheapest one"},
        {"label": "Pro", "description": "The fastest one"},
    ],
}

ASK_THE_PAGE = '''
import json


class Tools:
    async def ask_the_page(self, __event_call__=None) -> str:
        """
        Ask the user for their name.
        """
        answer = await __event_call__(
            {"type": "input", "data": {"title": "Your name?", "message": "For the greeting."}}
        )
        return json.dumps(answer)
'''


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True})
    assert saved.status_code == 200, saved.text


def open_chat_id(page: Page) -> str:
    page.wait_for_url(re.compile(r"/c/[^/?#]+"), timeout=REPLY_TIMEOUT_MS)
    return re.search(r"/c/([^/?#]+)", page.url).group(1)


def wait_until_stored_as_pending(client: httpx.Client, chat_id: str, tool: str) -> None:
    """The reload must find the question in the chat, so wait for the server to have saved it."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        for message in messages.values():
            for item in message.get("output") or []:
                if item.get("name") == tool and item.get("status") == "pending":
                    return
        time.sleep(0.1)
    raise AssertionError(f"the {tool} call was never stored as waiting for the user")


def reload_chat(chat: ChatSurface) -> ChatSurface:
    chat = chat.reload()
    expect(chat_input(chat)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return chat


def tool_results(upstream, prompt: str) -> list[str]:
    final = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)][-1]
    return [str(entry["content"]) for entry in reply.conversation(final) if entry["role"] == "tool"]


def test_ask_user_comes_back_after_a_reload_and_its_answer_finishes_the_reply(
    chat_for, make_user, upstream
):
    account: Actor = make_user()
    chat = chat_for(account)
    prompt = f"help me choose a plan {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("ask_user", {"questions": [PLAN_QUESTION]}, match=reply.answering(prompt)),
        reply.text("Pro it is.", match=reply.answering(prompt)),
    )
    send(chat, prompt)
    expect(chat.panel.get_by_text("Which plan fits you best?")).to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    with account.client() as client:
        wait_until_stored_as_pending(client, open_chat_id(chat.page), "ask_user")

    chat = reload_chat(chat)
    question = chat.panel.get_by_text("Which plan fits you best?")
    expect(question, "the Ask User card is gone after the reload").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    card = chat.panel.locator("section", has=chat.page.get_by_text("Which plan fits you best?"))
    card.get_by_role("button", name=re.compile(r"^Pro\b")).click()

    expect_reply(chat, "Pro it is.")
    [result] = tool_results(upstream, prompt)
    assert json.loads(result)["answers"]["plan"]["label"] == "Pro", result


def test_a_tool_approval_comes_back_after_a_reload_and_allowing_it_finishes_the_reply(
    tool_approval_on, chat_for, make_user, upstream
):
    account: Actor = make_user()
    chat = chat_for(account)
    set_tool_approval_mode(chat, "Ask for approval")
    prompt = f"what time is it? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(prompt)),
        reply.text("Checked the clock.", match=reply.answering(prompt)),
    )
    send(chat, prompt)
    allow = conversation(chat).get_by_role("button", name="Allow", exact=True)
    expect(allow).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    with account.client() as client:
        wait_until_stored_as_pending(client, open_chat_id(chat.page), "get_current_timestamp")

    chat = reload_chat(chat)
    allow = conversation(chat).get_by_role("button", name="Allow", exact=True)
    expect(allow, "the approval card is gone after the reload").to_have_count(
        1, timeout=REPLY_TIMEOUT_MS
    )
    allow.click()

    expect_reply(chat, "Checked the clock.")
    [result] = tool_results(upstream, prompt)
    assert result != REJECTED and result.strip(), result


# a temporary side chat is gone after the reload; several models: the reload asks the one tool
@pytest.mark.chat_surfaces(*(place for place in EVERYWHERE if place != TEMPORARY_SIDE))
def test_a_tools_input_dialog_is_asked_again_after_a_reload_and_its_answer_finishes_the_reply(
    admin, chat_for, make_user, upstream
):
    prompt = f"who am I? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("ask_the_page", {}, match=reply.answering(prompt)),
        reply.text("Nice to meet you, Ada.", match=reply.answering(prompt)),
    )
    with python_tool(admin, ASK_THE_PAGE, name="Ask the page"):
        chat = chat_for(make_user())
        turn_on_tool(chat, "Ask the page")
        send(chat, prompt)
        expect(chat.page.get_by_role("dialog", name="Your name?")).to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )

        chat = reload_chat(chat)
        dialog = chat.page.get_by_role("dialog", name="Your name?")
        expect(dialog, "the tool's question was not asked again after the reload").to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )
        dialog.get_by_label("Enter your message").fill("Ada")
        dialog.get_by_role("button", name="Confirm").click()

        expect_reply(chat, "Nice to meet you, Ada.")
        expect(dialog).to_have_count(0)
    assert tool_results(upstream, prompt) == [json.dumps("Ada")]
