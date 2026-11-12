"""Journey: another chat referenced from the composer's Reference Chats reaches the model.

The composer's More menu lists the account's chats under Reference Chats, searched by title. The
chat picked is attached to the message and sent with it, so the model can open that conversation
(with its chat tools since bd6fc15be, which keeps File Context off with native function calling),
and the reference stays on the message after a reload. The open chat is not offered as a reference
to itself.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose chat references read no messages
the reaching test fails, and in a frontend build whose picker also lists the open chat the other
test fails.

Both tests run in the main chat and in a side chat (`utils/chat_surface.py`), the open chat
dragged into a tab beside a new chat for the second. That side variant is red on dev c15f28649
and on dev e791cbdde: the tab's chat is offered as a reference to itself
(open-webui/open-webui#34203, fix PR open-webui/open-webui#34204).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.cached_chat import this_turns_tool_results
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation, expect_reply, open_add_context, send
from utils.menus import late_menu_entry

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

REFERENCED_FACT = "The ferry to the island leaves at 07:40 from pier three."


@pytest.fixture
def account(make_user):
    return make_user()


@pytest.fixture
def referenced(account) -> tuple[str, str]:
    """The id and title of a stored chat to reference."""
    title = f"Ferry times {uuid.uuid4().hex[:6]}"
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "When does the ferry leave?"},
                {"role": "assistant", "content": REFERENCED_FACT},
            ],
        )
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert renamed.status_code == 200, renamed.text
    return chat_id, title


def _chat_picker(chat: ChatSurface) -> Locator:
    menu = open_add_context(chat)
    menu.get_by_role("menuitem", name="Reference Chats").click()
    return menu


def test_a_referenced_chat_reaches_the_model_with_the_message(
    chat_for, account, referenced, upstream
):
    referenced_id, referenced_title = referenced
    question = f"when should I be at the pier? {uuid.uuid4().hex[:6]}"
    answering = reply.answering(question)
    upstream.queue(
        reply.tool_call("view_chat", {"chat_id": referenced_id}, match=answering),
        reply.text("Be there by half past seven.", match=answering),
    )
    chat = chat_for(account)

    picker = _chat_picker(chat)
    picker.get_by_placeholder("Search Chats").fill(referenced_title)
    late_menu_entry(picker, referenced_title).click()
    send(chat, question)

    expect_reply(chat, "Be there by half past seven.")
    asked, answered = [body for body in upstream.chat_requests() if answering(body)]
    question_sent = str(reply.conversation(asked)[-1]["content"])
    assert f'type="chat" id="{referenced_id}"' in question_sent, question_sent
    opened = this_turns_tool_results(answered)
    assert REFERENCED_FACT in str(opened), f"the referenced chat never reached the model: {opened}"

    chat = chat.reload()
    expect_reply(chat, "Be there by half past seven.")
    expect(conversation(chat).get_by_text(referenced_title)).to_be_visible()


def test_the_open_chat_is_not_offered_as_a_reference_to_itself(chat_for, account, referenced):
    """In a tab: open-webui/open-webui#34203, fix PR open-webui/open-webui#34204."""
    chat_id, title = referenced
    chat = chat_for(account, chat_id=chat_id)
    expect(conversation(chat).get_by_text(REFERENCED_FACT)).to_be_visible()

    picker = _chat_picker(chat)
    picker.get_by_placeholder("Search Chats").fill(title)
    expect(picker.get_by_text("No chats found")).to_be_visible()
    expect(late_menu_entry(picker, title)).to_have_count(0)
