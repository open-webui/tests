"""Journey: another chat referenced from the composer's Reference Chats reaches the model.

The composer's More menu lists the account's chats under Reference Chats, searched by title. The
chat picked is attached to the message and sent with it, so the model is given that conversation,
and the reference stays on the message after a reload. The open chat is not offered as a reference
to itself.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose chat references read no messages
the reaching test fails, and in a frontend build whose picker also lists the open chat the other
test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input, conversation, expect_reply, send

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


def _chat_picker(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name="Reference Chats").click()
    return menu


def test_a_referenced_chat_reaches_the_model_with_the_message(
    page_for, account, referenced, upstream
):
    _, referenced_title = referenced
    question = f"when should I be at the pier? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Be there by half past seven.", match=reply.answering(question)))
    page = page_for(account)

    picker = _chat_picker(page)
    picker.get_by_placeholder("Search Chats").fill(referenced_title)
    picker.get_by_role("button", name=referenced_title).click()
    send(page, question)

    expect_reply(page, "Be there by half past seven.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    sent = " ".join(str(message.get("content")) for message in request["messages"])
    assert REFERENCED_FACT in sent, f"the referenced chat never reached the model: {sent}"

    page.reload()
    expect_reply(page, "Be there by half past seven.")
    expect(conversation(page).get_by_text(referenced_title)).to_be_visible()


def test_the_open_chat_is_not_offered_as_a_reference_to_itself(page_for, account, referenced):
    chat_id, title = referenced
    page = page_for(account)
    page.goto(f"/c/{chat_id}")
    expect(conversation(page).get_by_text(REFERENCED_FACT)).to_be_visible()

    picker = _chat_picker(page)
    picker.get_by_placeholder("Search Chats").fill(title)
    expect(picker.get_by_text("No chats found")).to_be_visible()
    expect(picker.get_by_role("button", name=title)).to_have_count(0)
