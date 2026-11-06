"""Journey: a side chat answers from the chat so far and leaves the chat itself untouched.

Typing `/side <question>` (or `/btw`) in an open chat opens a side conversation in a tab beside
it. The model gets the conversation so far as context plus the side question, and the side
exchange is kept out of the chat: it is not shown there and not saved with it, so a reload shows
the chat as it was. Since 3b1a95383 a side chat is saved as a chat of its own, titled with its
question, which also names its tab. Closing the side chat's tab takes it off the page; a new side
chat, opened here with `/btw`, starts empty and its model never sees the closed one's exchange.

Discriminates: passes on v0.13.0 and dev 5f1c6384e. In a v0.13.0 frontend build whose side chat
request leaves out the chat so far, the first test fails on what the model was sent; with closing
a side chat's tab keeping it, the second fails as the tab stays; with `/side` also sent on as a
message of the chat, the first fails as the question shows in the chat and, with that check left
out, as the stored chat holds it. Retargeted for the saved side chats of 3b1a95383: passes on dev
c165ee14f, and in its frontend build whose side chat copies none of the chat so far both
tests fail on what the model was sent.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ChatTurn, ask, wait_for_reply
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    send,
    side_chat_panel,
    side_chat_tab,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Which pass links Lienz and Matrei?"
ANSWER = "The Felbertauern road links them through a tunnel."
SIDE_QUESTION = "By the way, how long is that tunnel? ({tag})"
SIDE_ANSWER = "The Felbertauern tunnel is 5.3 kilometres long."
SECOND_SIDE_QUESTION = "Unrelated, what is the capital of Carinthia? ({tag})"
SECOND_SIDE_ANSWER = "Klagenfurt is the capital of Carinthia."


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    """Allow Side Chat on in the default permissions."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def tagged(question: str) -> str:
    """`question` made unique to this test, so its provider request is told apart from others."""
    return question.format(tag=uuid.uuid4().hex[:8])


def open_answered_chat(page_for, account, upstream) -> tuple[Page, str]:
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with account.client() as client:
        turn, _ = ask(client, PROMPT)
    page = page_for(account, path=f"/c/{turn.chat_id}")
    expect_reply(page, ANSWER)
    return page, turn.chat_id


def ask_aside(page: Page, upstream, question: str, answer: str, command: str = "side") -> Locator:
    """Open a side chat with `/<command> <question>` and wait for its answer there."""
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, f"/{command} {question}")
    expect(side_chat_tab(page, question)).to_have_attribute("aria-selected", "true")
    panel = side_chat_panel(page, question)
    expect(panel).to_contain_text(answer, timeout=REPLY_TIMEOUT_MS)
    return panel


def conversation_sent_for(upstream, question: str) -> list[tuple[str, str]]:
    """The non-system turns the provider was sent with `question`, as (role, text) pairs."""
    sent = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    assert len(sent) == 1, f"expected one provider request for {question!r}, got {len(sent)}"
    turns = []
    for entry in reply.conversation(sent[0]):
        if entry["role"] == "system":
            continue
        content = entry["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        turns.append((entry["role"], content))
    return turns


def stored_chat(account, chat_id: str) -> dict:
    with account.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    assert found.status_code == 200, found.text
    return found.json()["chat"]


def stored_reply_to(account, chat_id: str, question: str) -> dict:
    """The stored reply to `question` in chat `chat_id`, once the server has marked it done."""
    messages = stored_chat(account, chat_id)["history"]["messages"]
    asked = [message for message in messages.values() if message.get("content") == question]
    assert len(asked) == 1 and asked[0]["childrenIds"], f"{question!r} was not saved: {messages}"
    turn = ChatTurn(chat_id, asked[0]["id"], asked[0]["childrenIds"][0])
    with account.client() as client:
        return wait_for_reply(client, turn)


def listed_chats(account) -> dict[str, str]:
    """The account's chats in its chat list, as id to title."""
    with account.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1})
    assert listed.status_code == 200, listed.text
    return {chat["id"]: chat["title"] for chat in listed.json()}


def test_a_side_chat_answers_from_the_chat_and_stays_out_of_it(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    page, chat_id = open_answered_chat(page_for, account, upstream)

    panel = ask_aside(page, upstream, question, SIDE_ANSWER)

    expect(panel).to_contain_text(question)
    expect(chat_input(page)).to_have_text("")
    # both conversations carry the same label; only the chat's own message list has this id
    main_messages = page.locator("#messages-container")
    expect(main_messages).to_contain_text(ANSWER)
    expect(main_messages.get_by_text(SIDE_ANSWER)).to_have_count(0)
    expect(main_messages.get_by_text(question)).to_have_count(0)
    stored = json.dumps(stored_chat(account, chat_id))
    assert question not in stored and SIDE_ANSWER not in stored, (
        f"the side exchange was saved into the chat: {stored}"
    )
    assert ANSWER in stored
    chats = listed_chats(account)
    side_chat_ids = [listed for listed, title in chats.items() if title == question]
    assert chat_id in chats and len(side_chat_ids) == 1 and len(chats) == 2, (
        f"expected the chat and one side chat titled {question!r} in the chat list, got {chats}"
    )
    side_reply = stored_reply_to(account, side_chat_ids[0], question)
    assert SIDE_ANSWER in json.dumps(side_reply), (
        f"the side chat saved another answer: {side_reply}"
    )
    assert conversation_sent_for(upstream, question) == [
        ("user", PROMPT),
        ("assistant", ANSWER),
        ("user", question),
    ], "the side chat's model did not get the chat so far followed by the side question"

    page.reload()

    expect_reply(page, ANSWER)
    expect(conversation(page).get_by_text(PROMPT)).to_be_visible()
    # the side chat's title also shows in the sidebar's chat list
    expect(main_messages.get_by_text(SIDE_ANSWER)).to_have_count(0)
    expect(main_messages.get_by_text(question)).to_have_count(0)


def test_a_closed_side_chat_is_gone_and_the_next_one_starts_empty(
    page_for, make_user, upstream, side_chat_allowed
):
    first, second = tagged(SIDE_QUESTION), tagged(SECOND_SIDE_QUESTION)
    page, _ = open_answered_chat(page_for, make_user(), upstream)
    ask_aside(page, upstream, first, SIDE_ANSWER)

    page.get_by_role("button", name=f"Close {first}", exact=True).click()

    expect(side_chat_tab(page, first)).to_have_count(0)
    expect(page.get_by_text(SIDE_ANSWER)).to_have_count(0)

    panel = ask_aside(page, upstream, second, SECOND_SIDE_ANSWER, command="btw")

    expect(panel.get_by_text(SIDE_ANSWER)).to_have_count(0)
    assert conversation_sent_for(upstream, second) == [
        ("user", PROMPT),
        ("assistant", ANSWER),
        ("user", second),
    ], "the new side chat's model was sent more than the chat and its own question"
