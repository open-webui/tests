"""Journey: a side chat answers from the chat so far and leaves the chat itself untouched.

Typing `/side <question>` (or `/btw`) in an open chat opens a temporary side conversation in a
tab beside it. The model gets the conversation so far as context plus the side question, and the
side exchange is kept out of the chat: it is not shown there, not saved with it and not stored as
a chat of its own, so a reload shows the chat as it was. Closing the side chat's tab discards it;
a new side chat, opened here with `/btw`, starts empty and its model never sees the closed one's
exchange.

Discriminates: passes on v0.13.0 and dev 5f1c6384e. In a v0.13.0 frontend build whose side chat
request leaves out the chat so far, the first test fails on what the model was sent; with closing
a side chat's tab keeping it, the second fails as the tab stays; with `/side` also sent on as a
message of the chat, the first fails as the question shows in the chat and, with that check left
out, as the stored chat holds it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ask
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Which pass links Lienz and Matrei?"
ANSWER = "The Felbertauern road links them through a tunnel."
SIDE_QUESTION = "By the way, how long is that tunnel? ({tag})"
SIDE_ANSWER = "The Felbertauern tunnel is 5.3 kilometres long."
SECOND_SIDE_QUESTION = "Unrelated, what is the capital of Carinthia? ({tag})"
SECOND_SIDE_ANSWER = "Klagenfurt is the capital of Carinthia."


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    """Allow Side Chat (and Temporary Chat, which it needs) on in the default permissions."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(temporary=True, side_chat=True)
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


def side_chat_tab(page: Page, number: int) -> Locator:
    return page.get_by_role("tab", name=f"Side chat {number}", exact=True)


def side_chat_panel(page: Page, number: int) -> Locator:
    return page.get_by_role("tabpanel", name=f"Side chat {number}", exact=True)


def ask_aside(
    page: Page, upstream, number: int, question: str, answer: str, command: str = "side"
) -> Locator:
    """Open side chat `number` with `/<command> <question>` and wait for its answer there."""
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, f"/{command} {question}")
    expect(side_chat_tab(page, number)).to_have_attribute("aria-selected", "true")
    panel = side_chat_panel(page, number)
    expect(panel).to_contain_text(answer, timeout=REPLY_TIMEOUT_MS)
    return panel


def conversation_sent_for(upstream, question: str) -> list[tuple[str, str]]:
    """The non-system turns the provider was sent with `question`, as (role, text) pairs."""
    sent = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    assert len(sent) == 1, f"expected one provider request for {question!r}, got {len(sent)}"
    turns = []
    for entry in sent[0]["messages"]:
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


def chat_ids(account) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1})
    assert listed.status_code == 200, listed.text
    return [chat["id"] for chat in listed.json()]


def test_a_side_chat_answers_from_the_chat_and_stays_out_of_it(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    page, chat_id = open_answered_chat(page_for, account, upstream)

    panel = ask_aside(page, upstream, 1, question, SIDE_ANSWER)

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
    assert chat_ids(account) == [chat_id], "the side chat was stored as a chat of its own"
    assert conversation_sent_for(upstream, question) == [
        ("user", PROMPT),
        ("assistant", ANSWER),
        ("user", question),
    ], "the side chat's model did not get the chat so far followed by the side question"

    page.reload()

    expect_reply(page, ANSWER)
    expect(conversation(page).get_by_text(PROMPT)).to_be_visible()
    # a reload reopens the tabs panel, which can start a new, empty side chat there
    expect(page.get_by_text(SIDE_ANSWER)).to_have_count(0)
    expect(page.get_by_text(question)).to_have_count(0)


def test_a_closed_side_chat_is_gone_and_the_next_one_starts_empty(
    page_for, make_user, upstream, side_chat_allowed
):
    first, second = tagged(SIDE_QUESTION), tagged(SECOND_SIDE_QUESTION)
    page, _ = open_answered_chat(page_for, make_user(), upstream)
    ask_aside(page, upstream, 1, first, SIDE_ANSWER)

    page.get_by_role("button", name="Close Side chat 1", exact=True).click()

    expect(side_chat_tab(page, 1)).to_have_count(0)
    expect(page.get_by_text(SIDE_ANSWER)).to_have_count(0)

    panel = ask_aside(page, upstream, 2, second, SECOND_SIDE_ANSWER, command="btw")

    expect(panel.get_by_text(SIDE_ANSWER)).to_have_count(0)
    assert conversation_sent_for(upstream, second) == [
        ("user", PROMPT),
        ("assistant", ANSWER),
        ("user", second),
    ], "the new side chat's model was sent more than the chat and its own question"
