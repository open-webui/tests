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
A side chat opened next to a chat on a model with default tools sends its first message with those
tools on (#33788, 18f199b80); it passes on dev f659c0d12 and fails in a build of it whose side
chat leaves the model's defaults off.

The saved side chat is a chat like any other: it is listed in the sidebar under its question, its
tab is one of the chat's saved tabs and is back after a reload, and opening it from the sidebar
carries on from the chat so far and its own exchange without touching the chat it came from.
Deleting it from the sidebar takes it off the list and leaves that chat as it was. Any saved chat
dragged from the sidebar onto the right side of the open chat shows "Open in side panel" and
opens in a tab beside it, without copying it; without the Side Chat permission it does not.

Discriminates (the saved side chat tests): pass on the dev f659c0d12 build. On that build patched
to save a side chat as "New Chat" every test that asks aside fails, as neither a tab nor a row
carries the question; with saved side chat tabs left out of the saved view, the first test and the
drop test fail after the reload; with the Side Chat check dropped from the drop target, the test
without the permission fails as the drop hint shows.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ChatTurn, ask, wait_for_reply
from harness.chat_history import seed_chat
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    send,
    side_chat_panel,
    side_chat_tab,
)
from utils.sidebar_chats import chat_menu, chat_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Which pass links Lienz and Matrei?"
ANSWER = "The Felbertauern road links them through a tunnel."
SIDE_QUESTION = "By the way, how long is that tunnel? ({tag})"
SIDE_ANSWER = "The Felbertauern tunnel is 5.3 kilometres long."
SECOND_SIDE_QUESTION = "Unrelated, what is the capital of Carinthia? ({tag})"
SECOND_SIDE_ANSWER = "Klagenfurt is the capital of Carinthia."
OTHER_TITLE = "Packing for the Grossglockner"
OTHER_CHAT = [
    {"role": "user", "content": "What should I pack for the Grossglockner road?"},
    {"role": "assistant", "content": "A warm jacket, the road tops out at 2,504 metres."},
]


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    """Allow Side Chat on in the default permissions."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def side_chat_denied(admin, preserve) -> None:
    """Switch Side Chat off in the default permissions."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=False)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def seed_titled_chat(account, messages: list[dict], title: str) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(client, messages)
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert renamed.status_code == 200, renamed.text
    return chat_id


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


def main_chat(page: Page) -> Locator:
    """The chat's own messages; a side chat's tab carries the same label but not this id."""
    return page.locator("#messages-container")


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def drag_onto_the_chat(page: Page, row: Locator) -> Locator:
    """Drag a sidebar chat over the right side of the open conversation; returns the drop hint.

    The mouse is still down when this returns, so the caller sees the hint and then drops."""
    target = main_chat(page).bounding_box()
    start = row.bounding_box()
    page.mouse.move(start["x"] + start["width"] / 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        target["x"] + target["width"] * 0.9, target["y"] + target["height"] / 2, steps=20
    )
    return page.get_by_role("status").filter(has_text="Open in side panel")


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
    main_messages = main_chat(page)
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

    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    expect(main_chat(page).get_by_text(PROMPT)).to_be_visible()
    expect(main_messages.get_by_text(SIDE_ANSWER)).to_have_count(0)
    expect(main_messages.get_by_text(question)).to_have_count(0)
    # the saved side chat's tab is back after the reload, with its exchange
    expect(side_chat_panel(page, question)).to_contain_text(SIDE_ANSWER)
    expect(chat_row(open_sidebar(page), question)).to_be_visible()


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


TIDE_TOOL = '''class Tools:
    def tide_times(self, harbour: str) -> str:
        """Tell the tide times of a harbour."""
        return "high water at noon"
'''


@pytest.fixture
def model_with_a_tool(admin):
    """A preset everyone may use whose default tools hold a tide tool; yields its id."""
    model_id = f"tides-{uuid.uuid4().hex[:8]}"
    with python_tool(admin, TIDE_TOOL, name=f"Tide desk {model_id}") as tool_id:
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
        try:
            yield model_id
        finally:
            with admin.client() as client:
                client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_side_chat_starts_with_the_models_default_tools(
    page_for, make_user, upstream, side_chat_allowed, model_with_a_tool
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    with account.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        turn, _ = ask(client, PROMPT, model=model_with_a_tool, tool_ids=[])
    page = page_for(account, path=f"/c/{turn.chat_id}")
    expect_reply(page, ANSWER)

    ask_aside(page, upstream, question, SIDE_ANSWER)

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    offered = {tool["function"]["name"] for tool in sent.get("tools") or []}
    assert "tide_times" in offered, (
        f"the side chat's first message went out without the model's default tool: {offered}"
    )


def test_a_saved_side_chat_reopens_from_the_sidebar_and_carries_on(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    page, chat_id = open_answered_chat(page_for, account, upstream)
    ask_aside(page, upstream, question, SIDE_ANSWER)
    side_chat_ids = [listed for listed, title in listed_chats(account).items() if title == question]
    assert len(side_chat_ids) == 1, f"no side chat titled {question!r}: {listed_chats(account)}"

    chat_row(open_sidebar(page), question).click()

    expect(page).to_have_url(re.compile(rf"/c/{side_chat_ids[0]}$"))
    expect_reply(page, SIDE_ANSWER)
    expect(conversation(page).get_by_text(question)).to_be_visible()
    follow_up = tagged("And how high is the pass road? ({tag})")
    upstream.queue(reply.text("It climbs to 1,650 metres.", match=reply.answering(follow_up)))
    send(page, follow_up)
    expect_reply(page, "It climbs to 1,650 metres.")
    assert conversation_sent_for(upstream, follow_up) == [
        ("user", PROMPT),
        ("assistant", ANSWER),
        ("user", question),
        ("assistant", SIDE_ANSWER),
        ("user", follow_up),
    ], "the reopened side chat did not carry on from the chat so far and its own exchange"
    parent = json.dumps(stored_chat(account, chat_id))
    assert follow_up not in parent and question not in parent, (
        f"carrying on in the side chat wrote into the chat it came from: {parent}"
    )


def test_a_deleted_side_chat_is_gone_and_the_chat_it_came_from_stays(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    page, chat_id = open_answered_chat(page_for, account, upstream)
    ask_aside(page, upstream, question, SIDE_ANSWER)
    sidebar = open_sidebar(page)
    expect(chat_row(sidebar, question)).to_be_visible()

    chat_menu(sidebar, question).get_by_role("menuitem", name="Delete").click()
    confirm = page.get_by_role("dialog", name="Delete chat?").get_by_role("button", name="Confirm")
    with page.expect_response(lambda response: response.request.method == "DELETE") as deleted:
        confirm.click()

    assert deleted.value.ok, f"deleting the side chat answered {deleted.value.status}"
    expect(chat_row(sidebar, question)).to_have_count(0)
    chats = listed_chats(account)
    assert list(chats) == [chat_id], f"deleting the side chat left {chats}"
    page.reload()
    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    # the parent's saved tabs may still name the side chat, but its exchange is gone
    expect(page.get_by_text(SIDE_ANSWER)).to_have_count(0)
    # the sidebar stays open across the reload
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    expect(chat_row(sidebar, chats[chat_id])).to_be_visible()
    expect(chat_row(sidebar, question)).to_have_count(0)
    assert ANSWER in json.dumps(stored_chat(account, chat_id))


def test_a_chat_dropped_from_the_sidebar_opens_beside_the_chat(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    other_id = seed_titled_chat(account, OTHER_CHAT, OTHER_TITLE)
    page, chat_id = open_answered_chat(page_for, account, upstream)
    row = chat_row(open_sidebar(page), OTHER_TITLE)
    expect(row).to_be_visible()

    hint = drag_onto_the_chat(page, row)
    expect(hint).to_be_visible()
    page.mouse.up()

    expect(hint).to_have_count(0)
    panel = page.get_by_role("tabpanel", name=OTHER_TITLE, exact=True)
    expect(panel).to_contain_text(OTHER_CHAT[1]["content"])
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}$"))
    expect(main_chat(page)).to_contain_text(ANSWER)
    assert set(listed_chats(account)) == {chat_id, other_id}, (
        f"dropping a chat created or lost chats: {listed_chats(account)}"
    )
    # a saved chat's tab is part of the chat's saved tabs
    page.reload()
    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    expect(page.get_by_role("tabpanel", name=OTHER_TITLE, exact=True)).to_contain_text(
        OTHER_CHAT[1]["content"]
    )


def test_without_side_chat_a_chat_dropped_from_the_sidebar_opens_nothing(
    page_for, make_user, upstream, side_chat_denied
):
    account = make_user()
    seed_titled_chat(account, OTHER_CHAT, OTHER_TITLE)
    page, chat_id = open_answered_chat(page_for, account, upstream)
    row = chat_row(open_sidebar(page), OTHER_TITLE)
    expect(row).to_be_visible()

    hint = drag_onto_the_chat(page, row)
    # the same drag shows the hint for an account with Side Chat on
    expect(hint).to_have_count(0)
    page.mouse.up()

    expect(page.get_by_role("tab", name=OTHER_TITLE, exact=True)).to_have_count(0)
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}$"))
    expect_reply(page, ANSWER)
