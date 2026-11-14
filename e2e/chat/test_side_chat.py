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
opens in a tab beside it, without copying it; since d7b3d08b4 that needs no Side Chat permission,
as the tab copies nothing from the chat it opens beside.

Discriminates (the saved side chat tests): pass on the dev f659c0d12 build. On that build patched
to save a side chat as "New Chat" every test that asks aside fails, as neither a tab nor a row
carries the question; with saved side chat tabs left out of the saved view, the first test and the
drop test fail after the reload.

A side chat stays temporary, answered but kept out of the chat list, when the account turns on
Temporary Chat in it with `/temporary` before its first message, or when the admin enforces
temporary chats. Discriminates: both pass on the e8c0ad85e build; in a build of it whose side
chat starts saved whatever is enforced the enforced test fails, and in one that ignores the
toggle the toggle test fails. Leaving the chat and coming back keeps the side tab temporary;
that test is red on dev 782f4b705 (#34215), where the tab comes back as a new saved chat and its
next message is saved, and passes on the fix build.

A saved side tab brought back by a reload shows only the side exchange, never the chat's
messages its saved copy starts with (open-webui/open-webui#34002, fixed in b01fe7fae). While a
reply is written, the tab of the chat that answers shows a spinner read out as "Generating...",
the chat's own tab and a side tab alike, and the other tab does not (6bd86a161). Discriminates:
both pass on dev 485103cde; in a frontend copy of it that drops the saved branch point when a
side chat loads, the first test fails after the reload, and in one without the tab spinner the
spinner test fails.

Since dev da850248d the address follows the selected tab, so the tests select the chat's own tab
before a reload and before reading its address.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ChatTurn, ask, wait_for_reply
from harness.chat_history import seed_chat
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import drop_chat_beside
from utils.chat_surface import main_chat as main_chat_view
from utils.chat_surface import open_side_chat as open_empty_side_chat
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
# a tab spinner reads out this before the tab's title
GENERATING = "Generating..."
TOLL = ["The toll ", "for ", "cars ", "is ", "12 euros."]
SIDE_TOLL = ["It ", "is ", "5.3 ", "kilometres ", "long."]
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


def reload_the_chat(page: Page) -> None:
    """Reload with the chat's own tab selected: since dev da850248d the address follows the
    selected tab, so a reload with a side tab selected opens the side chat instead."""
    main_chat_view(page).show()
    page.reload()


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
    """Named after its first message: open-webui/open-webui#34366.

    Fix branch Classic298:fix/side-chat-saved-as-new-chat.
    """
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

    reload_the_chat(page)

    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    expect(main_chat(page).get_by_text(PROMPT)).to_be_visible()
    expect(main_messages.get_by_text(SIDE_ANSWER)).to_have_count(0)
    expect(main_messages.get_by_text(question)).to_have_count(0)
    # the saved side chat's tab is back after the reload, with its exchange and only that (#34002)
    side_panel = side_chat_panel(page, question)
    expect(side_panel).to_contain_text(SIDE_ANSWER)
    expect(
        side_panel.get_by_text(ANSWER), "the reloaded side tab lists the chat's messages (#34002)"
    ).to_have_count(0)
    expect(side_panel.get_by_text(PROMPT)).to_have_count(0)
    expect(chat_row(open_sidebar(page), question)).to_be_visible()


def test_the_chat_tab_and_a_side_tab_show_a_spinner_while_they_answer(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    page, _ = open_answered_chat(page_for, account, upstream)
    chat_tab = main_chat_view(page).tab
    expect(chat_tab).to_be_visible()
    expect(chat_tab).not_to_contain_text(GENERATING)

    follow_up = tagged("And what is the toll for the tunnel? ({tag})")
    release = threading.Event()
    upstream.queue(reply.text(TOLL, hold=release, hold_after=3, match=reply.answering(follow_up)))
    try:
        send(page, follow_up)
        expect(main_chat(page)).to_contain_text("The toll for", timeout=REPLY_TIMEOUT_MS)
        expect(
            chat_tab, "the chat tab shows no spinner while its reply is written"
        ).to_contain_text(GENERATING)
    finally:
        release.set()
    expect_reply(page, TOLL[-1])
    expect(chat_tab, "the chat tab still spins after its reply ended").not_to_contain_text(
        GENERATING
    )

    question = tagged(SIDE_QUESTION)
    side_release = threading.Event()
    upstream.queue(
        reply.text(SIDE_TOLL, hold=side_release, hold_after=3, match=reply.answering(question))
    )
    try:
        send(page, f"/side {question}")
        side_tab = side_chat_tab(page, question)
        expect(side_chat_panel(page, question)).to_contain_text("It is", timeout=REPLY_TIMEOUT_MS)
        expect(
            side_tab, "the side tab shows no spinner while its reply is written"
        ).to_contain_text(GENERATING)
        expect(chat_tab, "the chat tab spins for the side chat's reply").not_to_contain_text(
            GENERATING
        )
    finally:
        side_release.set()
    expect(side_chat_panel(page, question)).to_contain_text(SIDE_TOLL[-1], timeout=REPLY_TIMEOUT_MS)
    expect(side_tab, "the side tab still spins after its reply ended").not_to_contain_text(
        GENERATING
    )


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
    reload_the_chat(page)
    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    # the parent's saved tabs may still name the side chat, but its exchange is gone
    expect(page.get_by_text(SIDE_ANSWER)).to_have_count(0)
    # the sidebar stays open across the reload
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    expect(chat_row(sidebar, chats[chat_id])).to_be_visible()
    expect(chat_row(sidebar, question)).to_have_count(0)
    assert ANSWER in json.dumps(stored_chat(account, chat_id))


def test_a_chat_dropped_from_the_sidebar_opens_beside_the_chat(
    page_for, make_user, upstream, side_chat_denied
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
    main_chat_view(page).show()
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}$"))
    expect(main_chat(page)).to_contain_text(ANSWER)
    assert set(listed_chats(account)) == {chat_id, other_id}, (
        f"dropping a chat created or lost chats: {listed_chats(account)}"
    )
    # a saved chat's tab is part of the chat's saved tabs
    reload_the_chat(page)
    expect(main_chat(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    expect(page.get_by_role("tabpanel", name=OTHER_TITLE, exact=True)).to_contain_text(
        OTHER_CHAT[1]["content"]
    )


def enforce_temporary_chats(admin) -> None:
    """Side Chat allowed and temporary chats enforced in the default permissions."""
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True, temporary=True, temporary_enforced=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def side_panel_showing(page: Page, text: str) -> Locator:
    """The side chat's panel once it shows `text`; a temporary one keeps its numbered label."""
    panel = page.get_by_role("tabpanel").filter(has_text=text)
    expect(panel).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    return panel


def test_with_temporary_chats_enforced_a_side_chat_stays_out_of_the_chat_list(
    page_for, admin, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    _, chat_id = open_answered_chat(page_for, account, upstream)
    # an account under enforced temporary chats cannot save the chat the side chat starts from
    enforce_temporary_chats(admin)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, ANSWER)
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))

    send(page, f"/side {question}")

    side_panel_showing(page, SIDE_ANSWER)
    expect(main_chat(page)).not_to_contain_text(SIDE_ANSWER)
    assert list(listed_chats(account)) == [chat_id], (
        f"a side chat under enforced temporary chats was saved: {listed_chats(account)}"
    )


def test_temporary_chat_turned_on_in_a_side_chat_keeps_it_out_of_the_chat_list(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    question = tagged(SIDE_QUESTION)
    page, chat_id = open_answered_chat(page_for, account, upstream)
    side = open_empty_side_chat(page)
    side.input.click()
    page.keyboard.type("/temporary")
    page.get_by_role("button", name=re.compile(r"^Temporary")).click()
    expect(side.input).to_have_text("")
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))

    side.send(question)

    side.expect_reply(SIDE_ANSWER)
    assert list(listed_chats(account)) == [chat_id], (
        f"a side chat switched to temporary was saved: {listed_chats(account)}"
    )


def test_a_temporary_side_chat_stays_temporary_after_leaving_the_chat_and_coming_back(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red on dev 782f4b705 (open-webui/open-webui#34215): the side tab comes back as a saved
    chat; the fix is on Classic298:fix/temporary-side-chat-saved-after-coming-back."""
    account = make_user()
    parent_title = f"Lienz to Matrei {uuid.uuid4().hex[:8]}"
    chat_id = seed_titled_chat(
        account,
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": ANSWER}],
        parent_title,
    )
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, ANSWER)
    side = open_empty_side_chat(page)
    side.input.click()
    page.keyboard.type("/temporary")
    page.get_by_role("button", name=re.compile(r"^Temporary")).click()
    expect(side.input).to_have_text("")
    question = tagged(SIDE_QUESTION)
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))
    side.send(question)
    side.expect_reply(SIDE_ANSWER)

    main = main_chat_view(page)
    main.show()
    sidebar = open_sidebar(page)
    sidebar.get_by_role("link", name="New Chat").last.click()
    expect(main.replies).to_have_count(0)
    sidebar.get_by_role("button", name=re.compile(rf"^{re.escape(parent_title)}(?: |$)")).click()
    expect(main.last_reply).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    side.show()

    expect(side.panel).to_contain_text("Messages disappear when you close this tab.")
    second_question = tagged(SECOND_SIDE_QUESTION)
    upstream.queue(reply.text(SECOND_SIDE_ANSWER, match=reply.answering(second_question)))
    side.send(second_question)
    side.expect_reply(SECOND_SIDE_ANSWER)
    assert list(listed_chats(account)) == [chat_id], (
        f"the temporary side chat was saved after coming back to it: {listed_chats(account)}"
    )


def test_an_empty_side_chat_says_whether_its_messages_are_saved(
    page_for, make_user, upstream, side_chat_allowed
):
    """Red on dev since the side chat swap of open-webui/open-webui#34064 (b428dbaf2), which
    dropped the note on purpose; whether a side chat keeps it is the maintainers' call."""
    page, _ = open_answered_chat(page_for, make_user(), upstream)
    side = open_empty_side_chat(page)
    expect(side.panel).to_contain_text("Messages are saved to your chats.")
    side.input.click()
    page.keyboard.type("/temporary")
    page.get_by_role("button", name=re.compile(r"^Temporary")).click()

    expect(side.panel).to_contain_text("Messages disappear when you close this tab.")


def test_a_saved_chat_in_a_tab_beside_a_temporary_chat_can_be_rated(
    page_for, make_user, side_chat_allowed
):
    """Red on dev c15f28649 (no rating in a tab, open-webui/open-webui#34014) and still on dev
    e791cbdde, after the side chat swap of open-webui/open-webui#34064 (issue draft pending)."""
    account = make_user()
    seed_titled_chat(account, OTHER_CHAT, OTHER_TITLE)
    page = page_for(account, path="/?temporary-chat=true")
    expect(chat_input(page)).to_be_visible()

    saved = drop_chat_beside(page, OTHER_TITLE)

    saved.last_reply.hover()
    expect(
        saved.button("Good Response"), "the saved chat's answer offers no rating"
    ).to_be_visible()
