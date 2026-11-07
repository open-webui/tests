"""A temporary chat is answered but never saved, and switching it off gives a normal chat.

The toggle sits in the header beside the model selector on a new chat, also in the tabbed
layout a wide window gets (#33680). With it on, the reply arrives as usual while the chat stays
out of the sidebar history, out of the chat search and out of the database. With it off, the
same message makes a saved chat. A temporary chat that has an answer can still be kept through
the header's Save Chat button, and an account that turned on Temporary Chat by Default in its
Interface settings starts every new chat temporary. A temporary chat offers no Fork chat under
its reply and no Fork in its / menu (#32833), since a fork copies a stored chat and there is
none. A per chat terminal also needs a stored chat, so the Terminal menu of a temporary chat
leaves it out and turning Temporary Chat on drops one picked before (#33866).

Twin of integration/chat/test_temporary_chat.py.

Discriminates: passes on dev 176d31d1d, fails with `is_saved_chat_id` treating a `temporary:` id
as saved (it reaches the sidebar, the search and the database). The fork test passes with the
#32833 fix and fails on dev ca8f451cc (Fork chat and the / menu's Fork are offered and fail).
On dev f659c0d12, a frontend copy with 40b7b5201 (#33680) undone turns every test red that
turns Temporary Chat on (no toggle on a new chat). The per chat terminal test passes on dev
f659c0d12 and fails in a frontend copy with 0df605850 (#33866) undone (the terminal stays picked
and listed).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.search_dialog import no_matches, open_search_from_sidebar, search
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "Which lake is the deepest in Carinthia?"
ANSWER = "That is the Millstatter See."
PROMPT_TWO = "Which pass links Salzburg and Carinthia?"
ANSWER_TWO = "The Tauern pass does."
PROMPT_THREE = "How high is the Grossglockner?"
ANSWER_THREE = "It is 3798 metres high."
PER_CHAT_TERMINAL = "Per chat shell"
SHARED_TERMINAL = "Shared shell"


def temporary_toggle(page: Page):
    return page.get_by_role("button", name="Temporary Chat")


def temporary_label(page: Page):
    return page.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)


def stored_chats(account, text: str) -> list[dict]:
    """The chats the server lists for the account whose title or content matches `text`."""
    with account.client() as client:
        found = client.get("/api/v1/chats/search", params={"text": text})
        listed = client.get("/api/v1/chats/", params={"page": 1})
    found.raise_for_status()
    listed.raise_for_status()
    matching = [*found.json(), *[chat for chat in listed.json() if text in chat["title"]]]
    return list({chat["id"]: chat for chat in matching}.values())


def sidebar_entry(page: Page):
    """The sidebar link of the chat the page shows."""
    chat_id = page.url.rsplit("/", 1)[-1]
    return page.locator(f'a[href="/c/{chat_id}"]')


def test_a_temporary_chat_is_answered_and_left_out_of_history_search_and_storage(
    page_for, make_user, upstream
):
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()

    temporary_toggle(page).click()
    expect(temporary_label(page)).to_be_visible()
    send(page, PROMPT)

    expect_reply(page, ANSWER)
    expect(page.locator('a[href^="/c/"]')).to_have_count(0)
    dialog = open_search_from_sidebar(page)
    search(dialog, "Carinthia")
    expect(no_matches(dialog)).to_be_visible()
    assert "/c/" not in page.url
    assert stored_chats(account, "Carinthia") == []


def test_turning_temporary_chat_off_again_saves_the_chat(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()

    temporary_toggle(page).click()
    expect(temporary_label(page)).to_be_visible()
    temporary_toggle(page).click()
    expect(temporary_label(page)).to_have_count(0)
    send(page, PROMPT)

    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    expect(sidebar_entry(page)).to_be_visible()
    assert len(stored_chats(account, "Carinthia")) == 1


def test_saving_a_temporary_chat_keeps_the_conversation(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER_TWO, match=reply.answering(PROMPT_TWO)))
    expect(chat_input(page)).to_be_visible()

    temporary_toggle(page).click()
    send(page, PROMPT_TWO)
    expect_reply(page, ANSWER_TWO)
    assert stored_chats(account, "Salzburg") == []
    page.get_by_role("button", name="Save Chat").click()

    expect(page.get_by_text("Conversation saved successfully")).to_be_visible()
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    expect(temporary_label(page)).to_have_count(0)
    expect(sidebar_entry(page)).to_be_visible()
    chats = stored_chats(account, "Salzburg")
    assert chats
    with account.client() as client:
        chat = client.get(f"/api/v1/chats/{chats[0]['id']}").json()["chat"]
    contents = [message["content"] for message in chat["history"]["messages"].values()]
    assert contents == [PROMPT_TWO, ANSWER_TWO]


def test_temporary_chat_by_default_starts_new_chats_temporary(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(PROMPT)))
    expect(chat_input(page)).to_be_visible()
    expect(temporary_label(page)).to_have_count(0)

    open_settings(page)
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Interface").click()
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        settings.get_by_role("switch", name="Temporary Chat by Default").click()
    page.keyboard.press("Escape")
    page.get_by_role("link", name="New Chat").click()

    expect(temporary_label(page)).to_be_visible()
    send(page, PROMPT)
    expect_reply(page, ANSWER)
    assert stored_chats(account, "Carinthia") == []


@pytest.mark.regression
def test_a_temporary_chat_offers_no_fork(page_for, make_user, upstream):
    """#32833: a temporary chat is never stored, so forking it could only fail."""
    page = page_for(make_user())
    upstream.queue(reply.text(ANSWER_THREE, match=reply.answering(PROMPT_THREE)))
    expect(chat_input(page)).to_be_visible()

    temporary_toggle(page).click()
    expect(temporary_label(page)).to_be_visible()
    send(page, PROMPT_THREE)
    expect_reply(page, ANSWER_THREE)

    expect(conversation(page).get_by_role("button", name="Copy").last).to_be_visible()
    expect(conversation(page).get_by_role("button", name="Fork chat")).to_have_count(0)
    chat_input(page).click()
    page.keyboard.type("/")
    menu = page.get_by_role("tooltip")
    expect(menu.get_by_role("button", name=re.compile(r"^Status:"))).to_be_visible()
    expect(menu.get_by_role("button", name=re.compile(r"^Fork:"))).to_have_count(0)


@pytest.fixture
def per_chat_and_shared_terminals(admin, make_user, preserve):
    """A fresh account offered a per chat terminal and a shared one."""
    preserve(TERMINAL_SERVERS_CONFIG)
    account = make_user()
    grant = {"access_grants": [read_grant(account.id)]}
    with serving_terminal() as server:
        per_chat = server.connection(
            name=PER_CHAT_TERMINAL,
            server_type="orchestrator",
            config={**grant, "contexts": {"chat": {"context_id": "chat_id"}}},
        )
        shared = server.connection(name=SHARED_TERMINAL, config=grant)
        with admin.client() as client:
            configure_terminals(client, per_chat, shared)
        yield account


def terminal_menu(page: Page, current: str = "Terminal") -> Locator:
    page.get_by_role("main").get_by_role("button", name=current, exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitemradio", name=SHARED_TERMINAL)).to_be_visible()
    return menu


def test_a_temporary_chat_offers_no_per_chat_terminal_and_drops_one_picked_before(
    page_for, per_chat_and_shared_terminals
):
    """#33866: a per chat terminal needs a stored chat, so a temporary message failed with it."""
    page = page_for(per_chat_and_shared_terminals)
    expect(chat_input(page)).to_be_visible()
    terminal_menu(page).get_by_role("menuitemradio", name=PER_CHAT_TERMINAL).click()
    picked = page.get_by_role("main").get_by_role("button", name=PER_CHAT_TERMINAL, exact=True)
    expect(picked).to_be_visible()

    temporary_toggle(page).click()
    expect(temporary_label(page)).to_be_visible()

    expect(picked, "the per chat terminal stayed picked in a temporary chat").to_have_count(0)
    menu = terminal_menu(page)
    expect(menu.get_by_role("menuitemradio", name=PER_CHAT_TERMINAL)).to_have_count(0)
