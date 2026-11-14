"""Journey: a memory is added, edited and deleted under Settings > Personalization.

A fresh account adds a memory through the Actions menu and sees it listed, edits it and sees the
new text, and a chat then sends the model the edited memory as context. Deleting it empties the
list, and it stays gone after the settings are opened again. Switching Memory off in the same tab
keeps a stored memory out of the next chat, and stays off after a reload, while another account's
memory still reaches the model. Clear memory in the Actions menu empties the list for good and
keeps every memory out of the next chat, and the search box narrows the list. A memory the model
saves with its memory tool during a chat is listed in the tab. A memory the model reads back with
its list_memories tool is pinned in integration/memory/test_memory_tool_payload.py.

Discriminates: passes on dev ac00d40e3; in a backend copy, with
`POST /api/v1/memories/{id}/update` answering without storing the new text the edited memory
never shows, and with `DELETE /api/v1/memories/{id}` answering true without deleting the memory
is listed again once the settings are reopened; in a frontend copy, a chat sending the memory
feature whatever the account's Memory switch says turns the switch test red. On dev ebc6add67,
a backend copy whose `DELETE /api/v1/memories/delete/user` answers true without deleting, and
whose add_memory tool answers success without storing, turns the clear and saved-in-a-chat tests
red, and a frontend copy whose memory search ignores its text turns the search test red.

The test whose subject is the model's memory tool (saving a memory) runs in the main chat and in a
side chat; the saved memory is then looked up in the settings dialog of the page. The others walk
the Personalization settings dialog and the Memory switch, which are page things, and stay on the
main chat. The side chat variants are green on dev c15f28649 and on head e791cbdde.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST = "I keep bees on the roof."
EDITED = "I keep bees and two hens on the roof."
LATER = "I ride a bicycle to work every day."
MEMORY_FIELD = "Memory"


def _personalization(page: Page) -> Locator:
    page.goto("/?settings=personalization")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Personalization")).to_be_visible()
    return settings


def test_a_memory_is_added_edited_used_and_deleted(page_for, make_user, upstream):
    page = page_for(make_user())
    settings = _personalization(page)
    settings.get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Add Memory").click()
    adding = page.get_by_role("dialog").filter(has_text="Add Memory")
    adding.get_by_role("textbox", name=MEMORY_FIELD, exact=True).fill(FIRST)
    adding.get_by_role("button", name="Add", exact=True).click()
    expect(settings.get_by_text(FIRST)).to_be_visible()

    settings.get_by_role("button", name="Edit").click()
    editing = page.get_by_role("dialog").filter(has_text="Edit Memory")
    expect(editing.get_by_role("textbox", name=MEMORY_FIELD, exact=True)).to_have_value(FIRST)
    editing.get_by_role("textbox", name=MEMORY_FIELD, exact=True).fill(EDITED)
    editing.get_by_role("button", name="Update").click()
    expect(settings.get_by_text(EDITED)).to_be_visible()

    settings = _personalization(page)
    expect(settings.get_by_text(EDITED)).to_be_visible()
    expect(settings.get_by_text(FIRST)).to_have_count(0)

    question = "what do I keep on the roof?"
    upstream.queue(reply.text("Bees and two hens.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "Bees and two hens.")
    sent = json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))
    assert EDITED in sent, "the chat was not sent the edited memory"

    settings = _personalization(page)
    settings.get_by_role("button", name="Remove").click()
    page.get_by_role("dialog", name="Delete Memory?").get_by_role("button", name="Confirm").click()
    expect(settings.get_by_text(EDITED)).to_have_count(0)

    settings = _personalization(page)
    expect(settings.get_by_text("Memories accessible by LLMs will be shown here.")).to_be_visible()
    expect(settings.get_by_text(EDITED)).to_have_count(0)


def _memory_switch(page: Page) -> Locator:
    return _personalization(page).locator("#tab-personalization").get_by_role("switch").first


def _account_remembering(make_user, memory: str):
    account = make_user()
    with account.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": memory, "type": "user"})
    added.raise_for_status()
    return account


def _sent_for(page: Page, upstream, question: str) -> str:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "Noted.")
    return json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))


def test_switching_memory_off_keeps_memories_out_of_the_chat(page_for, make_user, upstream):
    page = page_for(_account_remembering(make_user, FIRST))
    switch = _memory_switch(page)
    expect(switch).to_have_attribute("aria-checked", "true")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")

    sent = _sent_for(page, upstream, "where are the bees?")
    assert FIRST not in sent, "a switched-off memory was sent to the model"

    page.reload()
    expect(_memory_switch(page)).to_have_attribute("aria-checked", "false")

    other = page_for(_account_remembering(make_user, FIRST))
    assert FIRST in _sent_for(other, upstream, "where are my bees?")


def _remembering(make_user, *memories: str):
    account = make_user()
    with account.client() as client:
        for memory in memories:
            added = client.post("/api/v1/memories/add", json={"content": memory, "type": "user"})
            added.raise_for_status()
    return account


def _stored_memories(account) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/memories/")
    listed.raise_for_status()
    return [memory["content"] for memory in listed.json()]


def test_clearing_memory_forgets_every_memory_of_the_account(page_for, make_user, upstream):
    account = _remembering(make_user, FIRST, LATER)
    other = _remembering(make_user, FIRST)
    page = page_for(account)
    settings = _personalization(page)
    expect(settings.get_by_text(LATER)).to_be_visible()

    settings.get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Clear memory").click()
    page.get_by_role("dialog", name="Clear Memory").get_by_role("button", name="Confirm").click()
    expect(page.get_by_text("Memory cleared successfully")).to_be_visible()

    settings = _personalization(page)
    expect(settings.get_by_text("Memories accessible by LLMs will be shown here.")).to_be_visible()
    expect(settings.get_by_text(FIRST)).to_have_count(0)
    sent = _sent_for(page, upstream, "what do you know about me?")
    assert FIRST not in sent and LATER not in sent, "a cleared memory was sent to the model"
    assert _stored_memories(other) == [FIRST], "clearing one account's memory touched another's"


def test_searching_memories_narrows_the_list(page_for, make_user):
    settings = _personalization(page_for(_remembering(make_user, FIRST, LATER)))
    expect(settings.get_by_text(FIRST)).to_be_visible()

    settings.get_by_placeholder("Search Memories").fill("bicycle")
    expect(settings.get_by_text(LATER)).to_be_visible()
    expect(settings.get_by_text(FIRST)).to_have_count(0)

    settings.get_by_role("button", name="Clear search").click()
    expect(settings.get_by_text(FIRST)).to_be_visible()


def test_a_memory_the_model_saves_during_a_chat_is_listed(chat_for, make_user, upstream):
    account = make_user()
    chat = chat_for(account)
    request = "please remember that I cycle to work"
    upstream.queue(
        reply.tool_call("add_memory", {"content": LATER}, match=reply.answering(request)),
        reply.text("I will remember that.", match=reply.answering(request)),
    )
    send(chat, request)
    expect_reply(chat, "I will remember that.")

    settings = _personalization(chat.page)
    expect(settings.get_by_text(LATER)).to_be_visible()
    assert _stored_memories(account) == [LATER]
