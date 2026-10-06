"""Journey: chats move into another account through every format the importer reads.

Export Chats in one account's Data Controls downloads a file that Import Chats in a fresh
account's Data Controls reads back: the chat shows in the sidebar with its conversation, and the
next message carries on from it, so the model is sent the imported turns. The same importer reads
a ChatGPT `conversations.json` (turned into a chat of its user and assistant turns, its folder
entries skipped) and the bare chat objects of old exports, and a file of exported chats dropped
onto the sidebar's Chats list is imported as well.

Discriminates: passes on dev ebc6add67; in a frontend copy whose importer sends the ChatGPT file
without converting it, sends a bare chat object without wrapping it and whose sidebar drop
ignores the dropped file, the format and drop tests fail, and in a backend copy whose
`POST /api/v1/chats/import` stores every chat with an empty history the carry-on test fails.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "how long do the tide pools stay open?"
ANSWER = "About two hours either side of low tide."


def _bare_chat(title: str) -> dict:
    """A chat object as the client stores it, the shape old exports listed without a wrapper."""
    question = {"id": "q1", "parentId": None, "childrenIds": ["a1"], "role": "user"}
    answer = {"id": "a1", "parentId": "q1", "childrenIds": [], "role": "assistant", "done": True}
    messages = [{**question, "content": QUESTION}, {**answer, "content": ANSWER}]
    return {
        "title": title,
        "models": [MOCK_MODEL_ID],
        "messages": messages,
        "history": {"currentId": "a1", "messages": {entry["id"]: entry for entry in messages}},
    }


def _store_chat(account, title: str) -> None:
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": _bare_chat(title)})
    assert created.status_code == 200, created.text


def _open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def _data_controls(page: Page) -> Locator:
    page.goto("/?settings=data_controls")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("Export Chats", exact=True)).to_be_visible()
    return settings


def _export(page: Page) -> bytes:
    settings = _data_controls(page)
    with page.expect_download() as downloaded:
        settings.get_by_role("button", name="Export", exact=True).click()
    with open(downloaded.value.path(), "rb") as exported:
        return exported.read()


def _import(page: Page, content: bytes, imported: int) -> None:
    settings = _data_controls(page)
    with page.expect_file_chooser() as chooser:
        settings.get_by_role("button", name="Import", exact=True).click()
    chooser.value.set_files(
        {"name": "chats.json", "mimeType": "application/json", "buffer": content}
    )
    expect(page.get_by_text(f"Successfully imported {imported} chats.")).to_be_visible()
    page.keyboard.press("Escape")


def _open_imported(page: Page, title: str) -> None:
    page.goto("/")
    _open_sidebar(page).get_by_role("button", name=title).click()
    expect(page).to_have_url(re.compile(r"/c/"))


def _titles(account) -> set[str]:
    with account.client() as client:
        listed = client.get("/api/v1/chats/")
    assert listed.status_code == 200, listed.text
    return {chat["title"] for chat in listed.json()}


def test_exported_chats_import_into_a_fresh_account_and_carry_on(page_for, make_user, upstream):
    owner, newcomer = make_user(), make_user()
    title = f"Tide pools {uuid.uuid4().hex[:6]}"
    _store_chat(owner, title)
    exported = _export(page_for(owner))

    page = page_for(newcomer)
    _import(page, exported, 1)
    _open_imported(page, title)
    expect(conversation(page).get_by_text(QUESTION)).to_be_visible()
    expect_reply(page, ANSWER)

    follow_up = "and at spring tide?"
    upstream.queue(reply.text("A little longer.", match=reply.answering(follow_up)))
    send(page, follow_up)
    expect_reply(page, "A little longer.")
    sent = next(filter(reply.answering(follow_up), upstream.chat_requests()))["messages"]
    assert [entry["content"] for entry in sent if entry["role"] != "system"] == [
        QUESTION,
        ANSWER,
        follow_up,
    ]

    page.reload()
    expect_reply(page, "A little longer.")
    assert _titles(newcomer) == {title}
    assert _titles(owner) == {title}


def _chatgpt_turn(turn_id: str, parent: str, child: str | None, role: str, text: str) -> dict:
    message = {
        "id": turn_id,
        "author": {"role": role},
        "content": {"content_type": "text", "parts": [text]},
        "create_time": 1_700_000_100,
        "metadata": {"model_slug": "gpt-4o"} if role == "assistant" else {},
    }
    children = [child] if child else []
    return {"id": turn_id, "message": message, "parent": parent, "children": children}


def _chatgpt_conversation(title: str) -> dict:
    root = {"id": "root", "message": None, "parent": None, "children": ["q1"]}
    return {
        "id": f"conversation-{uuid.uuid4().hex[:8]}",
        "title": title,
        "create_time": 1_700_000_000,
        "update_time": 1_700_000_200,
        "mapping": {
            "root": root,
            "q1": _chatgpt_turn("q1", "root", "a1", "user", QUESTION),
            "a1": _chatgpt_turn("a1", "q1", None, "assistant", ANSWER),
        },
    }


def test_a_chatgpt_export_imports_as_a_chat_of_its_turns(page_for, make_user):
    account = make_user()
    title = f"From ChatGPT {uuid.uuid4().hex[:6]}"
    project_entry = {"id": "project-1", "title": "A ChatGPT project"}
    export = [_chatgpt_conversation(title), project_entry]
    page = page_for(account)

    _import(page, json.dumps(export).encode(), 1)

    _open_imported(page, title)
    expect(conversation(page).get_by_text(QUESTION)).to_be_visible()
    expect_reply(page, ANSWER)
    assert _titles(account) == {title}


def test_bare_chat_objects_of_an_old_export_import(page_for, make_user):
    account = make_user()
    title = f"Old export {uuid.uuid4().hex[:6]}"
    page = page_for(account)

    _import(page, json.dumps([_bare_chat(title)]).encode(), 1)

    _open_imported(page, title)
    expect(conversation(page).get_by_text(QUESTION)).to_be_visible()
    expect_reply(page, ANSWER)


def test_exported_chats_dropped_on_the_sidebar_chat_list_are_imported(page_for, make_user):
    owner, newcomer = make_user(), make_user()
    title = f"Dropped {uuid.uuid4().hex[:6]}"
    _store_chat(owner, title)
    exported = _export(page_for(owner)).decode()
    page = page_for(newcomer)
    sidebar = _open_sidebar(page)
    chats_section = sidebar.get_by_role("button", name="Chats", exact=True)
    expect(chats_section).to_be_visible()

    dropped = page.evaluate_handle(
        """(text) => {
            const transfer = new DataTransfer();
            transfer.items.add(new File([text], 'chats.json', { type: 'application/json' }));
            return transfer;
        }""",
        exported,
    )
    chats_section.dispatch_event("drop", {"dataTransfer": dropped})

    expect(sidebar.get_by_role("button", name=title)).to_be_visible()
    assert _titles(newcomer) == {title}
