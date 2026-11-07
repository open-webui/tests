"""Journey: a file uploaded once is reused in a later chat, and a deleted one leaves its chats.

Attach Files in the chat input's menu lists the account's earlier uploads, narrowed by a search on
part of the name; picking one attaches it to a new chat without uploading it again, and the model
can read its text. A follow-up in the chat a file was sent in can read the file again. Once the
file is deleted, that chat still opens with the file named on its message, a follow-up is answered
without the file's text and the file is gone from Attach Files. Since bd6fc15be File Context is
off with native function calling, so the scripted model reads the file through the chat file
tools.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose Attach Files list
never searches and whose picked file is never attached, the reuse and search tests fail; in a
backend copy whose file delete keeps the file's row, the deleted file test fails (the follow-up is
still given its text and it is still offered).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from utils.cached_chat import attach, attach_from_menu, this_turns_tool_results
from utils.chat_ui import conversation, expect_reply, open_add_context, send
from utils.menus import late_menu_entry

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MAP_TEXT = "The harbour map shows the pilot station on pier three."


def upload(account: Actor, name: str, text: str) -> str:
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/", files={"file": (name, text.encode(), "text/plain")}
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def file_ids(account: Actor) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/files/", params={"content": False})
    assert listed.status_code == 200, listed.text
    return [entry["id"] for entry in listed.json()["items"]]


def ask(page: Page, upstream, question: str) -> str:
    """Send `question`; what the model found searching the chat's files for the harbour map."""
    answering = reply.answering(question)
    upstream.queue(
        reply.tool_call("grep_chat_files", {"pattern": "harbour map"}, match=answering),
        reply.text("Noted.", match=answering),
    )
    send(page, question)
    expect(conversation(page).get_by_text(question)).to_be_visible()
    expect_reply(page, "Noted.")
    *_, answered = [body for body in upstream.chat_requests() if answering(body)]
    return json.dumps(this_turns_tool_results(answered))


def question(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


def open_attach_files(page: Page) -> Locator:
    menu = open_add_context(page)
    menu.get_by_role("menuitem", name="Attach Files").click()
    expect(menu.get_by_placeholder("Search Files")).to_be_visible()
    return menu


def test_an_earlier_file_picked_in_attach_files_reaches_the_model_in_a_new_chat(
    page_for, make_user, upstream
):
    account = make_user()
    upload(account, "harbour-map.txt", MAP_TEXT)
    uploads_before = file_ids(account)
    page = page_for(account)

    attach_from_menu(page, "Attach Files", "harbour-map.txt")
    sent = ask(page, upstream, question("where is the pilot station?"))

    assert "pilot station on pier three" in sent
    assert file_ids(account) == uploads_before


def test_attach_files_finds_a_file_by_part_of_its_name(page_for, make_user):
    account = make_user()
    upload(account, "tide-table.txt", "high at noon")
    upload(account, "ferry-plan.txt", "north pier at seven")
    page = page_for(account)
    menu = open_attach_files(page)
    expect(late_menu_entry(menu, "tide-table.txt")).to_be_visible()

    menu.get_by_placeholder("Search Files").fill("ferry")

    expect(late_menu_entry(menu, "tide-table.txt")).to_have_count(0)
    expect(late_menu_entry(menu, "ferry-plan.txt")).to_be_visible()
    menu.get_by_placeholder("Search Files").fill("lighthouse")
    expect(menu.get_by_text("No files found")).to_be_visible()


def test_a_deleted_file_leaves_its_chat_answering_without_it(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    attach(page, "harbour-map.txt", MAP_TEXT)
    expect(
        page.locator("form").get_by_role("button").filter(has_text="harbour-map.txt")
    ).to_be_visible()
    assert "pilot station on pier three" in ask(page, upstream, question("where is the pilot?"))
    page.wait_for_url(re.compile(r"/c/"))
    assert "pilot station on pier three" in ask(page, upstream, question("and the tug?"))
    chat_url = page.url

    [file_id] = file_ids(account)
    with account.client() as client:
        assert client.delete(f"/api/v1/files/{file_id}").status_code == 200

    page.goto(chat_url)
    expect(
        conversation(page).get_by_role("button").filter(has_text="harbour-map.txt")
    ).to_be_visible()
    after_delete = ask(page, upstream, question("and the ferry?"))
    assert "pilot station on pier three" not in after_delete
    menu = open_attach_files(page)
    expect(menu.get_by_text("No files found")).to_be_visible()
