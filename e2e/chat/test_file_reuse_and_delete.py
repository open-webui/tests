"""Journey: a file uploaded once is reused in a later chat, and a deleted one leaves its chats.

Attach Files in the chat input's menu lists the account's earlier uploads, narrowed by a search on
part of the name; picking one attaches it to a new chat without uploading it again, and the model
is given its text. A follow-up in the chat a file was sent in is given the file again. Once the
file is deleted, that chat still opens with the file named on its message, a follow-up is answered
without the file's text and the file is gone from Attach Files.

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
from utils.cached_chat import attach, attach_from_menu
from utils.chat_ui import chat_input, conversation, expect_reply, send

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
    """Send `question` and return everything the provider was sent for it."""
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(page, question)
    expect(conversation(page).get_by_text(question)).to_be_visible()
    expect_reply(page, "Noted.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return json.dumps(request["messages"])


def question(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


def open_attach_files(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name="Attach Files").click()
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
    expect(menu.get_by_role("button", name="tide-table.txt")).to_be_visible()

    menu.get_by_placeholder("Search Files").fill("ferry")

    expect(menu.get_by_role("button", name="tide-table.txt")).to_have_count(0)
    expect(menu.get_by_role("button", name="ferry-plan.txt")).to_be_visible()
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
