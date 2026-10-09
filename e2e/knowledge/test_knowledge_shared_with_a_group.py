"""Journey: a knowledge base shared with a group, used by its members in chat and in the workspace.

An owner shares a base with a group. With read access a member attaches it with `#` in a chat:
the model is sent the file's text and the reply cites the file, while the base's page offers the
member no way to add, rename or remove a file. With write access a member uploads a file into the
base from its page, and the owner's next chat that attaches the base cites the member's file. An
account outside the group is not offered the base in `#`.

Discriminates: passes on dev 0f5a58f5f. In a backend copy whose knowledge search leaves out bases
shared through a group, the read test fails (the member is not offered the base); in one whose
upload links a file into a base only for its owner or an admin, the write test fails (the file
never reaches the base). A frontend build that shows the file menu on every row fails the read
test.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant, make_group
from harness.actors import Actor
from harness.knowledge_bases import add_text_file
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KNOWLEDGE_USER = {"workspace": {"knowledge": True}}


@pytest.fixture
def curator(make_user) -> Iterator[Actor]:
    """A fresh admin, whose knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def _unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def _shared_base(owner: Actor, name: str, group_id: str, permission: str) -> str:
    grants = [grant("group", group_id, "read")]
    if permission == "write":
        grants.append(grant("group", group_id, "write"))
    with owner.client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": name, "description": "the harbour", "access_grants": grants},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _attach_from_hash(page: Page, name: str) -> Locator:
    """The base offered by `#` for its name, in a new chat."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"#{name.split()[0]}")
    return page.get_by_role("tooltip").get_by_role("button", name=name)


def _ask_and_open_citation(page: Page, upstream, name: str, question: str, answer: str) -> Locator:
    """Attach the base, ask, and open the reply's single source; returns the citation dialog."""
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    _attach_from_hash(page, name).click()
    send(page, question)
    expect_reply(page, answer)
    chat = conversation(page)
    chat.get_by_role("button", name="Toggle 1 source").click()
    return chat


def _sent_for(upstream, question: str) -> str:
    return json.dumps(next(filter(reply.answering(question), upstream.chat_requests())))


def _file_row(base: Locator, filename: str) -> Locator:
    entry = base.page.get_by_role("button", name=re.compile(rf"^{re.escape(filename)}(\s|$)"))
    return base.locator("[data-knowledge-row]").filter(has=entry)


def test_a_reader_cites_the_shared_base_in_chat_but_cannot_change_its_files(
    admin, make_user, curator, page_for, upstream
):
    reader, outsider = make_user(), make_user()
    group_id = make_group(admin, [reader], KNOWLEDGE_USER)
    make_group(admin, [outsider], KNOWLEDGE_USER)
    name = _unique("Harbour")
    knowledge_id = _shared_base(curator, name, group_id, "read")
    with curator.client() as client:
        add_text_file(client, knowledge_id, "gate.txt", "The harbour gate code is 4242.")

    page = page_for(reader)
    question = "what is the harbour gate code?"
    chat = _ask_and_open_citation(page, upstream, name, question, "It is 4242.")
    assert "harbour gate code is 4242" in _sent_for(upstream, question)
    chat.get_by_role("button", name="View source: gate.txt").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name="gate.txt")).to_be_visible()
    expect(citation).to_contain_text("The harbour gate code is 4242.")

    page.goto(f"/workspace/knowledge/{knowledge_id}")
    base = page.get_by_role("main")
    expect(base.get_by_text("Read Only")).to_be_visible()
    row = _file_row(base, "gate.txt")
    expect(row).to_be_visible()
    # the menu's trigger wraps the labelled button
    row.get_by_role("button", name="More").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Download")).to_be_visible()
    expect(menu.get_by_role("button", name="Rename")).to_have_count(0)
    expect(menu.get_by_role("button", name="Remove from knowledge")).to_have_count(0)
    page.keyboard.press("Escape")
    expect(base.get_by_role("button", name="Add Content")).to_have_count(0)

    outsider_page = page_for(outsider)
    expect(_attach_from_hash(outsider_page, name)).to_have_count(0)


def test_a_writer_adds_a_file_that_the_owners_next_chat_cites(
    admin, make_user, curator, page_for, upstream
):
    writer = make_user()
    group_id = make_group(admin, [writer], KNOWLEDGE_USER)
    name = _unique("Harbour")
    knowledge_id = _shared_base(curator, name, group_id, "write")

    writer_page = page_for(writer)
    writer_page.goto(f"/workspace/knowledge/{knowledge_id}")
    base = writer_page.get_by_role("main")
    expect(base.get_by_role("textbox", name="Knowledge Name")).to_be_enabled()
    with writer_page.expect_file_chooser() as chooser:
        # the menu's trigger wraps the labelled button
        base.get_by_role("button", name="Add Content").last.click()
        writer_page.get_by_role("menu").get_by_role("button", name="Upload files").click()
    chooser.value.set_files(
        {"name": "fuel.txt", "mimeType": "text/plain", "buffer": b"The fuel dock opens at 7.\n"}
    )
    expect(writer_page.get_by_text("File added successfully.")).to_be_visible()
    expect(_file_row(base, "fuel.txt")).to_be_visible()

    owner_page = page_for(curator)
    question = "when does the fuel dock open?"
    chat = _ask_and_open_citation(owner_page, upstream, name, question, "At seven.")
    assert "fuel dock opens at 7" in _sent_for(upstream, question)
    chat.get_by_role("button", name="View source: fuel.txt").click()
    expect(owner_page.get_by_role("dialog")).to_contain_text("The fuel dock opens at 7.")
