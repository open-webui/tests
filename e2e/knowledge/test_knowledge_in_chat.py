"""Journey: a knowledge base built in the workspace answers a chat that attaches it with `#`.

A fresh admin creates a knowledge base on the workspace's Knowledge page and uploads a text file
into it, then starts a chat, picks the base from the `#` picker and asks a question. The model
is sent the file's text as context, and the reply cites the file: its source list names the
file and opening the citation shows the text.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the retrieval of attached
collections switched off the model is sent the question alone, without the file's text.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FILE_NAME = "lake-house.txt"
FILE_TEXT = "The lake house gate code is 4242.\nThe boat key hangs by the door.\n"
QUESTION = "what is the gate code?"


@pytest.fixture
def curator(make_user):
    """A fresh admin, whose knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def test_an_uploaded_file_is_retrieved_and_cited_in_a_chat(page_for, curator, upstream):
    name = f"Lake {uuid.uuid4().hex[:6]}"
    page = page_for(curator)
    page.goto("/workspace/knowledge/create")
    creating = page.get_by_role("dialog")
    creating.get_by_role("textbox", name="Name your knowledge base").fill(name)
    creating.get_by_role("textbox", name="Describe your knowledge base and objectives").fill(
        "the lake house"
    )
    creating.get_by_role("button", name="Create Knowledge").click()
    expect(page).to_have_url(re.compile(r"/workspace/knowledge/[0-9a-f-]+$"))

    base = page.get_by_role("main")
    base.get_by_role("button", name="Add Content").first.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload files").click()
    chooser.value.set_files(
        {"name": FILE_NAME, "mimeType": "text/plain", "buffer": FILE_TEXT.encode()}
    )
    expect(base.get_by_role("button", name=re.compile(re.escape(FILE_NAME)))).to_be_visible()

    upstream.queue(reply.text("It is 4242.", match=reply.answering(QUESTION)))
    page.goto("/")
    chat_input(page).click()
    page.keyboard.type("#Lake")
    page.get_by_role("tooltip").get_by_role("button", name=name).click()
    send(page, QUESTION)
    expect_reply(page, "It is 4242.")

    sent = json.dumps(upstream.chat_requests()[-1]["messages"])
    assert "gate code is 4242" in sent

    chat = conversation(page)
    chat.get_by_role("button", name="Toggle 1 source").click()
    chat.get_by_role("button", name=f"View source: {FILE_NAME}").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=FILE_NAME)).to_be_visible()
    expect(citation).to_contain_text("The lake house gate code is 4242.")
