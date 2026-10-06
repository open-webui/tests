"""Journey: Tika picked as the Content Extraction Engine in Admin Settings > Documents reads files.

With Tika and its server URL saved, a document a user attaches to a chat is sent to the Tika
server, at the endpoint of the Tika version picked, and the text Tika returns is what the model
reads with the question. A plain text file is still read as it is, without Tika.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Documents form sends the stored
engine, URL and version back in place of the edited ones, both Tika tests fail.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.listener import ReceivedRequest, json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EXTRACTED = "The tide table lists high water at 06:42 and 19:05."
QUESTION = "When is high water according to the report?"
ENDPOINTS = {"3": ("/tika/text", "X-TIKA:content"), "4": ("/tika/json/md", "tk:content")}


def serve_tika(listener, version: str) -> str:
    path, content_key = ENDPOINTS[version]

    def extract(_request: ReceivedRequest):
        return json_answer({content_key: EXTRACTED, "Content-Type": "application/pdf"})

    listener.route("PUT", path, extract)
    return path


def save_tika(page: Page, url: str, version: str) -> None:
    page.goto("/admin/settings/documents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    engine = settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Tika", exact=True)
    )
    engine.select_option("tika")
    settings.get_by_placeholder("Enter Tika Server URL").fill(url)
    settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Tika 4.x")
    ).select_option(version)
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def ask_about(page: Page, upstream, name: str, mime_type: str, content: bytes) -> str:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})
    expect(page.get_by_role("button", name=name)).to_be_visible()
    upstream.queue(reply.text("At 06:42.", match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, "At 06:42.")
    return json.dumps(next(filter(reply.answering(QUESTION), upstream.chat_requests())))


@pytest.mark.parametrize("version", ["3", "4"])
def test_a_document_is_read_through_the_tika_version_picked(
    version, page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    path = serve_tika(listener, version)
    save_tika(page_for(make_user(role="admin")), listener.base_url, version)

    sent = ask_about(
        page_for(make_user()), upstream, "report.pdf", "application/pdf", b"%PDF-1.4 scanned"
    )

    assert EXTRACTED in sent
    assert [request.method for request in listener.requests_to(path)] == ["PUT"]


def test_a_text_file_is_read_without_tika(page_for, make_user, upstream, listener, preserve):
    preserve(RETRIEVAL_CONFIG)
    serve_tika(listener, "3")
    save_tika(page_for(make_user(role="admin")), listener.base_url, "3")
    note = "The harbour master says high water is at 06:42."

    sent = ask_about(page_for(make_user()), upstream, "note.txt", "text/plain", note.encode())

    assert note in sent
    assert listener.requests_to("/tika/text") == []
