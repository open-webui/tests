"""Journey: the file limits and retrieval an admin sets in Admin Settings > Documents reach chats.

Max Upload Count and Max Upload Size refuse, in the chat input, the files beyond them, with the
limit named. Allowed File Extensions refuses a file of another kind. Top K decides how many
pieces of an uploaded file the model is given, and Bypass Embedding and Retrieval gives it the
whole file.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose retrieval settings update ignores
Top K, Bypass Embedding and Retrieval, Max Upload Size, Max Upload Count and Allowed File
Extensions, every test but the default upload fails.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LANDMARKS = ["Hochosterwitz", "Landskron", "Griffen", "Taggenbrunn", "Glanegg", "Liebenfels"]
QUESTION = "Which castles does the guide list?"


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's own settings."""
    return page_for(make_user(role="admin"))


def open_documents(page: Page) -> Locator:
    page.goto("/admin/settings/documents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("Allowed File Extensions", exact=True)).to_be_visible()
    return settings


def field(settings: Locator, label: str) -> Locator:
    return settings.get_by_text(label, exact=True).locator("xpath=following-sibling::div//input")


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def set_switch(settings: Locator, name: str, turn_on: bool) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def choose_files(page: Page, *files: dict) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files(list(files))


def text_file(name: str, text: str) -> dict:
    return {"name": name, "mimeType": "text/plain", "buffer": text.encode()}


def guide() -> dict:
    """A guide whose paragraphs each make one chunk of their own at a chunk size of 60."""
    paragraphs = [f"{castle} castle stands above the valley floor." for castle in LANDMARKS]
    return text_file("castles.txt", "\n\n".join(paragraphs))


def landmarks_sent(upstream) -> list[str]:
    sent = json.dumps(next(filter(reply.answering(QUESTION), upstream.chat_requests())))
    return [castle for castle in LANDMARKS if castle in sent]


def ask_about_the_guide(page: Page, upstream) -> list[str]:
    choose_files(page, guide())
    expect(page.get_by_role("button", name="castles.txt")).to_be_visible()
    upstream.queue(reply.text("Six castles.", match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, "Six castles.")
    return landmarks_sent(upstream)


def test_a_max_upload_count_refuses_more_files_at_once(admin_page, page_for, make_user, preserve):
    preserve(RETRIEVAL_CONFIG)
    settings = open_documents(admin_page)
    field(settings, "Max Upload Count").fill("1")
    save(admin_page, settings)
    page = page_for(make_user())

    choose_files(page, text_file("north.txt", "north"), text_file("south.txt", "south"))

    expect(
        page.get_by_text("You can only chat with a maximum of 1 file(s) at a time.")
    ).to_be_visible()
    expect(page.get_by_role("button", name="north.txt")).to_have_count(0)


def test_a_max_upload_size_refuses_a_larger_file(admin_page, page_for, make_user, preserve):
    preserve(RETRIEVAL_CONFIG)
    settings = open_documents(admin_page)
    field(settings, "Max Upload Size").fill("1")
    save(admin_page, settings)
    page = page_for(make_user())

    choose_files(page, text_file("atlas.txt", "x" * (2 * 1024 * 1024)))

    expect(page.get_by_text("File size should not exceed 1 MB.")).to_be_visible()
    expect(page.get_by_role("button", name="atlas.txt")).to_have_count(0)


def test_allowed_file_extensions_refuse_a_file_of_another_kind(
    admin_page, page_for, make_user, preserve
):
    preserve(RETRIEVAL_CONFIG)
    settings = open_documents(admin_page)
    field(settings, "Allowed File Extensions").fill("md")
    save(admin_page, settings)
    page = page_for(make_user())

    choose_files(page, text_file("notes.txt", "a plain text note"))

    expect(page.get_by_text("File type txt is not allowed")).to_be_visible()
    expect(page.get_by_role("button", name="notes.txt")).to_have_count(0)


def test_a_file_of_an_allowed_kind_still_uploads(page_for, make_user):
    page = page_for(make_user())

    choose_files(page, text_file("notes.txt", "a plain text note"))

    expect(page.get_by_role("button", name="notes.txt")).to_be_visible()


def test_top_k_decides_how_many_pieces_of_a_file_the_model_gets(
    admin_page, page_for, make_user, preserve, upstream
):
    preserve(RETRIEVAL_CONFIG)
    settings = open_documents(admin_page)
    set_switch(settings, "Bypass Embedding and Retrieval", turn_on=False)
    field(settings, "Chunk Size").fill("60")
    field(settings, "Chunk Overlap").fill("0")
    field(settings, "Top K").fill("2")
    save(admin_page, settings)

    sent = ask_about_the_guide(page_for(make_user()), upstream)

    assert len(sent) == 2, sent


def test_bypass_embedding_and_retrieval_gives_the_model_the_whole_file(
    admin_page, page_for, make_user, preserve, upstream
):
    preserve(RETRIEVAL_CONFIG)
    settings = open_documents(admin_page)
    field(settings, "Chunk Size").fill("60")
    field(settings, "Chunk Overlap").fill("0")
    field(settings, "Top K").fill("2")
    set_switch(settings, "Bypass Embedding and Retrieval", turn_on=True)
    save(admin_page, settings)

    sent = ask_about_the_guide(page_for(make_user()), upstream)

    assert sent == LANDMARKS
