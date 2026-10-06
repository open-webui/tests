"""Journey: a chat downloaded as text, JSON or PDF, or copied, from its header and sidebar menus.

The chat header's menu (Chat actions) and the chat's sidebar menu both offer Download with Plain
text (.txt), Export chat (.json) and PDF document (.pdf); the header menu also copies the whole
chat to the clipboard. Each file is read as the browser saved it: the text and the copy hold every
message of the open branch under its role, the JSON is the stored chat, a PDF written from text
carries the reply and the default picture PDF holds a page image.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose header menu writes the chat
text without the replies the header text, copy and text PDF tests fail, with the sidebar menu's
text download writing nothing the sidebar text test fails, and with both menus exporting an empty
list the two JSON tests fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE = "Packing list"
QUESTION = "What should I pack for a week of hiking?"
FIRST_ANSWER = "Boots, a rain jacket and two warm layers."
FOLLOW_UP = "And for food?"
SECOND_ANSWER = "Oats, nuts and dried fruit keep well."
CHAT_TEXT = (
    f"### USER\n{QUESTION}\n\n### ASSISTANT\n{FIRST_ANSWER}\n\n"
    f"### USER\n{FOLLOW_UP}\n\n### ASSISTANT\n{SECOND_ANSWER}"
)


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def chat_id(owner) -> str:
    with owner.client() as client:
        seeded, _ = seed_chat(
            client,
            [
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": FIRST_ANSWER},
                {"role": "user", "content": FOLLOW_UP},
                {"role": "assistant", "content": SECOND_ANSWER},
            ],
        )
        renamed = client.post(f"/api/v1/chats/{seeded}", json={"chat": {"title": TITLE}})
        assert renamed.status_code == 200, renamed.text
    return seeded


def _header_menu(page: Page, chat_id: str) -> Locator:
    page.goto(f"/c/{chat_id}")
    expect(page.get_by_text(SECOND_ANSWER)).to_be_visible()
    page.get_by_label("Chat actions").click()
    return page.get_by_role("menu")


def _sidebar_menu(page: Page) -> Locator:
    page.goto("/")
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    row = sidebar.locator("#sidebar-chat-group").filter(has_text=TITLE)
    row.hover()
    row.get_by_role("button", name=TITLE).first.focus()
    row.get_by_role("button", name="Chat Menu").first.click()
    return page.get_by_role("menu")


def _download(page: Page, menu: Locator, entry: str, tmp_path: Path) -> Path:
    """Choose `entry` under the menu's Download and return the file the browser saved."""
    menu.get_by_role("button", name="Download").hover()
    with page.expect_download() as download_info:
        page.get_by_role("menu").get_by_role("button", name=entry).click()
    download = download_info.value
    saved = tmp_path / download.suggested_filename
    download.save_as(saved)
    return saved


def test_the_header_text_download_holds_every_message_under_its_role(
    page_for, owner, chat_id, tmp_path
):
    page = page_for(owner)
    saved = _download(page, _header_menu(page, chat_id), "Plain text (.txt)", tmp_path)

    assert saved.name == f"chat-{TITLE}.txt"
    assert saved.read_text() == CHAT_TEXT


def test_the_sidebar_text_download_holds_every_message_under_its_role(
    page_for, owner, chat_id, tmp_path
):
    page = page_for(owner)
    saved = _download(page, _sidebar_menu(page), "Plain text (.txt)", tmp_path)

    assert saved.name == f"chat-{TITLE}.txt"
    assert saved.read_text() == CHAT_TEXT


def _exported_chat(saved: Path) -> dict:
    exported = json.loads(saved.read_text())
    assert len(exported) == 1, exported
    return exported[0]


def test_the_header_json_export_is_the_stored_chat(page_for, owner, chat_id, tmp_path):
    page = page_for(owner)
    saved = _download(page, _header_menu(page, chat_id), "Export chat (.json)", tmp_path)

    exported = _exported_chat(saved)
    assert exported["id"] == chat_id
    assert exported["title"] == TITLE
    contents = [message["content"] for message in exported["chat"]["history"]["messages"].values()]
    assert sorted(contents) == sorted([QUESTION, FIRST_ANSWER, FOLLOW_UP, SECOND_ANSWER])


def test_the_sidebar_json_export_is_the_stored_chat(page_for, owner, chat_id, tmp_path):
    page = page_for(owner)
    saved = _download(page, _sidebar_menu(page), "Export chat (.json)", tmp_path)

    exported = _exported_chat(saved)
    assert exported["id"] == chat_id
    assert exported["title"] == TITLE


def test_copy_puts_the_whole_chat_on_the_clipboard(page_for, owner, chat_id):
    page = page_for(owner, permissions=["clipboard-read", "clipboard-write"])
    _header_menu(page, chat_id).get_by_role("button", name="Copy", exact=True).click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    assert page.evaluate("navigator.clipboard.readText()") == CHAT_TEXT


def test_the_default_pdf_is_a_picture_of_the_chat(page_for, owner, chat_id, tmp_path):
    page = page_for(owner)
    saved = _download(page, _header_menu(page, chat_id), "PDF document (.pdf)", tmp_path)

    assert saved.name == f"chat-{TITLE}.pdf"
    written = saved.read_bytes()
    assert written.startswith(b"%PDF-")
    assert b"/Subtype /Image" in written


def test_a_pdf_without_styling_is_written_from_the_chat_text(page_for, owner, chat_id, tmp_path):
    with owner.client() as client:
        saved_settings = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"stylizedPdfExport": False}}
        )
        assert saved_settings.status_code == 200, saved_settings.text
    page = page_for(owner)
    saved = _download(page, _header_menu(page, chat_id), "PDF document (.pdf)", tmp_path)

    written = saved.read_bytes()
    assert written.startswith(b"%PDF-")
    assert SECOND_ANSWER.encode() in written
    assert b"### ASSISTANT" in written
