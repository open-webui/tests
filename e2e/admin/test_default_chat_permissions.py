"""Journey: the chat permissions an admin switches off in the defaults leave a user's chat page.

Admin Panel > Users > Groups > Default permissions holds a switch for each chat control a user
gets: the Controls panel, sharing, downloading and deleting a chat, editing, reading aloud,
rating, continuing, regenerating, forking and deleting a reply, dictation, Voice mode, comparing
models and uploading files. Each switch is on by default and the control is there; switched off
in the dialog and saved, the control is gone from the next account's open chat. An admin keeps
every control whatever the defaults say.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Default permissions dialog
saves the permissions it opened with, every "withdrawn" test and the file upload refusal fail
while the "offered" and admin ones still pass.
"""

from __future__ import annotations

from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ask
from utils.chat_ui import last_reply
from utils.model_selector import SELECTOR_BUTTON

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT = "How long is the Glockner road?"
ANSWER = "The Grossglockner High Alpine Road is 48 kilometres long."
FOLLOW_UP = "And how high does it climb?"
FOLLOW_UP_ANSWER = "It climbs to 2504 metres at the Edelweissspitze."


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch, exact=True)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


def open_answered_chat(page_for, account, upstream) -> Page:
    """A chat of two answered questions, opened; only a later question can be deleted."""
    upstream.queue(
        reply.text(ANSWER, match=reply.answering(PROMPT)),
        reply.text(FOLLOW_UP_ANSWER, match=reply.answering(FOLLOW_UP)),
    )
    with account.client() as client:
        first, _ = ask(client, PROMPT)
        history = [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": ANSWER}]
        ask(
            client,
            FOLLOW_UP,
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=history,
        )
    page = page_for(account)
    page.goto(f"/c/{first.chat_id}")
    expect(last_reply(page)).to_contain_text(FOLLOW_UP_ANSWER)
    return page


def message_button(role: str, name: str) -> Callable[[Page], Locator]:
    """The button `name` on the last message of `role`, shown while the message is hovered."""

    def locate(page: Page) -> Locator:
        message = page.locator(f"[id^='message-']:has(.chat-{role})").last
        message.hover()
        return message.locator("button").and_(page.get_by_label(name, exact=True))

    return locate


def reply_button(name: str) -> Callable[[Page], Locator]:
    return message_button("assistant", name)


def chat_menu_entry(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        page.get_by_role("button", name="Chat actions").first.click()
        return page.get_by_role("menu").get_by_role("button", name=name, exact=True)

    return locate


def page_button(name: str) -> Callable[[Page], Locator]:
    return lambda page: page.get_by_role("button", name=name, exact=True)


def compare_button(page: Page) -> Locator:
    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    return page.get_by_role("button", name="Compare", exact=True)


CONTROLS = {
    "Allow Chat Controls": page_button("Controls"),
    "Allow Chat Share": chat_menu_entry("Share"),
    "Allow Chat Export": chat_menu_entry("Download"),
    "Allow Chat Delete": chat_menu_entry("Delete"),
    "Allow Chat Edit": reply_button("Edit"),
    "Allow Text to Speech": reply_button("Read Aloud"),
    "Allow Rate Response": reply_button("Good Response"),
    "Allow Continue Response": reply_button("Continue Response"),
    "Allow Regenerate Response": reply_button("Regenerate"),
    "Allow Chat Import": reply_button("Fork chat"),
    "Allow Delete Messages": message_button("user", "Delete"),
    "Allow Speech to Text": page_button("Voice Input"),
    "Allow Call": page_button("Voice mode"),
    "Allow Multiple Models in Chat": compare_button,
}


@pytest.mark.parametrize("switch", CONTROLS)
def test_the_control_is_offered_by_default(switch, page_for, make_user, upstream):
    page = open_answered_chat(page_for, make_user(), upstream)

    expect(CONTROLS[switch](page)).to_be_visible()


@pytest.mark.parametrize("switch", CONTROLS)
def test_the_control_is_withdrawn_once_switched_off(
    switch, page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), switch, turn_on=False)

    page = open_answered_chat(page_for, make_user(), upstream)

    expect(CONTROLS[switch](page)).to_have_count(0)


@pytest.mark.parametrize("switch", ["Allow Chat Delete", "Allow Regenerate Response"])
def test_an_admin_keeps_the_control_when_it_is_switched_off(
    switch, page_for, admin, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), switch, turn_on=False)

    page = open_answered_chat(page_for, admin, upstream)

    expect(CONTROLS[switch](page)).to_be_visible()


def test_with_file_upload_off_upload_files_opens_no_file_picker(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow File Upload", turn_on=False)
    page = open_answered_chat(page_for, make_user(), upstream)
    choosers = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))

    page.get_by_role("button", name="More", exact=True).last.click()
    page.get_by_role("menu").get_by_role("button", name="Upload Files").click()

    expect(page.get_by_role("menu")).to_be_visible()
    page.wait_for_timeout(1000)  # a picker would open at once
    assert choosers == []


def test_with_file_upload_on_upload_files_attaches_the_file(page_for, make_user, upstream):
    page = open_answered_chat(page_for, make_user(), upstream)

    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files(
        files=[{"name": "route.txt", "mimeType": "text/plain", "buffer": b"Heiligenblut"}]
    )

    expect(page.get_by_text("route.txt")).to_be_visible()
