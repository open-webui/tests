"""Journey: the Files section of a chat's Controls and the permissions that shape the panel.

A file attached to a message stays with the chat: later messages are sent with it too, and the
Controls panel lists it under Files. Removed there, it is no longer sent with the next message
and stays removed after a reload. The Default permissions an admin keeps under Allow Chat
Controls (Allow Chat Valves, Allow Chat System Prompt and Allow Chat Params) each take one section
from a user's Controls when switched off, leaving the others, while an admin keeps all three.

Discriminates: passes on dev 30f3f6a8f; on a frontend build whose Controls ignore the chat
permissions every "withdrawn" test fails, and on one whose Files section keeps a removed file
(or does not save the removal) the removal tests fail.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.cached_chat import attach
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SECTIONS = ["Valves", "System Prompt", "Advanced Params"]
SWITCHES = {
    "Valves": "Allow Chat Valves",
    "System Prompt": "Allow Chat System Prompt",
    "Advanced Params": "Allow Chat Params",
}


def open_controls(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    expect(page.get_by_role("button", name="Controls", exact=True).last).to_be_visible()
    return page.locator("body")


def section(panel: Locator, title: str) -> Locator:
    return panel.get_by_role("button", name=title, exact=True)


def listed_file(panel: Locator, name: str) -> Locator:
    # only the Files section lists a chat file with a remove button
    return panel.get_by_role("button", name=re.compile(rf"^{re.escape(name)}.*Remove File$"))


def remove_listed_file(panel: Locator, name: str) -> None:
    listed = listed_file(panel, name)
    listed.hover()
    listed.get_by_role("button", name="Remove File", exact=True).click()


def ask(page: Page, upstream) -> str:
    """Sends a fresh question and returns everything the provider was sent for it."""
    question = f"what do my notes say? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Noted.")
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return str(request["messages"])


# --------------------------------------------------------------------------- files


@pytest.fixture
def chat_with_a_file(page_for, make_user, upstream) -> Page:
    page = page_for(make_user())
    attach(page, "harbour.txt", "The harbour notes mention cormorants.")
    expect(page.locator("form").get_by_text("harbour.txt")).to_be_visible()
    assert "cormorants" in ask(page, upstream)
    expect(page).to_have_url(re.compile(r"/c/"))
    return page


def test_a_file_attached_earlier_is_sent_again_and_listed_under_files(chat_with_a_file, upstream):
    page = chat_with_a_file
    assert "cormorants" in ask(page, upstream)

    panel = open_controls(page)

    expect(section(panel, "Files")).to_be_visible()
    expect(listed_file(panel, "harbour.txt")).to_be_visible()


def test_a_file_removed_under_files_is_no_longer_sent(chat_with_a_file, upstream):
    page = chat_with_a_file
    panel = open_controls(page)
    remove_listed_file(panel, "harbour.txt")
    expect(section(panel, "Files")).to_have_count(0)

    assert "cormorants" not in ask(page, upstream)


def test_a_file_removed_under_files_stays_removed_after_a_reload(chat_with_a_file, upstream):
    page = chat_with_a_file
    panel = open_controls(page)
    with page.expect_response(lambda response: re.search(r"/api/v1/chats/[^/]+$", response.url)):
        remove_listed_file(panel, "harbour.txt")

    page.reload()
    expect_reply(page, "Noted.")
    # the panel stays open across the reload
    expect(section(panel, "Advanced Params")).to_be_visible()
    expect(section(panel, "Files")).to_have_count(0)
    assert "cormorants" not in ask(page, upstream)


# --------------------------------------------------------------------------- permissions


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


def test_every_section_is_offered_by_default(page_for, make_user):
    panel = open_controls(page_for(make_user()))

    for title in SECTIONS:
        expect(section(panel, title)).to_be_visible()


@pytest.mark.parametrize("withdrawn", SECTIONS)
def test_a_section_is_withdrawn_once_its_switch_is_off(
    withdrawn, page_for, admin, make_user, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), SWITCHES[withdrawn], turn_on=False)

    panel = open_controls(page_for(make_user()))

    for title in SECTIONS:
        expected = expect(section(panel, title))
        if title == withdrawn:
            expected.to_have_count(0)
        else:
            expected.to_be_visible()


def test_an_admin_keeps_every_section_when_they_are_switched_off(page_for, admin, preserve):
    preserve("permissions")
    admin_page = page_for(admin)
    for switch in SWITCHES.values():
        save_default_permission(admin_page, switch, turn_on=False)

    panel = open_controls(page_for(admin))

    for title in SECTIONS:
        expect(section(panel, title)).to_be_visible()
