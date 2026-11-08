"""Journey: the chat permissions an admin switches off in the defaults leave a user's chat page.

Admin Panel > Users > Groups > Default permissions holds a switch for each chat control a user
gets: the Controls panel, sharing, downloading and deleting a chat, editing, reading aloud,
rating, continuing, regenerating, forking and deleting a reply, dictation, Voice mode, comparing
models and uploading files. Each switch is on by default and the control is there; switched off
in the dialog and saved, the control is gone from the next account's open chat. An admin keeps
every control whatever the defaults say. Allow Chat Export also decides whether a folder's menu in
the sidebar and the Actions menu of Archived Chats offer Export. Allow Side Chat (0.13.0) is the
one switch that starts off: without it the / menu has no Side chat and a typed `/side` only says
the account may not use it, nothing reaches the model; with it, or as an admin, `/side` opens one,
named after its question since 3b1a95383 saves it as a chat.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Default permissions dialog
saves the permissions it opened with, every "withdrawn" test and the file upload refusal fail
while the "offered" and admin ones still pass. In the pre-fix build of 0.12.4 (a21335bdd and
c9936e81e reverted) the export test fails with the permission withheld: the folder menu offers
Export, and with the folder check skipped, so does Archived Chats. On dev ca8f451cc (before the
#32832 fix) the own-message Edit test fails with the permission withheld: the user's own
message still offers Edit. The side chat tests pass on v0.13.0 and dev 5f1c6384e; with the
backend's side chat default switched on the default test fails, and in a v0.13.0 frontend build
that skips the Allow Side Chat check the withheld test fails as the / menu offers Side chat.
Retargeted for 3b1a95383, the side chat tests pass on dev c165ee14f.

Since 3b1a95383 Allow Side Chat no longer needs Allow Temporary Chat: an account with only Side
Chat opens a side chat with `/side`, and it is saved to its chat list. The editor should offer
Allow Side Chat with Temporary Chat off as well; that test is red on 0.13.1, a real bug, as the
switch is hidden there. Discriminates: the side chat without Temporary Chat test passes on the
e8c0ad85e build and fails in a build of it whose side chat still requires Temporary Chat.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat import ask
from utils.chat_controls import controls_entry
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    last_reply,
    replies,
    send,
    side_chat_panel,
    side_chat_tab,
)
from utils.model_selector import SELECTOR_BUTTON
from utils.sidebar_folders import folder_link, folder_row
from utils.user_menu import open_settings

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
    page = page_for(account, path=f"/c/{first.chat_id}")
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
        return page.get_by_role("menu").get_by_role("menuitem", name=name, exact=True)

    return locate


def page_button(name: str) -> Callable[[Page], Locator]:
    return lambda page: page.get_by_role("button", name=name, exact=True)


def compare_button(page: Page) -> Locator:
    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    return page.get_by_role("button", name="Compare", exact=True)


CONTROLS = {
    "Allow Chat Controls": controls_entry,
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


def create_folder(account) -> str:
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    with account.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name})
    assert created.status_code == 200, created.text
    return name


def folder_menu_entries(page: Page, name: str) -> Locator:
    """The entries of the "More" menu on the folder's sidebar row, once they show."""
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    folder_link(sidebar, name).focus()
    folder_row(sidebar, name).get_by_role("button").last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Delete")).to_be_visible()
    return menu


def archive_menu_entries(page: Page) -> Locator:
    """The entries of the Actions menu in Settings > Archived Chats, once they show."""
    open_settings(page)
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Archived Chats").click()
    settings.locator("#tab-archived-chats").get_by_role("button", name="Actions").first.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Unarchive All")).to_be_visible()
    return menu


# #32703 (fix a21335bdd) and #32684 (fix c9936e81e): a folder's Export and the archive's Export
# were offered whatever Allow Chat Export said.
@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "withheld"])
def test_folder_and_archive_export_follow_allow_chat_export(
    allowed, page_for, admin, make_user, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Chat Export", turn_on=allowed)
    account = make_user()
    folder = create_folder(account)
    page = page_for(account)
    expected = 1 if allowed else 0

    folder_export = folder_menu_entries(page, folder).get_by_role("menuitem", name="Export")
    expect(folder_export, "a folder's Export entry").to_have_count(expected)
    page.keyboard.press("Escape")
    archive_export = archive_menu_entries(page).get_by_role("menuitem", name="Export")
    expect(archive_export, "the Archived Chats Export entry").to_have_count(expected)


# #32832: with Allow Chat Edit off, a user could still edit their own messages.
@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "withheld"])
def test_edit_on_the_users_own_message_follows_allow_chat_edit(
    allowed, page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Chat Edit", turn_on=allowed)
    page = open_answered_chat(page_for, make_user(), upstream)

    edit = message_button("user", "Edit")(page)
    expect(edit, "Edit on the user's own message").to_have_count(1 if allowed else 0)


def test_allow_side_chat_is_off_by_default(page_for, admin):
    page = page_for(admin, path="/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()

    switch = page.get_by_role("dialog").get_by_role("switch", name="Allow Side Chat", exact=True)
    expect(switch).to_have_attribute("aria-checked", "false")


SIDE_QUESTION = "By the way, is the road open in winter? ({tag})"
SIDE_ANSWER = "No, it closes from November to May."


def side_chat_menu_row(page: Page) -> Locator:
    """The Side chat row of the / menu, once the menu shows its Settings row."""
    chat_input(page).click()
    page.keyboard.type("/s")
    menu = page.get_by_role("tooltip")
    expect(menu.get_by_role("button", name=re.compile(r"^Settings"))).to_be_visible()
    return menu.get_by_role("button", name="Side chat", exact=True)


# An admin opens side chats whatever Allow Side Chat says.
@pytest.mark.parametrize(
    ("allowed", "role"), [(True, "user"), (False, "admin")], ids=["allowed", "admin-withheld"]
)
def test_side_opens_a_side_chat_with_allow_side_chat_or_as_admin(
    allowed, role, page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Side Chat", turn_on=allowed)
    question = SIDE_QUESTION.format(tag=uuid.uuid4().hex[:8])
    page = open_answered_chat(page_for, make_user(role=role), upstream)
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))

    expect(side_chat_menu_row(page), "the / menu's Side chat").to_be_visible()
    chat_input(page).press("Control+a")
    chat_input(page).press("Delete")
    send(page, f"/side {question}")

    expect(side_chat_panel(page, question)).to_contain_text(SIDE_ANSWER, timeout=REPLY_TIMEOUT_MS)


def test_side_does_nothing_without_allow_side_chat(page_for, admin, make_user, upstream, preserve):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Side Chat", turn_on=False)
    question = SIDE_QUESTION.format(tag=uuid.uuid4().hex[:8])
    page = open_answered_chat(page_for, make_user(), upstream)
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))

    expect(side_chat_menu_row(page), "the / menu's Side chat").to_have_count(0)
    chat_input(page).press("Control+a")
    chat_input(page).press("Delete")
    send(page, f"/side {question}")

    expect(page.get_by_text("You do not have permission to use side chat.")).to_be_visible()
    # an unsent side chat keeps its "Side chat 1" label, so no tab may open beside the chat's own,
    # which leads the tab bar on a wide screen since 617618c9e
    expect(page.get_by_role("tablist", name="Tabs").get_by_role("tab")).to_have_count(1)
    expect(side_chat_tab(page, question)).to_have_count(0)
    expect(replies(page)).to_have_count(2)
    asked = [body for body in upstream.chat_requests() if reply.mentioning(question)(body)]
    assert asked == [], f"the side question reached the model without the permission: {asked}"


def test_allow_side_chat_stays_on_offer_with_temporary_chat_off(page_for, admin, preserve):
    """Red on e8c0ad85e: the editor still hides Allow Side Chat while Temporary Chat is off."""
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow Temporary Chat", turn_on=False)
    page = page_for(admin, path="/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("switch", name="Allow Temporary Chat", exact=True)).to_have_attribute(
        "aria-checked", "false"
    )

    expect(
        dialog.get_by_role("switch", name="Allow Side Chat", exact=True),
        "Allow Side Chat with Temporary Chat off",
    ).to_be_visible()


def test_side_opens_a_saved_side_chat_without_the_temporary_chat_permission(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True, temporary=False)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    account = make_user()
    question = SIDE_QUESTION.format(tag=uuid.uuid4().hex[:8])
    page = open_answered_chat(page_for, account, upstream)
    upstream.queue(reply.text(SIDE_ANSWER, match=reply.answering(question)))

    send(page, f"/side {question}")

    expect(side_chat_panel(page, question)).to_contain_text(SIDE_ANSWER, timeout=REPLY_TIMEOUT_MS)
    with account.client() as client:

        def titles() -> list[str]:
            listed = client.get("/api/v1/chats/", params={"page": 1})
            assert listed.status_code == 200, listed.text
            return [chat["title"] for chat in listed.json()]

        assert wait_until(lambda: question in titles(), timeout=10), (
            f"the side chat was not saved to the chat list: {titles()}"
        )


def test_with_file_upload_off_upload_files_opens_no_file_picker(
    page_for, admin, make_user, upstream, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Allow File Upload", turn_on=False)
    page = open_answered_chat(page_for, make_user(), upstream)
    choosers = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))

    page.get_by_role("button", name="More", exact=True).last.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()

    expect(page.get_by_role("menu")).to_be_visible()
    page.wait_for_timeout(1000)  # a picker would open at once
    assert choosers == []


def test_with_file_upload_on_upload_files_attaches_the_file(page_for, make_user, upstream):
    page = open_answered_chat(page_for, make_user(), upstream)

    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files(
        files=[{"name": "route.txt", "mimeType": "text/plain", "buffer": b"Heiligenblut"}]
    )

    expect(page.get_by_text("route.txt")).to_be_visible()
