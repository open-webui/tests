"""Journey: the chat settings' bulk actions archive, list, delete, export and import chats.

Archive All Chats and Delete All Chats each ask for confirmation, and cancelling keeps every chat.
The archived chats list searches by title, and its rows unarchive a chat back into the sidebar or
delete it for good. Export downloads a file holding the account's chats, archived ones included and
no one else's, and importing that file into an emptied account brings the chats back. Each result
is read back after a reload, so what shows is what the server stored. The archive's Actions menu
unarchives every chat back into the sidebar and exports the archived chats alone, and Delete All
also closes the account's share links, so someone it shared a chat with is sent home.

Discriminates: passes on dev 176d31d1d; in a frontend copy each test fails when its control does
nothing: the archive, delete and unarchive buttons skipping their request, the confirmation
dialog confirming on cancel, the archive search ignoring its text and the export saving an empty
list or the import dropping the file. On dev ebc6add67 a frontend copy whose Unarchive All skips
its request and whose archive export saves an empty list turns those two tests red, and a
backend copy whose delete-all leaves the account's shares in place turns the share link test red.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from harness.chat_history import seed_chat
from utils.chat_ui import conversation

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    return make_user()


def _seed_chats(account, *titles: str, archived: bool = False) -> None:
    with account.client() as client:
        for title in titles:
            created = client.post("/api/v1/chats/new", json={"chat": {"title": title}})
            assert created.status_code == 200, created.text
            if archived:
                toggled = client.post(f"/api/v1/chats/{created.json()['id']}/archive")
                assert toggled.status_code == 200, toggled.text


def _titles(account, archived: bool = False) -> set[str]:
    path = "/api/v1/chats/all/archived" if archived else "/api/v1/chats/"
    with account.client() as client:
        listed = client.get(path)
    assert listed.status_code == 200, listed.text
    return {chat["title"] for chat in listed.json()}


def _open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def _open_settings_tab(page: Page, tab: str) -> Locator:
    page.get_by_role("button", name="User menu").first.click()
    page.get_by_role("menu").get_by_role("button", name="Settings").click()
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name=tab).click()
    return settings


def _click_bulk_action(settings: Locator, button: str) -> Locator:
    """Press the button and return the confirmation dialog it opens."""
    settings.get_by_role("button", name=button, exact=True).click()
    dialog = settings.page.get_by_role("dialog").last
    expect(dialog.get_by_role("button", name="Confirm")).to_be_visible()
    return dialog


def _archived_row(settings: Locator, title: str) -> Locator:
    return settings.get_by_role("link", name=title)


def _archived_actions(settings: Locator, title: str) -> Locator:
    """The row of an archived chat, which holds its Unarchive and Delete buttons."""
    return settings.locator("#tab-archived-chats div.flex.w-full").filter(has_text=title)


def test_archive_all_moves_every_chat_into_the_archive(owner, make_user, page_for):
    stranger = make_user()
    _seed_chats(owner, "Alpha plans", "Beta plans")
    _seed_chats(stranger, "Stranger plans")
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()

    settings = _open_settings_tab(page, "Data Controls")
    _click_bulk_action(settings, "Archive All").get_by_role("button", name="Confirm").click()
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_have_count(0)
    expect(sidebar.get_by_role("button", name="Beta plans")).to_have_count(0)

    page.reload()
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Alpha plans")).to_be_visible()
    expect(_archived_row(settings, "Beta plans")).to_be_visible()
    assert _titles(owner) == set()
    assert _titles(stranger) == {"Stranger plans"}


def test_cancelling_archive_all_keeps_the_chats_in_the_sidebar(owner, page_for):
    _seed_chats(owner, "Alpha plans", "Beta plans")
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()

    settings = _open_settings_tab(page, "Data Controls")
    _click_bulk_action(settings, "Archive All").get_by_role("button", name="Cancel").click()
    expect(page.get_by_role("dialog")).to_have_count(1)

    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()
    expect(sidebar.get_by_role("button", name="Beta plans")).to_be_visible()
    assert _titles(owner) == {"Alpha plans", "Beta plans"}
    assert _titles(owner, archived=True) == set()


def test_archived_chats_are_searched_by_title(owner, page_for):
    _seed_chats(owner, "Garden notes", "Tax notes", "Garden layout", archived=True)
    settings = _open_settings_tab(page_for(owner), "Archived Chats")
    expect(_archived_row(settings, "Tax notes")).to_be_visible()

    settings.locator("#tab-archived-chats").get_by_placeholder("Search").fill("Garden")
    expect(_archived_row(settings, "Tax notes")).to_have_count(0)
    expect(_archived_row(settings, "Garden notes")).to_be_visible()
    expect(_archived_row(settings, "Garden layout")).to_be_visible()

    settings.get_by_role("button", name="Clear search").click()
    expect(_archived_row(settings, "Tax notes")).to_be_visible()


def test_an_unarchived_chat_returns_to_the_sidebar(owner, page_for):
    _seed_chats(owner, "Kept in the archive", "Brought back", archived=True)
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Brought back")).to_be_visible()
    expect(sidebar.get_by_role("button", name="Brought back")).to_have_count(0)

    _archived_actions(settings, "Brought back").get_by_role("button", name="Unarchive Chat").click()
    expect(_archived_row(settings, "Brought back")).to_have_count(0)
    expect(sidebar.get_by_role("button", name="Brought back")).to_be_visible()

    page.reload()
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Kept in the archive")).to_be_visible()
    expect(_archived_row(settings, "Brought back")).to_have_count(0)
    assert _titles(owner) == {"Brought back"}


def test_a_chat_deleted_from_the_archive_is_gone_for_good(owner, page_for):
    _seed_chats(owner, "Kept in the archive", "Deleted from the archive", archived=True)
    page = page_for(owner)
    settings = _open_settings_tab(page, "Archived Chats")
    doomed = "Deleted from the archive"
    expect(_archived_row(settings, doomed)).to_be_visible()

    _archived_actions(settings, doomed).get_by_role("button", name="Delete Chat").click()
    page.get_by_role("dialog").last.get_by_role("button", name="Confirm").click()
    expect(_archived_row(settings, doomed)).to_have_count(0)

    page.reload()
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Kept in the archive")).to_be_visible()
    expect(_archived_row(settings, doomed)).to_have_count(0)
    assert _titles(owner, archived=True) == {"Kept in the archive"}


def test_delete_all_removes_every_chat_after_confirmation(owner, make_user, page_for):
    stranger = make_user()
    _seed_chats(owner, "Alpha plans")
    _seed_chats(owner, "Beta plans", archived=True)
    _seed_chats(stranger, "Stranger plans")
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()

    settings = _open_settings_tab(page, "Data Controls")
    _click_bulk_action(settings, "Delete All").get_by_role("button", name="Confirm").click()
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_have_count(0)

    page.reload()
    settings = _open_settings_tab(page, "Archived Chats")
    expect(settings.get_by_text("You have no archived conversations.")).to_be_visible()
    assert _titles(owner) == set()
    assert _titles(owner, archived=True) == set()
    assert _titles(stranger) == {"Stranger plans"}


def test_cancelling_delete_all_keeps_every_chat(owner, page_for):
    _seed_chats(owner, "Alpha plans")
    _seed_chats(owner, "Beta plans", archived=True)
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()

    settings = _open_settings_tab(page, "Data Controls")
    _click_bulk_action(settings, "Delete All").get_by_role("button", name="Cancel").click()
    expect(page.get_by_role("dialog")).to_have_count(1)

    expect(sidebar.get_by_role("button", name="Alpha plans")).to_be_visible()
    assert _titles(owner) == {"Alpha plans"}
    assert _titles(owner, archived=True) == {"Beta plans"}


def _export_chats(page: Page, tmp_path) -> tuple[str, list[dict]]:
    settings = _open_settings_tab(page, "Data Controls")
    with page.expect_download() as download_info:
        settings.get_by_role("button", name="Export", exact=True).click()
    download = download_info.value
    path = tmp_path / download.suggested_filename
    download.save_as(path)
    return download.suggested_filename, json.loads(path.read_text())


def test_export_downloads_the_accounts_chats_and_no_one_elses(owner, make_user, page_for, tmp_path):
    stranger = make_user()
    _seed_chats(owner, "Alpha plans")
    _seed_chats(owner, "Beta plans", archived=True)
    _seed_chats(stranger, "Stranger plans")

    filename, exported = _export_chats(page_for(owner), tmp_path)

    assert filename.startswith("chat-export-")
    assert filename.endswith(".json")
    assert {chat["title"] for chat in exported} == {"Alpha plans", "Beta plans"}


def test_an_exported_file_imports_back_into_an_emptied_account(owner, page_for, tmp_path):
    marker = uuid.uuid4().hex[:6]
    _seed_chats(owner, f"Alpha {marker}", f"Beta {marker}")
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    _, exported = _export_chats(page, tmp_path)
    assert len(exported) == 2
    settings = page.get_by_role("dialog")
    _click_bulk_action(settings, "Delete All").get_by_role("button", name="Confirm").click()
    expect(sidebar.get_by_role("button", name=f"Alpha {marker}")).to_have_count(0)
    assert _titles(owner) == set()

    with page.expect_file_chooser() as chooser_info:
        settings.get_by_role("button", name="Import", exact=True).click()
    chooser_info.value.set_files(
        {
            "name": "chats.json",
            "mimeType": "application/json",
            "buffer": json.dumps(exported).encode(),
        }
    )
    expect(page.get_by_text("Successfully imported 2 chats.")).to_be_visible()
    expect(sidebar.get_by_role("button", name=f"Alpha {marker}")).to_be_visible()

    page.reload()
    assert _titles(owner) == {f"Alpha {marker}", f"Beta {marker}"}


def _archive_action(settings: Locator, action: str) -> None:
    settings.locator("#tab-archived-chats").get_by_role("button", name="Actions").first.click()
    settings.page.get_by_role("menu").get_by_role("button", name=action).click()


def test_unarchive_all_brings_every_archived_chat_back(owner, make_user, page_for):
    stranger = make_user()
    _seed_chats(owner, "Kept out", "Brought in", archived=True)
    _seed_chats(stranger, "Stranger archive", archived=True)
    page = page_for(owner)
    sidebar = _open_sidebar(page)
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Brought in")).to_be_visible()

    _archive_action(settings, "Unarchive All")
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Unarchive All"
    ).click()
    expect(page.get_by_text("All chats have been unarchived.")).to_be_visible()
    expect(settings.get_by_text("You have no archived conversations.")).to_be_visible()
    expect(sidebar.get_by_role("button", name="Brought in")).to_be_visible()

    page.reload()  # the sidebar stays open
    expect(sidebar.get_by_role("button", name="Kept out")).to_be_visible()
    assert _titles(owner) == {"Kept out", "Brought in"}
    assert _titles(stranger, archived=True) == {"Stranger archive"}


def test_export_from_the_archive_downloads_only_the_archived_chats(owner, page_for, tmp_path):
    _seed_chats(owner, "Live plans")
    _seed_chats(owner, "Old plans", "Older plans", archived=True)
    page = page_for(owner)
    settings = _open_settings_tab(page, "Archived Chats")
    expect(_archived_row(settings, "Old plans")).to_be_visible()

    with page.expect_download() as download_info:
        _archive_action(settings, "Export")
    download = download_info.value
    path = tmp_path / download.suggested_filename
    download.save_as(path)
    exported = json.loads(path.read_text())

    assert download.suggested_filename.startswith("archived-chat-export-")
    assert {chat["title"] for chat in exported} == {"Old plans", "Older plans"}


def _share_with(account, reader) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "where to?"},
                {"role": "assistant", "content": "the dunes"},
            ],
        )
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("user", reader.id, "read")]},
        )
        assert granted.status_code == 200, granted.text
    return shared.json()["share_id"]


def test_delete_all_closes_the_accounts_share_links(owner, make_user, page_for):
    reader = make_user()
    share_id = _share_with(owner, reader)
    reader_page = page_for(reader)
    reader_page.goto(f"/s/{share_id}")
    expect(conversation(reader_page).get_by_text("the dunes")).to_be_visible()

    page = page_for(owner)
    settings = _open_settings_tab(page, "Data Controls")
    confirm = _click_bulk_action(settings, "Delete All")
    with page.expect_response(lambda response: response.request.method == "DELETE"):
        confirm.get_by_role("button", name="Confirm").click()
    assert _titles(owner) == set()

    reader_page.reload()
    expect(reader_page).to_have_url(re.compile(r"/$"))
    expect(conversation(reader_page).get_by_text("the dunes")).to_have_count(0)
    shared_chats = settings.get_by_text("Shared Chats", exact=True)
    shared_chats.locator("xpath=ancestor::div[.//button][1]").get_by_role("button").click()
    shared = page.get_by_role("dialog").filter(has_text="Unshare All Shared Chats")
    expect(shared.get_by_text("No results found")).to_be_visible()
