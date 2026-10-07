"""Journey: Settings > Data Controls exports a person's chats and manages their links and files.

Export Chats downloads every chat of the account as JSON and nobody else's. Shared Chats lists
the chats the account shared by link: unsharing one there closes its link at once, so the
account it was shared with is sent home from it, and Unshare All Shared Chats closes every link.
Manage Files lists the files the account uploaded and no one else's, with their number, narrows
them by a search on part of the name and opens one to show its text; a file deleted there is
gone, at once and without asking when Shift is held.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Export Chats saves an empty
list, whose unshare and Unshare All handlers skip their requests and whose file delete skips its
request, every test up to the deleted file fails. On dev ebc6add67, in a frontend build whose
Manage Files search ignores the query, whose rows open nothing and whose Shift delete still asks,
the search, open and Shift tests fail.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from utils.chat_ui import conversation
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def data_controls(page: Page) -> Locator:
    page.goto("/?settings=data_controls")
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("Export Chats", exact=True)).to_be_visible()
    return dialog


def row_button(settings: Locator, row_label: str, button: str) -> Locator:
    label = settings.get_by_text(row_label, exact=True)
    row = label.locator(f"xpath=ancestor::div[.//button[normalize-space()='{button}']][1]")
    return row.get_by_role("button", name=button, exact=True)


def new_chat(client, title: str) -> str:
    question = {"id": "q1", "parentId": None, "childrenIds": [], "role": "user", "content": title}
    history = {"currentId": "q1", "messages": {"q1": question}}
    created = client.post("/api/v1/chats/new", json={"chat": {"title": title, "history": history}})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def share(client, chat_id: str, *readers) -> str:
    shared = client.post(f"/api/v1/chats/{chat_id}/share")
    assert shared.status_code == 200, shared.text
    grants = [grant("user", reader.id, "read") for reader in readers]
    granted = client.post(
        f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": grants}
    )
    assert granted.status_code == 200, granted.text
    return shared.json()["share_id"]


def link_status(client, share_id: str) -> int:
    return client.get(f"/api/v1/chats/share/{share_id}").status_code


def test_export_chats_downloads_the_accounts_chats_and_no_one_elses(page_for, make_user):
    account, other = make_user(), make_user()
    titles = [f"Tide table {uuid.uuid4().hex[:6]}", f"Ferry plan {uuid.uuid4().hex[:6]}"]
    with account.client() as client:
        for title in titles:
            new_chat(client, title)
    with other.client() as client:
        foreign = new_chat(client, "Someone else's chat")
    page = page_for(account)
    settings = data_controls(page)

    with page.expect_download() as downloaded:
        row_button(settings, "Export Chats", "Export").click()
    exported = json.loads(Path(downloaded.value.path()).read_text())

    assert sorted(chat["title"] for chat in exported) == sorted(titles)
    assert foreign not in [chat["id"] for chat in exported]


def open_shared_chats(page: Page) -> Locator:
    row_button(data_controls(page), "Shared Chats", "Manage").click()
    shared = page.get_by_role("dialog").filter(has_text="Unshare All Shared Chats")
    expect(shared).to_be_visible()
    return shared


def test_unsharing_a_chat_in_shared_chats_closes_its_link(page_for, make_user):
    account, reader = make_user(), make_user()
    title = f"Harbour walk {uuid.uuid4().hex[:6]}"
    with account.client() as client:
        kept_id = new_chat(client, "Lighthouse visit")
        kept_share = share(client, kept_id)
        unshared_share = share(client, new_chat(client, title), reader)
    reader_page = page_for(reader)
    reader_page.goto(f"/s/{unshared_share}")
    expect(conversation(reader_page).get_by_text(title)).to_be_visible()
    page = page_for(account)
    shared = open_shared_chats(page)
    row = shared.locator("div").filter(has=page.get_by_role("link", name=title)).last

    row.hover()
    tooltip_button(row, "Unshare Chat").click()

    expect(page.get_by_text("Chat unshared successfully.")).to_be_visible()
    expect(shared.get_by_text(title)).to_have_count(0)
    with account.client() as client:
        assert link_status(client, unshared_share) == 401
        assert link_status(client, kept_share) == 200
    reader_page.reload()
    expect(reader_page).to_have_url(re.compile(r"/$"))
    expect(conversation(reader_page).get_by_text(title)).to_have_count(0)


def test_unshare_all_closes_every_link(page_for, make_user):
    account = make_user()
    with account.client() as client:
        links = [share(client, new_chat(client, f"Shared {index}")) for index in range(2)]
    page = page_for(account)
    shared = open_shared_chats(page)

    shared.get_by_role("button", name="Unshare All Shared Chats").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Unshare All"
    ).click()

    expect(page.get_by_text("All shared chats have been unshared.")).to_be_visible()
    with account.client() as client:
        assert [link_status(client, share_id) for share_id in links] == [401, 401]


def test_a_file_deleted_in_manage_files_is_gone(page_for, make_user):
    account = make_user()
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": ("harbour-map.txt", b"pier 3", "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    page = page_for(account)
    row_button(data_controls(page), "Manage Files", "Manage").click()
    files = page.get_by_role("dialog").filter(has=page.get_by_placeholder("Search Files"))
    expect(files.get_by_text("harbour-map.txt", exact=True)).to_be_visible()
    named = page.get_by_text("harbour-map.txt", exact=True)
    row = files.locator("div").filter(has=named).filter(has=page.get_by_role("button")).last

    tooltip_button(row, "Delete File").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(page.get_by_text("File deleted successfully.")).to_be_visible()
    expect(files.get_by_text("harbour-map.txt")).to_have_count(0)
    with account.client() as client:
        assert client.get(f"/api/v1/files/{file_id}").status_code == 404


def upload(account, name: str, text: str) -> str:
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": (name, text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def open_manage_files(page: Page) -> Locator:
    row_button(data_controls(page), "Manage Files", "Manage").click()
    files = page.get_by_role("dialog").filter(has=page.get_by_placeholder("Search Files"))
    expect(files).to_be_visible()
    return files


def file_row(page: Page, files: Locator, name: str) -> Locator:
    named = page.get_by_text(name, exact=True)
    return files.locator("div").filter(has=named).filter(has=page.get_by_role("button")).last


def test_manage_files_lists_the_accounts_files_and_searches_them_by_name(page_for, make_user):
    account, other = make_user(), make_user()
    upload(account, "tide-table.txt", "high at noon")
    upload(account, "ferry-plan.txt", "north pier at seven")
    upload(other, "someone-elses.txt", "not yours")
    page = page_for(account)
    files = open_manage_files(page)
    expect(files.get_by_text("tide-table.txt", exact=True)).to_be_visible()
    expect(files.get_by_text("ferry-plan.txt", exact=True)).to_be_visible()
    expect(files.get_by_text("someone-elses.txt")).to_have_count(0)
    expect(files.get_by_text("2", exact=True)).to_be_visible()

    files.get_by_placeholder("Search Files").fill("ferry")

    expect(files.get_by_text("tide-table.txt")).to_have_count(0)
    expect(files.get_by_text("ferry-plan.txt", exact=True)).to_be_visible()
    expect(files.get_by_text("1", exact=True)).to_be_visible()


def test_a_file_opened_in_manage_files_shows_its_text(page_for, make_user):
    account = make_user()
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            files={"file": ("harbour-map.txt", b"pier 3 has the pilot", "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    page = page_for(account)
    files = open_manage_files(page)

    files.get_by_text("harbour-map.txt", exact=True).click()

    viewer = page.get_by_role("dialog").filter(has_text="Extracted Lines")
    expect(viewer.get_by_text("pier 3 has the pilot")).to_be_visible()


def test_a_file_deleted_with_shift_held_goes_without_asking(page_for, make_user):
    account = make_user()
    file_id = upload(account, "old-chart.txt", "an old chart")
    page = page_for(account)
    files = open_manage_files(page)
    expect(files.get_by_text("old-chart.txt", exact=True)).to_be_visible()

    tooltip_button(file_row(page, files, "old-chart.txt"), "Delete File").click(modifiers=["Shift"])

    expect(page.get_by_text("File deleted successfully.")).to_be_visible()
    expect(page.get_by_role("dialog", name="Confirm your action")).to_have_count(0)
    expect(files.get_by_text("old-chart.txt")).to_have_count(0)
    with account.client() as client:
        assert client.get(f"/api/v1/files/{file_id}").status_code == 404
