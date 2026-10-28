"""Journey: Settings > Data Controls manages a person's shared links and files.

Export Chats is driven in e2e/chat/test_bulk_chat_actions.py. Shared Chats lists the chats the
account shared by link: unsharing one there closes its link at once, so the account it was shared
with is sent home from it, and Unshare All Shared Chats closes every link. Manage Files lists the
files the account uploaded and no one else's, with their number, narrows them by a search on part of
the name and opens one to show its text; a file deleted there is gone, at once and without asking
when Shift is held. An admin sees only their own files there too. Several files, picked one by one
or every file matching the search, are deleted together with a count of what was deleted, what
failed and what was left, or downloaded as one zip of exactly those files; a single file downloads
under its own name, and Stop ends either operation before the rest is done.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose unshare and Unshare All handlers
skip their requests and whose file delete skips its request, every test up to the deleted file
fails. On dev ebc6add67, in a frontend build whose Manage Files search ignores the query, whose rows
open nothing and whose Shift delete still asks, the search, open and Shift tests fail. On dev
226dca18c, in a frontend build that lists every file the admin can reach, counts a failed delete as
deleted, whose Select all matching ignores the search, whose zip takes every listed file, whose
single download is named files.zip and whose Stop is ignored, each of the seven tests after the
Shift test fails.
"""

from __future__ import annotations

import io
import pathlib
import re
import uuid
import zipfile

import pytest
from playwright.sync_api import Locator, Page, Route, expect

from harness.access import grant
from utils.chat_ui import conversation
from utils.setting_rows import setting_button
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def data_controls(page: Page) -> Locator:
    page.goto("/?settings=data_controls")
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("Export Chats", exact=True)).to_be_visible()
    return dialog


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


def open_shared_chats(page: Page) -> Locator:
    setting_button(data_controls(page), "Shared Chats").click()
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
    reader_page = page_for(reader, path=f"/s/{unshared_share}")
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
    setting_button(data_controls(page), "Manage Files").click()
    files = page.get_by_role("dialog").filter(has=page.get_by_placeholder("Search Files"))
    expect(files.get_by_text("harbour-map.txt", exact=True)).to_be_visible()

    files.get_by_role("button", name="Actions for harbour-map.txt").click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click()
    confirming = page.get_by_role("dialog").filter(has_text="Delete 1 file?")
    confirming.get_by_role("button", name="Delete", exact=True).click()

    expect(page.get_by_text("1 deleted, 0 failed, 0 not processed.")).to_be_visible()
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
    setting_button(data_controls(page), "Manage Files").click()
    files = page.get_by_role("dialog").filter(has=page.get_by_placeholder("Search Files"))
    expect(files).to_be_visible()
    return files


def test_manage_files_lists_the_accounts_files_and_searches_them_by_name(page_for, make_user):
    """While searching, the count shows the matching files, open-webui/open-webui#32815."""
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
    """Shift on a file's Delete skips the confirmation, open-webui/open-webui#32814."""
    account = make_user()
    file_id = upload(account, "old-chart.txt", "an old chart")
    page = page_for(account)
    files = open_manage_files(page)
    expect(files.get_by_text("old-chart.txt", exact=True)).to_be_visible()

    files.get_by_role("button", name="Actions for old-chart.txt").click()
    page.get_by_role("menu").get_by_role("button", name="Delete").click(modifiers=["Shift"])

    expect(page.get_by_text("1 deleted, 0 failed, 0 not processed.")).to_be_visible()
    expect(page.get_by_role("dialog").filter(has_text="Delete 1 file?")).to_have_count(0)
    expect(files.get_by_text("old-chart.txt")).to_have_count(0)
    with account.client() as client:
        assert client.get(f"/api/v1/files/{file_id}").status_code == 404


def test_an_admin_sees_only_their_own_files_in_manage_files(page_for, make_user):
    admin, other = make_user(role="admin"), make_user()
    upload(admin, "admin-logbook.txt", "the admin's own")
    upload(other, "crew-roster.txt", "not the admin's")
    page = page_for(admin)
    files = open_manage_files(page)

    expect(files.get_by_text("admin-logbook.txt", exact=True)).to_be_visible()
    expect(files.get_by_text("crew-roster.txt")).to_have_count(0)
    expect(files.get_by_text("1", exact=True)).to_be_visible()


def selection_action(page: Page, files: Locator, action: str) -> None:
    files.get_by_role("button", name="Selection actions").click()
    page.get_by_role("menu").get_by_role("button", name=action).click()


def confirm_delete(page: Page, count: int) -> None:
    confirming = page.get_by_role("dialog").filter(has_text=f"Delete {count} files?")
    confirming.get_by_role("button", name="Delete", exact=True).click()


def own_file_names(account) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/files/user/search", params={"limit": 1000})
    assert listed.status_code == 200, listed.text
    return sorted(file["filename"] for file in listed.json())


def test_deleting_picked_files_counts_what_was_deleted_and_what_failed(page_for, make_user):
    account = make_user()
    ids = {name: upload(account, name, name) for name in ("keel.txt", "mast.txt", "sail.txt")}
    upload(account, "rudder.txt", "kept")
    page = page_for(account)
    files = open_manage_files(page)
    for name in ids:
        files.get_by_role("checkbox", name=f"Select {name}").click()
    expect(files.get_by_text("3 selected")).to_be_visible()
    # gone before the page deletes it, so its delete fails
    with account.client() as client:
        assert client.delete(f"/api/v1/files/{ids['sail.txt']}").status_code == 200

    selection_action(page, files, "Delete")
    confirm_delete(page, 3)

    expect(page.get_by_text("2 deleted, 1 failed, 0 not processed.")).to_be_visible()
    expect(files.get_by_text("rudder.txt", exact=True)).to_be_visible()
    expect(files.get_by_text("keel.txt")).to_have_count(0)
    assert own_file_names(account) == ["rudder.txt"]


def test_select_all_matching_takes_only_the_files_matching_the_search(page_for, make_user):
    account = make_user()
    # one more than the 50 a page loads, so Select all matching is offered
    for index in range(51):
        upload(account, f"buoy-{index:02}.txt", "a buoy")
    for index in range(2):
        upload(account, f"anchor-{index}.txt", "an anchor")
    page = page_for(account)
    files = open_manage_files(page)
    expect(files.get_by_text("53", exact=True)).to_be_visible()

    files.get_by_placeholder("Search Files").fill("buoy")
    # while searching the count is the matches loaded so far, a page of 50
    expect(files.get_by_text("50", exact=True)).to_be_visible()
    expect(files.get_by_text("anchor-")).to_have_count(0)
    files.get_by_role("checkbox", name="Select loaded files").click()
    expect(files.get_by_text("50 selected")).to_be_visible()
    files.get_by_role("button", name="Select all matching files").click()
    expect(files.get_by_text("51 selected")).to_be_visible()
    selection_action(page, files, "Delete")
    confirm_delete(page, 51)

    expect(page.get_by_text("51 deleted, 0 failed, 0 not processed.")).to_be_visible()
    assert own_file_names(account) == ["anchor-0.txt", "anchor-1.txt"]


def test_downloading_picked_files_zips_exactly_those_files(page_for, make_user):
    account = make_user()
    texts = {"first log": "log.txt", "second log": "log.txt", "the chart": "chart.txt"}
    for text, name in texts.items():
        upload(account, name, text)
    upload(account, "skip-me.txt", "not picked")
    page = page_for(account)
    files = open_manage_files(page)
    files.get_by_role("checkbox", name="Select loaded files").click()
    files.get_by_role("checkbox", name="Select skip-me.txt").click()
    expect(files.get_by_text("3 selected")).to_be_visible()

    with page.expect_download() as downloaded:
        selection_action(page, files, "Download")

    assert downloaded.value.suggested_filename == "files.zip"
    zipped = pathlib.Path(downloaded.value.path()).read_bytes()
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        names = archive.namelist()
        contents = sorted(archive.read(name).decode() for name in names)
    assert contents == sorted(texts), f"the zip held {names} with {contents}"
    assert "chart.txt" in names and len(set(names)) == 3, f"the zip's names were {names}"


def test_a_single_downloaded_file_keeps_its_name(page_for, make_user):
    account = make_user()
    upload(account, "tide table 2026.txt", "high water at noon")
    page = page_for(account)
    files = open_manage_files(page)

    files.get_by_role("button", name="Actions for tide table 2026.txt").click()
    with page.expect_download() as downloaded:
        page.get_by_role("menu").get_by_role("button", name="Download").click()

    assert downloaded.value.suggested_filename == "tide table 2026.txt"
    assert pathlib.Path(downloaded.value.path()).read_bytes() == b"high water at noon"


class HeldRequests:
    """Holds the page's requests that `wanted` picks until `release`, then lets later ones pass."""

    def __init__(self, wanted):
        self.wanted, self.held, self.released = wanted, [], False

    def __call__(self, route: Route) -> None:
        if self.wanted(route.request) and not self.released:
            self.held.append(route)
        else:
            route.continue_()

    def release(self) -> None:
        self.released = True
        for route in self.held:
            route.continue_()


def running_operation(page: Page) -> Locator:
    """Manage Files while an operation runs, when its progress takes the search box's place."""
    return page.get_by_role("dialog", name="Files", exact=True)


def is_file_delete(request) -> bool:
    return request.method == "DELETE" and "/api/v1/files/" in request.url


def test_stop_ends_a_bulk_delete_and_leaves_the_rest(page_for, make_user):
    account = make_user()
    for index in range(6):
        upload(account, f"pier-{index}.txt", "a pier")
    page = page_for(account)
    files = open_manage_files(page)
    files.get_by_role("checkbox", name="Select loaded files").click()
    expect(files.get_by_text("6 selected")).to_be_visible()
    deletes: list[str] = []

    def record_delete(request) -> None:
        if is_file_delete(request):
            deletes.append(request.url)

    page.on("request", record_delete)
    holding = HeldRequests(is_file_delete)
    page.route("**/api/v1/files/*", holding)

    selection_action(page, files, "Delete")
    with page.expect_request(is_file_delete):
        confirm_delete(page, 6)
    running = running_operation(page)
    expect(running.get_by_role("status").filter(has_text="Processed 0 of 6")).to_be_visible()
    running.get_by_role("button", name="Stop", exact=True).click()
    expect(running.get_by_role("button", name="Stopping...")).to_be_visible()
    holding.release()

    report = page.get_by_text(re.compile(r"\d+ deleted, 0 failed, \d+ not processed\."))
    expect(report).to_be_visible()
    counts = re.fullmatch(r"(\d+) deleted, 0 failed, (\d+) not processed\.", report.inner_text())
    assert counts, report.inner_text()
    deleted, remaining = int(counts[1]), int(counts[2])
    assert (deleted, remaining) == (len(deletes), 6 - len(deletes)), report.inner_text()
    assert remaining > 0, "Stop let every file be deleted"
    assert len(own_file_names(account)) == remaining


def test_stop_ends_a_download_without_saving_anything(page_for, make_user):
    account = make_user()
    for name in ("north.txt", "south.txt"):
        upload(account, name, name)
    page = page_for(account)
    files = open_manage_files(page)
    files.get_by_role("checkbox", name="Select loaded files").click()
    expect(files.get_by_text("2 selected")).to_be_visible()
    downloads: list[str] = []

    def record_download(download) -> None:
        downloads.append(download.suggested_filename)

    page.on("download", record_download)

    def is_file_content(request) -> bool:
        return "/content" in request.url and "/api/v1/files/" in request.url

    holding = HeldRequests(is_file_content)
    page.route("**/api/v1/files/**", holding)

    with page.expect_request(is_file_content):
        selection_action(page, files, "Download")
    running = running_operation(page)
    expect(running.get_by_role("status").filter(has_text="Preparing 0 of 2")).to_be_visible()
    running.get_by_role("button", name="Stop", exact=True).click()
    expect(running.get_by_role("button", name="Stopping...")).to_be_visible()
    holding.release()

    expect(files.get_by_placeholder("Search Files")).to_be_visible()
    page.wait_for_timeout(1000)  # a download the stop missed would show up by now
    assert downloads == [], "a stopped download still saved a file"
