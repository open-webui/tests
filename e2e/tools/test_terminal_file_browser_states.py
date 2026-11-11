"""Journey: what the file browser shows while a terminal is slow, refuses or is taken away.

A fake terminal answers what the file browser asks, and the test decides how. A folder the
terminal refuses to list (401, 403 or 500, each seen by the page as that status) shows "Failed to
load directory" with the hint to check the connection, and lists again on Refresh once the
terminal answers. A folder whose arrow cannot open it says "Failed to load folder" and stays
closed. While a listing is on its way the browser claims nothing about the folder, neither
entries nor "This folder is empty", and an answer that arrives after the person went elsewhere is
dropped. A search shows "No matches", "Failed to search files" when the terminal refuses it,
loads further results as the end of the list comes into view and offers Retry when that fails.
New Folder, New File, Upload, Delete and Download each say that they failed when the terminal
refuses them, and a selection the terminal deletes only in part says how many of it went. The
folder is read again when the browser tab is shown again. A person whose access to the terminal
is taken away while Files is open sees "Terminal connection unavailable" in its place.

Discriminates: passes on the dev c15f28649 build, three runs. Each test fails on a frontend build
of it (A or B) at the step named. On A: a failed listing read as an empty folder (refused listing
test), a folder that cannot be opened in place saying nothing (arrow test), a late listing no
longer dropped (slow test), a failed search read as no matches (search test), a folder that was
not made reported as made (refused action test), every entry of a selection counted as deleted
(partial delete test), nothing read again when the tab is shown again (shown again test) and a
tab whose terminal is gone drawn as before (lost access test). On B: the old rows staying while
a folder loads (slow test), further results that fail offering no Retry (search test) and a
failed download saying nothing (refused action test). The messages of a refused New File, Upload
and Delete have no mutation of their own.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from utils.cached_chat import pick_terminal
from utils.file_browser import (
    LOAD_FAILED,
    choose_action,
    crumb,
    expect_listed,
    file_browser,
    open_actions,
    refresh,
    row,
    row_menu,
    select,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOME = "/home/user/"
LOGS = "/home/user/logs/"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
EMPTY = "This folder is empty"
REFUSAL = {"detail": "the disk went away"}


def _file(name: str) -> dict:
    return {"name": name, "type": "file", "size": 16, "modified": 1_800_000_000}


def _folder(name: str) -> dict:
    return {"name": name, "type": "directory", "modified": 1_800_000_000}


@dataclass
class Disk:
    """What the fake terminal holds, and what it does when asked."""

    folders: dict[str, list[dict]] = field(default_factory=dict)
    # a folder's path, an action or a file's name to the status it is refused with
    refused: dict[str, int] = field(default_factory=dict)
    # a folder whose listing waits for the event before it is answered
    held: dict[str, threading.Event] = field(default_factory=dict)
    matches: dict[int, dict | int] = field(default_factory=dict)
    searched: list[int] = field(default_factory=list)

    def listing(self, request):
        directory = _asked(request, "directory", HOME)
        if directory in self.held:
            # bounded, so a test that fails early does not keep the terminal's thread
            self.held[directory].wait(60)
        if directory in self.refused:
            return json_answer(REFUSAL, self.refused[directory])
        return json_answer({"entries": self.folders.get(directory, []), "writable": True})

    def page_of_matches(self, request):
        offset = int(_asked(request, "offset", "0"))
        self.searched.append(offset)
        answer = self.matches.get(offset, {"results": [], "next_offset": None})
        return json_answer(REFUSAL, answer) if isinstance(answer, int) else json_answer(answer)

    def refusing(self, route: str, done):
        """A handler for `route` that does `done(request)` unless the route is refused."""

        def handle(request):
            if route in self.refused:
                return json_answer(REFUSAL, self.refused[route])
            return json_answer(done(request))

        return handle

    def delete(self, request):
        folder, name = _asked(request, "path", "").rsplit("/", 1)
        if name in self.refused:
            return json_answer(REFUSAL, self.refused[name])
        self.folders[f"{folder}/"] = [
            entry for entry in self.folders[f"{folder}/"] if entry["name"] != name
        ]
        return json_answer({"path": f"{folder}/{name}", "type": "file"})


def _asked(request, name: str, default: str) -> str:
    return parse_qs(urlparse(request.path).query).get(name, [default])[0]


@pytest.fixture
def terminal():
    """A fake terminal with a home holding a folder and two files; `disk` scripts it."""
    with serving_terminal() as server:
        disk = Disk(
            folders={
                HOME: [_folder("logs"), _file("nets.txt"), _file("oars.txt")],
                LOGS: [_file("tide.txt")],
            }
        )
        server.disk = disk
        server.route("GET", "/openapi.json", json_answer(SPEC))
        # no live updates: what the list shows is what the page asked for
        server.route("GET", "/api/config", json_answer({"features": {}}))
        server.route(
            "GET",
            "/files/cwd",
            json_answer({"cwd": HOME, "home": HOME, "root": {"path": HOME, "label": "user"}}),
        )
        server.route("POST", "/files/cwd", json_answer({"cwd": HOME}))
        server.route("GET", "/files/list", disk.listing)
        server.route("GET", "/files/matches", disk.page_of_matches)
        server.route("GET", "/skills", json_answer([]))
        server.route("POST", "/files/mkdir", disk.refusing("mkdir", lambda request: {"path": ""}))
        server.route("POST", "/files/upload", disk.refusing("upload", lambda request: {"size": 0}))
        server.route("GET", "/files/view", disk.refusing("view", lambda request: {}))
        server.route("POST", "/files/archive", disk.refusing("archive", lambda request: {}))
        server.route("DELETE", "/files/delete", disk.delete)
        yield server


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def team(admin, person) -> str:
    """A group holding the person, through which the terminal is shared with them."""
    return make_group(admin, [person])


@pytest.fixture
def browser(page_for, admin, preserve, terminal, person, team) -> Locator:
    """The File browser of the person's chat with the fake terminal picked, showing its home."""
    preserve(TERMINAL_SERVERS_CONFIG)
    name = f"Lab box {uuid.uuid4().hex[:6]}"
    through_the_team = {"principal_type": "group", "principal_id": team, "permission": "read"}
    connection = terminal.connection(name=name, config={"access_grants": [through_the_team]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, name)
    region = file_browser(page)
    expect_listed(region, "logs/", "nets.txt", "oars.txt")
    return region


def _listing_of(page: Page, folder: str):
    """Waits for the page to get its answer to a listing of `folder`."""
    return page.expect_response(
        lambda response: (
            "/files/list" in response.url
            and parse_qs(urlparse(response.url).query).get("directory") == [folder]
        )
    )


def test_a_folder_the_terminal_refuses_to_list_says_so_and_lists_again_on_refresh(
    browser, terminal
):
    disk = terminal.disk
    for status in (401, 403, 500):
        disk.refused[HOME] = status
        with browser.page.expect_response(lambda response: "/files/list" in response.url) as told:
            refresh(browser)
        assert told.value.status == status, f"the page got {told.value.status} for a {status}"
        expect(browser.get_by_text(LOAD_FAILED), f"after a {status}").to_be_visible()
        expect(browser.locator("[data-file-row]")).to_have_count(0)
        expect(browser.get_by_text(EMPTY)).to_have_count(0)

        del disk.refused[HOME]
        refresh(browser)
        expect_listed(browser, "logs/", "nets.txt", "oars.txt")
        expect(browser.get_by_text(LOAD_FAILED)).to_have_count(0)

    disk.refused[LOGS] = 500
    browser.get_by_text("logs", exact=True).click()
    expect(browser.get_by_text(LOAD_FAILED), "a folder opened from the list").to_be_visible()
    expect(crumb(browser, "logs")).to_be_visible()
    browser.get_by_role("button", name="Back", exact=True).click()
    expect_listed(browser, "logs/", "nets.txt", "oars.txt")


def test_a_folder_whose_arrow_cannot_open_it_says_so_and_stays_closed(browser, terminal):
    disk = terminal.disk
    disk.refused[LOGS] = 500
    logs = row(browser, "logs")

    logs.get_by_role("button", name="Expand", exact=True).click()

    expect(browser.page.get_by_text("Failed to load folder")).to_be_visible()
    expect(logs.get_by_role("button", name="Expand", exact=True)).to_be_visible()
    expect_listed(browser, "logs/", "nets.txt", "oars.txt")

    del disk.refused[LOGS]
    logs.get_by_role("button", name="Expand", exact=True).click()
    expect_listed(browser, "logs/", "tide.txt", "nets.txt", "oars.txt")
    expect(logs.get_by_role("button", name="Collapse", exact=True)).to_be_visible()


def test_a_slow_listing_claims_nothing_until_it_arrives_and_a_late_answer_is_dropped(
    browser, terminal
):
    disk = terminal.disk
    page = browser.page
    arrives = disk.held[LOGS] = threading.Event()

    with _listing_of(page, LOGS) as late:
        browser.get_by_text("logs", exact=True).click()
        expect(crumb(browser, "logs")).to_be_visible()
        expect(browser.locator("[data-file-row]"), "rows while the list loads").to_have_count(0)
        expect(browser.get_by_text(EMPTY), "while the list loads").to_have_count(0)
        expect(browser.get_by_text(LOAD_FAILED), "while the list loads").to_have_count(0)

        # the person does not wait and goes back to the folder above
        crumb(browser, "user").click()
        expect_listed(browser, "logs/", "nets.txt", "oars.txt")
        arrives.set()
    late.value.finished()
    # nothing should change any more: give the page the time to mishandle the late answer
    page.wait_for_timeout(500)
    expect_listed(browser, "logs/", "nets.txt", "oars.txt")
    expect(crumb(browser, "logs"), "the folder of the late answer").to_have_count(0)

    del disk.held[LOGS]
    browser.get_by_text("logs", exact=True).click()
    expect_listed(browser, "tide.txt")


def _match(name: str) -> dict:
    return {
        "path": f"{HOME}{name}",
        "relative_path": name,
        "name": name,
        "type": "file",
        "name_match": True,
        "content_matches": [],
    }


def test_a_search_says_when_nothing_matches_or_it_failed_and_loads_more_results(browser, terminal):
    disk = terminal.disk
    browser.get_by_role("button", name="Search", exact=True).click()
    search = browser.get_by_placeholder("Search files and contents")

    search.fill("kelp")
    expect(browser.get_by_text("No matches", exact=True)).to_be_visible()
    expect(browser.locator("[data-file-row]")).to_have_count(0)

    disk.matches[0] = 500
    search.fill("rope")
    expect(browser.get_by_text("Failed to search files", exact=True)).to_be_visible()
    expect(browser.get_by_text("No matches", exact=True)).to_have_count(0)

    disk.matches[0] = {"results": [_match("tide-1.txt"), _match("tide-2.txt")], "next_offset": 2}
    disk.matches[2] = 500
    disk.searched.clear()
    search.fill("tide")
    expect(browser.get_by_text("tide-2.txt", exact=True)).to_be_visible()
    retry = browser.get_by_role("button", name="Retry", exact=True)
    expect(retry).to_be_visible()
    expect(browser.get_by_text("tide-3.txt", exact=True)).to_have_count(0)

    disk.matches[2] = {"results": [_match("tide-3.txt")], "next_offset": None}
    retry.click()
    expect(browser.get_by_text("tide-3.txt", exact=True)).to_be_visible()
    expect(browser.get_by_text("tide-1.txt", exact=True)).to_be_visible()
    expect(retry).to_have_count(0)
    assert disk.searched == [0, 2, 2], (
        f"the search should ask for the first results, then twice for the next: {disk.searched}"
    )

    search.press("Escape")
    expect(search).to_have_count(0)
    expect_listed(browser, "logs/", "nets.txt", "oars.txt")


def test_an_action_the_terminal_refuses_says_that_it_failed(browser, terminal):
    disk = terminal.disk
    page = browser.page
    disk.refused.update({"mkdir": 500, "upload": 500, "nets.txt": 500, "view": 500, "archive": 500})

    choose_action(browser, "New Folder")
    browser.get_by_placeholder("Folder name").fill("charts")
    browser.get_by_placeholder("Folder name").press("Enter")
    expect(page.get_by_text("Failed to create folder", exact=True)).to_be_visible()

    choose_action(browser, "New File")
    browser.get_by_placeholder("File name").fill("tides.md")
    browser.get_by_placeholder("File name").press("Enter")
    expect(page.get_by_text("Failed to create file", exact=True)).to_be_visible()

    menu = open_actions(browser)
    with page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload", exact=True).click()
    chooser.value.set_files({"name": "ledger.txt", "mimeType": "text/plain", "buffer": b"kegs\n"})
    expect(page.get_by_text("Upload failed: ledger.txt", exact=True)).to_be_visible()

    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Delete", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.get_by_text("Failed to delete nets.txt", exact=True)).to_be_visible()

    row_menu(browser, "oars.txt").get_by_role("menuitem", name="Download", exact=True).click()
    expect(page.get_by_text("Download failed", exact=True)).to_be_visible()
    expect(page.get_by_text("Download failed", exact=True)).to_have_count(0, timeout=15_000)
    choose_action(browser, "Download")
    expect(page.get_by_text("Download failed", exact=True), "for the folder").to_be_visible()

    expect_listed(browser, "logs/", "nets.txt", "oars.txt")
    refused = {
        (request.method, request.path.split("?")[0])
        for request in terminal.received
        if request.method != "GET" or "/files/view" in request.path
    }
    assert refused == {
        ("POST", "/files/mkdir"),
        ("POST", "/files/upload"),
        ("DELETE", "/files/delete"),
        ("GET", "/files/view"),
        ("POST", "/files/archive"),
    }, f"each action should have been asked of the terminal: {refused}"

    # the same actions report success once the terminal takes them
    disk.refused.clear()
    row_menu(browser, "nets.txt").get_by_role("menuitem", name="Delete", exact=True).click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm", exact=True).click()
    expect(page.get_by_text("nets.txt deleted", exact=True)).to_be_visible()
    expect_listed(browser, "logs/", "oars.txt")


def test_a_selection_deleted_only_in_part_says_how_many_went(browser, terminal):
    terminal.disk.refused["oars.txt"] = 500
    page = browser.page
    select(browser, "nets.txt")
    select(browser, "oars.txt")
    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()

    browser.get_by_role("button", name="Delete", exact=True).click()
    question = page.get_by_role("dialog")
    expect(question).to_contain_text('Are you sure you want to delete "2 items"?')
    question.get_by_role("button", name="Confirm", exact=True).click()

    expect(page.get_by_text("Deleted 1 of 2 items", exact=True)).to_be_visible()
    expect_listed(browser, "logs/", "oars.txt")
    expect(browser.get_by_text("selected")).to_have_count(0)


SHOW_THE_TAB_AS = """state => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
    document.dispatchEvent(new Event('visibilitychange'));
}"""


def test_the_folder_is_read_again_when_the_browser_tab_is_shown_again(browser, terminal):
    disk = terminal.disk
    page = browser.page
    # a headless browser never hides its page, so the page is told what a hidden tab is told
    page.evaluate(SHOW_THE_TAB_AS, "hidden")
    disk.folders[HOME] = [*disk.folders[HOME], _file("pots.txt")]
    asked_before = len(terminal.requests_to("/files/list"))

    page.evaluate(SHOW_THE_TAB_AS, "visible")

    expect_listed(browser, "logs/", "nets.txt", "oars.txt", "pots.txt")
    asked = [
        _asked(request, "directory", "")
        for request in terminal.requests_to("/files/list")[asked_before:]
    ]
    assert asked.count(HOME) == 1, f"the open folder should be read once more: {asked}"


def test_files_says_the_terminal_is_unavailable_once_the_person_loses_it(
    browser, admin, person, team
):
    page = browser.page

    # a change to the admin's connections reaches no open page; leaving the group does
    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{team}/users/remove", json={"user_ids": [person.id]}
        )
    assert removed.status_code == 200, removed.text

    expect(
        page.get_by_role("status").filter(has_text="Terminal connection unavailable")
    ).to_be_visible()
    expect(file_browser(page)).to_have_count(0)
