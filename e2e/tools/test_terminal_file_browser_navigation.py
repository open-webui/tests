"""Journey: a person finds their way around a terminal's files in the file browser.

A real Open Terminal runs with a folder of the test's own in its home. In a chat with the terminal
picked the person opens Files. Back and Forward step through the folders they visited and are
switched off at either end; a folder in the path above the list leads there, and a folder opened
after going back ends what Forward remembered. A click on the empty end of that path turns it into
a box that takes a path: Enter opens the folder typed, Escape leaves the folder as it was, `..`
opens the folder above, and a path that climbs above the terminal's home with `..` opens the home;
a path with no leading slash opens below the folder shown. A folder's arrow opens it in place, with
its entries under it and the rest of the list around them, and closes it again, as Expand and
Collapse in the folder's More menu do; Refresh also reads the folders that are open. The Sort menu
orders the list by name, size, date modified or date added with folders always first, a second pick
of the same order turns it around, and the two date orders show each entry's age. A right click on
the empty part of the list offers the folder's own actions. Show Hidden Files is still on after the
page was loaded again.

The right-click menu near the lower edge of the window is a known red, see its test.

Discriminates: passes on the dev c15f28649 build, three runs, the lower edge test aside. Each
test fails on two frontend builds of it, A and B, at the step named: Back on offer in the first
folder (A) and Forward doing nothing (B) for the back and forward test; Refresh not reading the
folders open in place (A) and Expand in a folder's menu doing nothing (B) for the arrow test; the
size order sorting by name (A) and a second pick not turning the order around (B) for the sort
test; New Folder in the right-click menu asking for a file's name (A) and its hidden files entry
doing nothing (B) for the right click test. On A alone: the hidden files switch not read back
when the browser opens (reload test) and an entry's menu without Open (menus test). The lower
edge test passes on A, which also keeps the right-click menu inside the window. The typed path
test fails on dev 32243bdf7 (no box to type into) and passes on branch feat/file-browser-type-path
d2d33848c.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import ask
from utils.chat_ui import close_menu
from utils.file_browser import (
    choose_action,
    connected_browser,
    crumb,
    enter,
    entries_of,
    expect_listed,
    file_browser,
    folder_menu,
    refresh,
    rest_pointer,
    row,
    row_menu,
    sort_by,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOUR = 3600
DAY = 24 * HOUR
# Linux dates new files by CLOCK_REALTIME_COARSE, which Python has no name for
DISK_CLOCK = 5


@pytest.fixture
def quay(open_terminal) -> Path:
    """A folder of the test's own in the terminal's home: a file, and a folder holding one."""
    folder = open_terminal.home / f"quay-{uuid.uuid4().hex[:6]}"
    (folder / "sheds").mkdir(parents=True)
    (folder / "sheds" / "ledger.txt").write_text("two ropes\n")
    (folder / "nets.txt").write_text("mended\n")
    return folder


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def browser(quay, page_for, person, admin, preserve, open_terminal) -> Locator:
    """The File browser of a chat with the Open Terminal picked, showing its home."""
    preserve(TERMINAL_SERVERS_CONFIG)
    region = connected_browser(page_for, admin, person, open_terminal)
    expect(region.get_by_text(quay.name, exact=True)).to_be_visible()
    return region


def _back(browser: Locator) -> Locator:
    return browser.get_by_role("button", name="Back", exact=True)


def _forward(browser: Locator) -> Locator:
    return browser.get_by_role("button", name="Forward", exact=True)


def test_back_and_forward_step_through_the_folders_visited(browser, quay):
    expect(_back(browser)).to_be_disabled()
    expect(_forward(browser)).to_be_disabled()

    browser.get_by_text(quay.name, exact=True).click()
    expect_listed(browser, "sheds/", "nets.txt")
    browser.get_by_text("sheds", exact=True).click()
    expect_listed(browser, "ledger.txt")
    expect(_forward(browser)).to_be_disabled()

    _back(browser).click()
    expect_listed(browser, "sheds/", "nets.txt")
    _back(browser).click()
    expect(row(browser, quay.name)).to_be_visible()
    expect(row(browser, "nets.txt")).to_have_count(0)
    expect(_back(browser), "Back at the first folder visited").to_be_disabled()

    _forward(browser).click()
    expect_listed(browser, "sheds/", "nets.txt")
    _forward(browser).click()
    expect_listed(browser, "ledger.txt")
    expect(_forward(browser), "Forward at the last folder visited").to_be_disabled()

    # a folder in the path above the list
    crumb(browser, quay.name).click()
    expect_listed(browser, "sheds/", "nets.txt")
    _back(browser).click()
    expect_listed(browser, "ledger.txt")
    _back(browser).click()
    expect_listed(browser, "sheds/", "nets.txt")
    expect(_forward(browser)).to_be_enabled()
    (quay / "hold").mkdir()
    enter_empty = browser.get_by_text("hold", exact=True)
    refresh(browser)
    enter_empty.click()
    expect(browser.get_by_text("This folder is empty")).to_be_visible()
    expect(_forward(browser), "Forward after a new folder was opened").to_be_disabled()


def _type_path(browser: Locator, shown: str) -> Locator:
    """Click the empty end of the path above the list, which shows `shown` last; returns the box."""
    trail = crumb(browser, shown).locator("xpath=..")
    width = trail.bounding_box()["width"]
    trail.click(position={"x": width - 2, "y": 4})
    box = browser.get_by_role("textbox", name="Path", exact=True)
    expect(box).to_be_focused()
    return box


def test_a_path_typed_into_the_path_above_the_list_opens_that_folder(browser, quay, open_terminal):
    """Red on dev 32243bdf7: the path above the list takes no typing. Pins the feature PR that
    lets a person type a path there, branch Classic298:feat/file-browser-type-path (d2d33848c),
    open-webui/open-webui#TBD."""
    enter(browser, quay.name, "nets.txt")

    path = _type_path(browser, quay.name)
    expect(path, "the box starts with the folder shown").to_have_value(f"{quay}/")
    path.fill(str(quay / "sheds"))
    path.press("Enter")
    expect_listed(browser, "ledger.txt")
    expect(crumb(browser, "sheds")).to_be_visible()

    path = _type_path(browser, "sheds")
    path.fill(str(quay))
    path.press("Escape")
    expect(crumb(browser, "sheds"), "Escape left the folder shown").to_be_visible()
    expect_listed(browser, "ledger.txt")

    path = _type_path(browser, "sheds")
    path.fill("..")
    path.press("Enter")
    expect_listed(browser, "sheds/", "nets.txt")
    expect(crumb(browser, ".."), "the path shows the folder `..` led to").to_have_count(0)

    path = _type_path(browser, quay.name)
    path.fill(str(quay / "nowhere"))
    path.press("Enter")
    expect(browser.get_by_text("Failed to load directory.", exact=False)).to_be_visible()

    # the folders in the path still lead there
    crumb(browser, quay.name).click()
    expect_listed(browser, "sheds/", "nets.txt")

    # the browser is rooted at the terminal's home, and `..` does not climb out of it
    above = open_terminal.home.parent / f"above-{uuid.uuid4().hex[:6]}"
    above.mkdir()
    (above / "deeds.txt").write_text("not for the browser\n")
    path = _type_path(browser, quay.name)
    path.fill(f"{open_terminal.home}/../{above.name}")
    path.press("Enter")
    expect(row(browser, quay.name), "a path above the home opens the home").to_be_visible()
    expect(row(browser, "deeds.txt")).to_have_count(0)

    # a path with no leading slash opens below the folder shown
    path = _type_path(browser, "Home")
    path.fill(f"{quay.name}/sheds")
    path.press("Enter")
    expect_listed(browser, "ledger.txt")
    expect(crumb(browser, quay.name), "the path shows the folder opened").to_be_visible()


def _arrow(browser: Locator, folder: str, name: str) -> Locator:
    return row(browser, folder).get_by_role("button", name=name, exact=True)


def test_a_folder_opens_and_closes_in_place_from_its_arrow_and_its_menu(browser, quay):
    enter(browser, quay.name, "nets.txt")
    expect_listed(browser, "sheds/", "nets.txt")

    _arrow(browser, "sheds", "Expand").click()
    expect_listed(browser, "sheds/", "ledger.txt", "nets.txt")
    # the tab still shows the folder around it
    expect(crumb(browser, quay.name)).to_be_visible()
    expect(crumb(browser, "sheds")).to_have_count(0)

    (quay / "sheds" / "anchor.txt").write_text("iron\n")
    (quay / "buoys.txt").write_text("three\n")
    refresh(browser)
    expect_listed(browser, "sheds/", "anchor.txt", "ledger.txt", "buoys.txt", "nets.txt")

    _arrow(browser, "sheds", "Collapse").click()
    expect_listed(browser, "sheds/", "buoys.txt", "nets.txt")

    row_menu(browser, "sheds").get_by_role("menuitem", name="Expand", exact=True).click()
    expect_listed(browser, "sheds/", "anchor.txt", "ledger.txt", "buoys.txt", "nets.txt")
    row_menu(browser, "sheds").get_by_role("menuitem", name="Collapse", exact=True).click()
    expect_listed(browser, "sheds/", "buoys.txt", "nets.txt")


def test_the_sort_menu_orders_the_list_and_a_second_pick_turns_the_order_around(
    browser, quay, person, open_terminal
):
    now = time.time()
    # name, size, age and the order they were made in each give another order
    for name, size, age in (
        ("buoy.txt", 100, 3 * HOUR + 600),
        ("chain.txt", 200, 5 * DAY + HOUR),
        ("anchor.txt", 300, 2 * DAY + HOUR),
    ):
        _wait_for_the_disks_clock()
        (quay / name).write_bytes(b"x" * size)
        os.utime(quay / name, (now - age, now - age))
    (quay / "nets.txt").unlink()
    enter(browser, quay.name, "chain.txt")
    expect_listed(browser, "sheds/", "anchor.txt", "buoy.txt", "chain.txt")
    expect(row(browser, "buoy.txt"), "no age while sorted by name").not_to_contain_text("ago")

    sort_by(browser, "Name")
    expect_listed(browser, "sheds/", "chain.txt", "buoy.txt", "anchor.txt")

    sort_by(browser, "Size")
    expect_listed(browser, "sheds/", "buoy.txt", "chain.txt", "anchor.txt")
    sort_by(browser, "Size")
    expect_listed(browser, "sheds/", "anchor.txt", "chain.txt", "buoy.txt")

    sort_by(browser, "Date Modified")
    expect_listed(browser, "sheds/", "chain.txt", "anchor.txt", "buoy.txt")
    expect(row(browser, "buoy.txt")).to_contain_text("3h ago")
    expect(row(browser, "anchor.txt")).to_contain_text("2d ago")
    expect(row(browser, "chain.txt")).to_contain_text("5d ago")
    sort_by(browser, "Date Modified")
    expect_listed(browser, "sheds/", "buoy.txt", "anchor.txt", "chain.txt")

    with person.client() as client:
        told = client.get(
            f"/api/v1/terminals/{_terminal_id(client)}/files/list",
            params={"directory": str(quay)},
        )
    assert told.status_code == 200, told.text
    added = {entry["name"]: entry["created"] for entry in told.json()["entries"]}
    files = ["anchor.txt", "buoy.txt", "chain.txt"]
    # the date a file was added comes from the disk, which may not keep one
    assert len({added[name] for name in files}) == 3, (
        f"the terminal does not tell the three files apart by when they were added: {added}"
    )
    sort_by(browser, "Date Added")
    expect_listed(browser, "sheds/", "buoy.txt", "chain.txt", "anchor.txt")
    expect(row(browser, "anchor.txt")).to_contain_text("just now")
    sort_by(browser, "Date Added")
    expect_listed(browser, "sheds/", "anchor.txt", "chain.txt", "buoy.txt")


def _wait_for_the_disks_clock() -> None:
    """Return once the clock the disk dates new files by has moved on to its next second."""
    # two files made within one tick of that clock carry the same date, and on a CI runner the
    # terminal reports that date in whole seconds
    second = int(time.clock_gettime(DISK_CLOCK))
    deadline = time.monotonic() + 5
    while int(time.clock_gettime(DISK_CLOCK)) == second:
        assert time.monotonic() < deadline, "the disk's clock stands still"
        time.sleep(0.01)


def _terminal_id(client) -> str:
    terminals = client.get("/api/v1/terminals/").json()
    assert len(terminals) == 1, f"the person should have the one terminal: {terminals}"
    return terminals[0]["id"]


def test_a_right_click_on_the_empty_part_of_the_list_offers_the_folders_actions(browser, quay):
    (quay / ".tide-table").write_text("high at six\n")
    enter(browser, quay.name, "nets.txt")

    menu = folder_menu(browser)
    assert menu.get_by_role("button").all_inner_texts() == [
        "Refresh",
        "New File",
        "New Folder",
        "Upload",
        "Upload Folder",
        "Download",
        "Show Hidden Files",
    ], "the entries of the menu a right click on the list opens"
    menu.get_by_role("button", name="New Folder", exact=True).click()
    expect(menu).to_have_count(0)
    browser.get_by_placeholder("Folder name").fill("charts")
    browser.get_by_placeholder("Folder name").press("Enter")
    expect(row(browser, "charts")).to_be_visible()
    assert (quay / "charts").is_dir(), "New Folder from the right click made no folder"

    folder_menu(browser).get_by_role("button", name="New File", exact=True).click()
    browser.get_by_placeholder("File name").fill("tides.md")
    browser.get_by_placeholder("File name").press("Enter")
    expect(row(browser, "tides.md")).to_be_visible()
    assert (quay / "tides.md").is_file(), "New File from the right click made no file"

    expect(row(browser, ".tide-table")).to_have_count(0)
    folder_menu(browser).get_by_role("button", name="Show Hidden Files", exact=True).click()
    expect(row(browser, ".tide-table")).to_be_visible()
    folder_menu(browser).get_by_role("button", name="Hide Hidden Files", exact=True).click()
    expect(row(browser, "nets.txt")).to_be_visible()
    expect(row(browser, ".tide-table")).to_have_count(0)

    # with the page's live updates the new file would show by itself in time; Refresh is at once
    (quay / "lamp.txt").write_text("oil\n")
    folder_menu(browser).get_by_role("button", name="Refresh", exact=True).click()
    expect(row(browser, "lamp.txt")).to_be_visible()


def test_a_right_click_near_the_lower_edge_shows_the_whole_menu(browser, quay):
    """Red on dev c15f28649 (same menu code in 0.13.3): the menu is drawn from the pointer
    downwards and to the right with no regard for the window, so near the lower edge its entries
    are outside it (open-webui/open-webui#34097)."""
    enter(browser, quay.name, "nets.txt")
    rest_pointer(browser)
    listing = browser.locator("[data-file-list]").locator("xpath=..").bounding_box()
    assert listing, "the list has no place on the page"

    browser.page.mouse.click(
        listing["x"] + listing["width"] / 2, listing["y"] + listing["height"] - 40, button="right"
    )

    entries = (
        browser.get_by_role("button", name="Upload Folder", exact=True)
        .locator("xpath=..")
        .get_by_role("button")
    )
    expect(entries).to_have_count(7)
    for index in range(7):
        entry = entries.nth(index)
        expect(
            entry, f"the entry {entry.inner_text()!r} of the right-click menu"
        ).to_be_in_viewport(ratio=1)


def test_hidden_files_are_still_shown_after_the_page_is_loaded_again(
    browser, quay, upstream, config
):
    (quay / ".tide-table").write_text("high at six\n")
    page = browser.page
    # a saved chat keeps its terminal and its tabs over a reload
    page.keyboard.press("Control+Backslash")
    ask(page, upstream, f"note the tides {uuid.uuid4().hex[:6]}", reply.text("Noted."))
    enter(file_browser(page), quay.name, "nets.txt")
    expect(row(browser, ".tide-table")).to_have_count(0)
    choose_action(browser, "Show Hidden Files")
    expect(row(browser, ".tide-table")).to_be_visible()

    page.reload()

    reloaded = file_browser(page)
    # a reload is a cold load, which page_for would otherwise wait out
    expect(row(reloaded, "nets.txt")).to_be_visible(timeout=config.navigation_timeout)
    expect(row(reloaded, ".tide-table"), "the hidden file after the reload").to_be_visible()
    choose_action(reloaded, "Hide Hidden Files")
    expect(row(reloaded, "nets.txt")).to_be_visible()
    expect(row(reloaded, ".tide-table")).to_have_count(0)


def test_the_menus_of_a_file_and_of_a_folder_offer_what_can_be_done_with_each(browser, quay):
    enter(browser, quay.name, "nets.txt")

    menu = row_menu(browser, "nets.txt")
    assert entries_of(menu) == [
        "Open",
        "Select for Compare",
        "Download",
        "Copy Path",
        "Rename",
        "Delete",
    ], "the entries of a file's More menu"
    close_menu(browser.page)

    menu = row_menu(browser, "sheds")
    assert entries_of(menu) == [
        "New Folder",
        "New File",
        "Open Folder",
        "Expand",
        "Copy Path",
        "Rename",
        "Delete",
    ], "the entries of a folder's More menu"
    menu.get_by_role("menuitem", name="Open Folder", exact=True).click()
    expect_listed(browser, "ledger.txt")
