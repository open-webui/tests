"""Journey: a person creates, renames, moves, deletes and reveals files in the file browser.

A real Open Terminal runs with a few files in its home. In a chat the person picks the terminal
from the input's Terminal menu, opens Files and works in the File browser: New Folder and New File
in its Actions menu make them in the terminal's home, Rename and Delete in a row's More menu
rename the file or remove it after a confirm, a file dragged onto a folder moves into it, and Show
Hidden Files lists a dotfile the browser hides by default. Every change is read back from the
terminal's disk. Select for Compare on one file and Compare with Selected on another open both
files side by side, read only, each under its name, and Swap sides puts the second file on the
left. Grid view in the Actions menu shows every entry as a tile named by the file, a folder with
the count of its items; the choice is still in place after a reload, and List view brings the rows
back. In a Git repository each changed file carries a marker naming its change (modified, added
and staged, untracked) with its added and removed line counts, or its status letter for a binary
file, a folder holding a change says so (a deleted file inside ranking above a modified one), and
an unchanged file carries none (a47f44e55). Since d61a91700 the code editor marks the lines of a
file added or modified since the last commit, also while the person types.

Discriminates: passes on dev 30f3f6a8f; on a build whose New Folder, New File, Rename and Delete
handlers send nothing and whose hidden-files switch does not change the listing, every test here
fails. The compare test passes on the dev 3b8205505 build and fails in a build of it whose Swap
sides leaves the files where they were. The move, grid and Git tests pass on the dev f659c0d12
build (three runs) and each fails on one frontend build of it with these mutations: dropping on a
folder moving nothing, the view switch doing nothing, folders above a change left unmarked, and
file change events no longer refreshing the Git status.
The rename test also pins PR open-webui/open-webui#33786 (fix 17579c522): the field losing focus
right after Enter renamed the old path again and showed an error. It passes on dev e8c0ad85e and
fails in a frontend copy with 17579c522 reverted (two moves are sent).
The tests added for dev e8c0ad85e pass on its build and each fails on a copy of the dev e8c0ad85e
build with these mutations: a move not reloading the other Files tabs of the terminal (two tabs
test), a held drag never opening the tab (held file test), the selection box selecting no row (box
test), dragged rows carrying no attachment (both attach tests), the line marks renamed (both editor
mark tests), and deleted files not marking their folders with binary files showing line counts
(marker test).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.backends import wait_until
from harness.open_terminal import HOME_TAB
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import open_files_tab, pick_terminal
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def home(open_terminal):
    """The terminal's home, with the files the tests act on."""
    folder = open_terminal.home
    (folder / "draft-letter.txt").write_text("Dear harbour master,\n")
    (folder / "old-receipt.txt").write_text("two ropes, one lamp\n")
    (folder / ".boat-secrets").write_text("spare key under the buoy\n")
    return folder


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def browser(home, page_for, person, admin, preserve, open_terminal) -> Locator:
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    region = page.get_by_role("region", name="File browser")
    expect(region.get_by_text("draft-letter.txt", exact=True)).to_be_visible()
    return region


def _page(browser: Locator) -> Page:
    return browser.page


def _choose_action(browser: Locator, action: str) -> None:
    browser.get_by_label("Actions", exact=True).click()
    _page(browser).get_by_role("menu").get_by_role("menuitem", name=action).click()


def _row_menu(browser: Locator, name: str) -> Locator:
    row = browser.get_by_text(name, exact=True).locator(
        "xpath=ancestor::*[.//button[@aria-label='More']][1]"
    )
    row.hover()
    row.get_by_label("More", exact=True).click()
    return _page(browser).get_by_role("menu")


def test_new_folder_makes_the_folder_in_the_terminal(browser, home):
    _choose_action(browser, "New Folder")
    browser.get_by_placeholder("Folder name").fill("charts")
    browser.get_by_placeholder("Folder name").press("Enter")

    expect(browser.get_by_text("charts", exact=True)).to_be_visible()
    assert (home / "charts").is_dir()


def test_new_file_makes_the_file_in_the_terminal(browser, home):
    _choose_action(browser, "New File")
    browser.get_by_placeholder("File name").fill("tide-log.md")
    browser.get_by_placeholder("File name").press("Enter")

    expect(browser.get_by_text("tide-log.md", exact=True)).to_be_visible()
    assert (home / "tide-log.md").is_file()


def test_rename_renames_the_file_in_the_terminal(browser, home):
    moves: list[str] = []
    _page(browser).on(
        "request",
        lambda request: moves.append(request.url) if "/files/move" in request.url else None,
    )
    _row_menu(browser, "draft-letter.txt").get_by_role("menuitem", name="Rename").click()
    field = browser.locator("input:focus")
    expect(field).to_have_value("draft-letter.txt")
    field.fill("sent-letter.txt")
    field.press("Enter")

    expect(_page(browser).get_by_text("Renamed to sent-letter.txt")).to_be_visible()
    # the field losing focus after Enter renamed the old path a second time (#33786)
    assert len(moves) == 1, f"one rename sent {len(moves)} moves"
    expect(browser.get_by_text("sent-letter.txt", exact=True)).to_be_visible()
    expect(browser.get_by_text("draft-letter.txt", exact=True)).to_have_count(0)
    assert (home / "sent-letter.txt").read_text() == "Dear harbour master,\n"
    assert not (home / "draft-letter.txt").exists()


def test_delete_removes_the_file_from_the_terminal_after_a_confirm(browser, home):
    _row_menu(browser, "old-receipt.txt").get_by_role("menuitem", name="Delete").click()
    assert (home / "old-receipt.txt").exists()
    _page(browser).get_by_role("button", name="Confirm").click()

    expect(browser.get_by_text("old-receipt.txt", exact=True)).to_have_count(0)
    assert not (home / "old-receipt.txt").exists()


def test_show_hidden_files_lists_a_dotfile(browser):
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_have_count(0)

    _choose_action(browser, "Show Hidden Files")
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_be_visible()

    _choose_action(browser, "Hide Hidden Files")
    expect(browser.get_by_text(".boat-secrets", exact=True)).to_have_count(0)


def test_compare_with_selected_opens_both_files_and_swap_sides_swaps_them(browser):
    _row_menu(browser, "draft-letter.txt").get_by_role(
        "menuitem", name="Select for Compare"
    ).click()
    expect(browser.get_by_text("Comparing with draft-letter.txt")).to_be_visible()
    _row_menu(browser, "old-receipt.txt").get_by_role(
        "menuitem", name="Compare with Selected"
    ).click()

    compare = _page(browser).get_by_role("dialog", name="Compare files")
    panes = compare.get_by_role("textbox")
    expect(panes).to_have_count(2)
    expect(panes.first).to_have_accessible_name("draft-letter.txt")
    expect(panes.first).to_have_text("Dear harbour master,")
    expect(panes.last).to_have_accessible_name("old-receipt.txt")
    expect(panes.last).to_have_text("two ropes, one lamp")
    # not_to_be_editable reads only input readonly on the pinned Playwright 1.49
    expect(panes.first).to_have_attribute("contenteditable", "false")
    expect(compare.get_by_role("status")).to_have_text("Change 1 of 1")

    compare.get_by_role("button", name="Swap sides").click()
    expect(panes.first).to_have_accessible_name("old-receipt.txt")
    expect(panes.first).to_have_text("two ropes, one lamp")
    expect(panes.last).to_have_accessible_name("draft-letter.txt")
    expect(panes.last).to_have_text("Dear harbour master,")


def _git(folder, *arguments: str) -> None:
    done = subprocess.run(
        ["git", *arguments], cwd=folder, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, f"git {arguments[0]} failed: {done.stderr}"


@pytest.fixture
def repository(home):
    """A Git repository in the home with a changed, a staged, a new and an unchanged file."""
    folder = home / f"logbook-{uuid.uuid4().hex[:6]}"
    (folder / "charts").mkdir(parents=True)
    _git(folder, "init", "-q", "-b", "main")
    _git(folder, "config", "user.name", "Ada Harbour")
    _git(folder, "config", "user.email", "ada@example.com")
    (folder / "tides.txt").write_text("high water at six\nlow water at noon\n")
    (folder / "crew.txt").write_text("Ada\n")
    (folder / "charts" / "bay.txt").write_text("rocks to the north\n")
    (folder / "wrecks").mkdir()
    (folder / "wrecks" / "brig.txt").write_text("off the point\n")
    (folder / "wrecks" / "schooner.txt").write_text("in the bay\n")
    (folder / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00first")
    _git(folder, "add", ".")
    _git(folder, "commit", "-q", "-m", "Start the log")
    (folder / "wrecks" / "brig.txt").unlink()
    (folder / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00second")
    (folder / "tides.txt").write_text("high water at seven\nlow water at noon\nneap tide\n")
    (folder / "charts" / "bay.txt").write_text("rocks to the north and east\n")
    (folder / "harbour.txt").write_text("berth 4\n")
    _git(folder, "add", "harbour.txt")
    (folder / "visitors.txt").write_text("the pilot\n")
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def _git_marker(browser: Locator, name: str) -> Locator:
    row = browser.get_by_text(name, exact=True).locator("xpath=ancestor::li[1]")
    return row.get_by_role("img", name=re.compile(r"·"))


def test_changed_files_and_their_folders_carry_git_markers(browser, repository):
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(repository.name, exact=True).click()
    expect(browser.get_by_text("visitors.txt", exact=True)).to_be_visible()

    expected = {
        "tides.txt": "Modified · Unstaged changes",
        "harbour.txt": "Added · Staged changes",
        "visitors.txt": "Untracked · Unstaged changes",
        "charts": "Contains Modified · Unstaged changes",
        "wrecks": "Contains Deleted · Unstaged changes",
        "chart.png": "Modified · Unstaged changes",
    }
    for name, label in expected.items():
        expect(_git_marker(browser, name), f"the git marker of {name}").to_have_accessible_name(
            label
        )
    expect(_git_marker(browser, "tides.txt")).to_have_text(re.compile(r"\+2\s*-1"))
    # a binary file has no line counts, so its marker shows the status letter
    expect(_git_marker(browser, "chart.png")).to_have_text("M")
    expect(_git_marker(browser, "crew.txt")).to_have_count(0)


def test_a_file_saved_in_the_browser_gets_its_git_marker(browser, repository):
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(repository.name, exact=True).click()
    expect(browser.get_by_text("crew.txt", exact=True)).to_be_visible()
    expect(_git_marker(browser, "crew.txt")).to_have_count(0)

    browser.get_by_text("crew.txt", exact=True).click()
    browser.get_by_text("Ada", exact=True).click()
    _page(browser).keyboard.press("End")
    _page(browser).keyboard.type(" and Ben")
    _page(browser).keyboard.press("Control+s")
    crew = repository / "crew.txt"
    assert wait_until(lambda: crew.read_text() == "Ada and Ben\n", 10), crew.read_text()
    # the Files tab is named after the folder it shows
    _page(browser).get_by_role("tab", name=repository.name, exact=True).click()

    expect(_git_marker(browser, "crew.txt")).to_have_accessible_name("Modified · Unstaged changes")


def test_a_file_dragged_onto_a_folder_moves_into_it(browser, home):
    letters = home / f"letters-{uuid.uuid4().hex[:6]}"
    letters.mkdir()
    browser.get_by_label("Refresh", exact=True).first.click()
    folder = browser.get_by_text(letters.name, exact=True)
    expect(folder).to_be_visible()

    browser.get_by_text("draft-letter.txt", exact=True).drag_to(folder)

    expect(browser.get_by_text("draft-letter.txt", exact=True)).to_have_count(0)
    moved = letters / "draft-letter.txt"
    assert wait_until(moved.exists, 10), "the dragged file never reached the folder"
    assert moved.read_text() == "Dear harbour master,\n"
    assert not (home / "draft-letter.txt").exists(), "the file is still in its old folder"


def test_the_grid_view_shows_folders_with_their_item_count_and_is_kept_after_a_reload(
    browser, home
):
    charts = home / f"charts-{uuid.uuid4().hex[:6]}"
    charts.mkdir()
    (charts / "bay.txt").write_text("rocks\n")
    (charts / "cove.txt").write_text("sand\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    expect(browser.get_by_text(charts.name, exact=True)).to_be_visible()
    # in the list a row's name is its whole line, so no row is named by the file alone
    expect(browser.get_by_role("button", name="draft-letter.txt", exact=True)).to_have_count(0)

    _choose_action(browser, "Grid view")

    folder = browser.get_by_role("button", name=charts.name, exact=True)
    expect(folder).to_contain_text("2 items")
    expect(browser.get_by_role("button", name="draft-letter.txt", exact=True)).to_be_visible()
    browser.get_by_label("Actions", exact=True).click()
    menu = _page(browser).get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Grid view")).to_have_attribute("aria-pressed", "true")
    _page(browser).keyboard.press("Escape")
    expect(menu).to_have_count(0)

    folder.click()
    expect(browser.get_by_role("button", name="cove.txt", exact=True)).to_be_visible()

    page = _page(browser)
    page.reload()
    # the reloaded chat keeps the terminal picked; its File browser opens from New tab again
    picked = page.get_by_role("main").get_by_role("button", name="Open Terminal", exact=True)
    expect(picked).to_be_visible()
    region = open_files_tab(page)
    expect(region.get_by_role("button", name="draft-letter.txt", exact=True)).to_be_visible()

    _choose_action(region, "List view")
    expect(region.get_by_role("button", name="draft-letter.txt", exact=True)).to_have_count(0)
    expect(region.get_by_text("draft-letter.txt", exact=True)).to_be_visible()


# several Files tabs, selecting by dragging, spring-loaded tabs and attaching by dragging

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")


def _folder_of(home, stem: str, *files: str, folders: tuple[str, ...] = ()):
    """A fresh folder in the home holding `files` (each with its name as its text) and `folders`."""
    folder = home / f"{stem}-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    for name in files:
        (folder / name).write_text(f"{name}\n")
    for name in folders:
        (folder / name).mkdir()
    return folder


def _visible_browser(page: Page) -> Locator:
    return page.get_by_role("region", name="File browser")


def _tab(page: Page, name: str) -> Locator:
    return page.get_by_role("tablist", name="Tabs").get_by_role("tab", name=name, exact=True)


def _enter(browser: Locator, folder, shows: str) -> None:
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(folder.name, exact=True).click()
    expect(browser.get_by_text(shows, exact=True)).to_be_visible()


def _row(browser: Locator, name: str) -> Locator:
    return browser.locator("[data-file-row]").filter(has=browser.page.get_by_text(name, exact=True))


def test_two_files_tabs_keep_their_own_folder_and_a_move_shows_in_both(browser, home):
    dock = _folder_of(home, "dock", "manifest.txt")
    page = _page(browser)
    _enter(browser, dock, "manifest.txt")
    expect(_tab(page, dock.name)).to_be_visible()

    # a new Files tab opens on the folder of the one before it
    second = open_files_tab(page)
    second.get_by_role("button", name="Home", exact=True).click()

    expect(_tab(page, HOME_TAB)).to_have_attribute("aria-selected", "true")
    expect(second.get_by_text("draft-letter.txt", exact=True)).to_be_visible()
    second.get_by_text("draft-letter.txt", exact=True).drag_to(
        second.get_by_text(dock.name, exact=True)
    )
    expect(second.get_by_text("draft-letter.txt", exact=True)).to_have_count(0)
    assert wait_until((dock / "draft-letter.txt").exists, 10), "the move never reached the disk"

    _tab(page, dock.name).click()
    first = _visible_browser(page)
    expect(first.get_by_text("manifest.txt", exact=True)).to_be_visible()
    # no Refresh: the move in the other Files tab of this terminal reloads this one
    expect(first.get_by_text("draft-letter.txt", exact=True)).to_be_visible()
    expect(_tab(page, HOME_TAB)).to_be_visible()


def test_a_box_dragged_over_files_selects_them(browser, home):
    crates = _folder_of(home, "crates", "apples.txt", "pears.txt", "plums.txt")
    _enter(browser, crates, "plums.txt")
    page = _page(browser)
    # the scrolling list holds the rows and the empty space under them
    listing = browser.locator("[data-file-list]").locator("xpath=..").bounding_box()
    pears = _row(browser, "pears.txt").bounding_box()
    assert listing and pears, "the list has no place on the page"

    # from the empty space at the foot of the list up into the middle row
    x = pears["x"] + pears["width"] / 2
    page.mouse.move(x, listing["y"] + listing["height"] - 8)
    page.mouse.down()
    page.mouse.move(x, pears["y"] + pears["height"] + 40, steps=5)
    page.mouse.move(x, pears["y"] + pears["height"] / 2, steps=5)
    page.mouse.up()

    expect(browser.get_by_text("2 selected", exact=True)).to_be_visible()
    browser.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Confirm", exact=True).click()
    assert wait_until(lambda: not (crates / "pears.txt").exists(), 10), "pears.txt was not deleted"
    assert wait_until(lambda: not (crates / "plums.txt").exists(), 10), "plums.txt was not deleted"
    assert (crates / "apples.txt").exists(), "a file outside the box was deleted"


def test_a_file_held_over_another_files_tab_moves_into_that_tabs_folder(browser, home):
    dock = _folder_of(home, "dock", "manifest.txt", folders=("hold",))
    page = _page(browser)
    _enter(browser, dock, "manifest.txt")
    open_files_tab(page).get_by_role("button", name="Home", exact=True).click()
    expect(_tab(page, HOME_TAB)).to_have_attribute("aria-selected", "true")
    home_browser = _visible_browser(page)
    source = home_browser.get_by_text("old-receipt.txt", exact=True)
    expect(source).to_be_visible()
    start = source.bounding_box()
    target_tab = _tab(page, dock.name).bounding_box()
    assert start and target_tab, "the file or the tab has no place on the page"

    page.mouse.move(start["x"] + 10, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(target_tab["x"] + 10, target_tab["y"] + target_tab["height"] / 2, steps=8)
    page.mouse.move(target_tab["x"] + 14, target_tab["y"] + target_tab["height"] / 2, steps=2)
    # held there, the Files tab of the dock opens by itself
    expect(_tab(page, dock.name)).to_have_attribute("aria-selected", "true")
    hold = _visible_browser(page).get_by_text("hold", exact=True)
    expect(hold).to_be_visible()
    target = hold.bounding_box()
    assert target, "the folder has no place on the page"
    page.mouse.move(target["x"] + 10, target["y"] + target["height"] / 2, steps=8)
    page.mouse.up()

    moved = dock / "hold" / "old-receipt.txt"
    assert wait_until(moved.exists, 10), "the held file never reached the folder of the other tab"
    assert not (home / "old-receipt.txt").exists(), "the file is still in its old folder"


def _attach_by_dragging(browser: Locator, folder, *names: str) -> Page:
    """Open the folder, select `names`, put the conversation beside the tabs and drag them onto
    the message box."""
    page = _page(browser)
    _enter(browser, folder, names[-1])
    for name in names:
        _row(browser, name).click(modifiers=["Control"])
    expect(browser.get_by_text(f"{len(names)} selected", exact=True)).to_be_visible()
    page.keyboard.press("Control+Backslash")
    expect(chat_input(page)).to_be_visible()
    _visible_browser(page).get_by_text(names[0], exact=True).drag_to(chat_input(page))
    return page


def test_files_dragged_onto_the_message_box_are_attached_and_sent(browser, home, person, upstream):
    logs = _folder_of(home, "logs", "monday.txt", "tuesday.txt", "wednesday.txt")
    page = _attach_by_dragging(browser, logs, "monday.txt", "tuesday.txt")
    composer = page.locator("form").filter(has=chat_input(page))
    expect(composer.get_by_text("monday.txt")).to_be_visible()
    expect(composer.get_by_text("tuesday.txt")).to_be_visible()
    expect(composer.get_by_text("wednesday.txt")).to_have_count(0)

    prompt = f"read these logs {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Both read.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Both read.")

    expect(page).to_have_url(CHAT_URL, timeout=REPLY_TIMEOUT_MS)
    chat_id = CHAT_URL.search(page.url).group(1)
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    files = [file for message in messages if message["role"] == "user" for file in message["files"]]
    assert sorted((file["type"], file["path"]) for file in files) == [
        ("filesystem", str(logs / "monday.txt")),
        ("filesystem", str(logs / "tuesday.txt")),
    ], f"the message should carry both dragged files with their paths: {files}"


@pytest.fixture
def uploads_denied(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["file_upload"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def test_without_file_upload_a_dragged_file_is_not_attached(uploads_denied, browser, home):
    logs = _folder_of(home, "logs", "monday.txt")
    page = _attach_by_dragging(browser, logs, "monday.txt")

    expect(page.get_by_text("You do not have permission to upload files.")).to_be_visible()
    composer = page.locator("form").filter(has=chat_input(page))
    expect(composer.get_by_text("monday.txt")).to_have_count(0)


def _marked_lines(browser: Locator, change: str) -> list[str]:
    """The line numbers the code editor marks as `change` (added or modified) since the commit."""
    # the marks are a colour in the line number gutter, with no text or name of their own
    return browser.locator(f".cm-lineNumbers .cm-git-line-{change}").all_inner_texts()


def test_the_editor_marks_the_lines_changed_since_the_last_commit(browser, repository):
    _enter(browser, repository, "tides.txt")
    browser.get_by_text("tides.txt", exact=True).click()
    expect(browser.get_by_text("neap tide", exact=True)).to_be_visible()

    assert wait_until(lambda: _marked_lines(browser, "modified") == ["1"], 10), (
        f"the changed first line is not marked: {_marked_lines(browser, 'modified')}"
    )
    assert _marked_lines(browser, "added") == ["3"], (
        f"only the new third line should be marked added: {_marked_lines(browser, 'added')}"
    )


def test_a_line_typed_into_a_committed_file_is_marked_at_once(browser, repository):
    _enter(browser, repository, "crew.txt")
    browser.get_by_text("crew.txt", exact=True).click()
    browser.get_by_text("Ada", exact=True).click()
    expect(
        browser.locator(".cm-lineNumbers .cm-gutterElement").filter(has_text="1")
    ).to_be_visible()
    assert _marked_lines(browser, "modified") == [] and _marked_lines(browser, "added") == []

    _page(browser).keyboard.press("End")
    _page(browser).keyboard.press("Enter")
    _page(browser).keyboard.type("Ben")

    assert wait_until(lambda: _marked_lines(browser, "added") == ["2"], 10), (
        f"the typed line is not marked added: {_marked_lines(browser, 'added')}"
    )
    assert _marked_lines(browser, "modified") == []
