"""Journey: the tabbed workspace of a chat and its Git panel, on a real Open Terminal.

Since v0.13.0 (2d7cf5ec9 and 58926cfe7) the Tabs button of a chat opens a workspace of tabs. A
person who picked an Open Terminal adds a shell and a new file from the workspace's New tab menu
and drags the file's tab in front of the shell. Use full-width tabs spreads the workspace over
the chat area with the conversation as a tab of its own, and the choice is still in place after a
reload. When the file browser is in a Git repository, the Git panel under the tabs shows a changed
file's diff, stages and commits it, and the commit shows in its History and on the disk.

Discriminates: passes on v0.13.0 and dev 5f1c6384e; on a v0.13.0 build whose dropped tab keeps
its old place, whose layout toggle saves nothing and whose Git panel ignores Stage, each test
fails (the tab order, the side layout after the reload, no Unstage), while a file browser journey
passes on that build.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import open_new_terminal, pick_terminal
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def terminal_page(page_for, make_user, admin, preserve, open_terminal) -> tuple[Page, str]:
    """A chat page of a fresh account with read access to the Open Terminal, and its name."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return page_for(person), connection["name"]


@pytest.fixture
def repository(open_terminal):
    """A Git repository in the terminal's home with one committed file, changed since."""
    folder = open_terminal.home / f"logbook-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    _git(folder, "init", "-q", "-b", "main")
    _git(folder, "config", "user.name", "Ada Harbour")
    _git(folder, "config", "user.email", "ada@example.com")
    (folder / "tides.txt").write_text("high water at six\nlow water at noon\n")
    _git(folder, "add", "tides.txt")
    _git(folder, "commit", "-q", "-m", "Start the tide log")
    (folder / "tides.txt").write_text("high water at seven\nlow water at noon\n")
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def _git(folder, *arguments: str) -> str:
    done = subprocess.run(
        ["git", *arguments], cwd=folder, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, f"git {arguments[0]} failed: {done.stderr}"
    return done.stdout


def _tab_strip(page: Page) -> Locator:
    return page.get_by_role("tablist", name="Tabs")


def _tab_names(page: Page) -> list[str]:
    return [name.strip() for name in _tab_strip(page).get_by_role("tab").all_inner_texts()]


def _add_tab(page: Page, entry: str) -> None:
    page.get_by_role("button", name="New tab", exact=True).click()
    page.get_by_role("menu").get_by_role("menuitem", name=entry).click()


def _drag(page: Page, source: Locator, target: Locator) -> None:
    """Drag `source` by the mouse onto the left edge of `target`, in steps as a hand moves."""
    start = source.bounding_box()
    end = target.bounding_box()
    page.mouse.move(start["x"] + start["width"] / 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(start["x"] + start["width"] / 2 - 10, start["y"] + start["height"] / 2, steps=5)
    page.mouse.move(end["x"] + 4, end["y"] + end["height"] / 2, steps=20)
    page.mouse.up()


def _unpick_terminal(page: Page, name: str) -> None:
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    choice = page.get_by_role("menu").get_by_role("menuitemradio", name=name)
    expect(choice).to_have_attribute("aria-checked", "true")
    choice.click()
    expect(page.get_by_role("region", name="File browser")).to_have_count(0)


def _file_browser_width(page: Page) -> float:
    files = page.get_by_role("region", name="File browser")
    expect(files).to_be_visible()
    return files.bounding_box()["width"]


def test_tabs_added_from_the_new_tab_menu_can_be_dragged_into_a_new_order(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    open_new_terminal(page)
    _add_tab(page, "New File")
    strip = _tab_strip(page)
    expect(strip.get_by_role("tab", name="Untitled-1")).to_be_visible()
    assert _tab_names(page) == ["Files", "Terminal 1", "Untitled-1"], (
        f"the New tab menu added the tabs as {_tab_names(page)}"
    )

    _drag(
        page,
        strip.get_by_role("tab", name="Untitled-1"),
        strip.get_by_role("tab", name="Terminal 1"),
    )
    expect(strip.get_by_role("tab").nth(1)).to_have_text("Untitled-1")
    assert _tab_names(page) == ["Files", "Untitled-1", "Terminal 1"], (
        f"dragging the new file's tab before the shell left the tabs as {_tab_names(page)}"
    )


def test_the_full_width_layout_is_kept_after_a_reload(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    side_width = _file_browser_width(page)

    page.get_by_role("button", name="Use full-width tabs").click()
    layout = page.get_by_role("button", name="Use side tabs")
    expect(layout).to_have_attribute("aria-pressed", "true")
    # the conversation becomes the first tab, in front of the Files tab
    expect(_tab_strip(page).get_by_role("tab")).to_have_count(2)
    _tab_strip(page).get_by_role("tab", name="Files").click()
    full_width = _file_browser_width(page)
    assert full_width > side_width * 1.5, (
        f"the file browser is {full_width}px wide in full-width tabs and {side_width}px beside"
    )

    # unpicked, so the reloaded draft brings no terminal back before the panel has loaded
    _tab_strip(page).get_by_role("tab").first.click()
    expect(chat_input(page)).to_be_visible()
    _unpick_terminal(page, terminal)
    page.reload()
    pick_terminal(page, terminal)
    expect(layout).to_have_attribute("aria-pressed", "true")
    expect(_tab_strip(page).get_by_role("tab")).to_have_count(2)
    _tab_strip(page).get_by_role("tab", name="Files").click()
    after_reload = _file_browser_width(page)
    assert after_reload > side_width * 1.5, (
        f"after a reload the file browser is back to {after_reload}px, the side layout"
    )


def test_the_git_panel_shows_a_changed_files_diff_and_commits_it(terminal_page, repository):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    files = page.get_by_role("region", name="File browser")
    files.get_by_text(repository.name, exact=True).click()
    expect(files.get_by_text("tides.txt", exact=True)).to_be_visible()

    git = page.get_by_role("region", name="Git")
    git.get_by_role("button", name="Expand Git panel").click()
    git.get_by_role("button", name="View changes tides.txt").first.click()
    expect(git.get_by_text("high water at seven")).to_be_visible()
    expect(git.get_by_text("high water at six")).to_be_visible()
    git.get_by_role("button", name="Back to changes").click()

    git.get_by_role("button", name="Stage tides.txt").click()
    expect(git.get_by_role("button", name="Unstage tides.txt")).to_be_visible()
    git.get_by_label("Commit summary").fill("Move high water to seven")
    git.get_by_role("button", name="Commit 1 file to main").click()
    expect(git.get_by_text("No changes")).to_be_visible()
    assert _git(repository, "log", "-1", "--format=%s").strip() == "Move high water to seven"
    assert _git(repository, "status", "--porcelain") == "", "the commit left changes behind"

    git.get_by_role("button", name="History").click()
    expect(git.get_by_text("Move high water to seven")).to_be_visible()
    expect(git.get_by_text("Start the tide log")).to_be_visible()
