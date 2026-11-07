"""Journey: the tabbed workspace of a chat and its Git panel, on a real Open Terminal.

Since v0.13.0 (2d7cf5ec9 and 58926cfe7) the Tabs button of a chat opens a workspace of tabs. A
person who picked an Open Terminal adds a shell and a new file from the workspace's New tab menu
and drags the file's tab in front of the shell. Use full-width tabs spreads the workspace over
the chat area with the conversation as a tab of its own, and the choice is still in place after a
reload, where a terminal picked keeps the conversation in front and Files opens from New tab
(89b924aae). When the file browser is in a Git repository, the Git panel under the tabs shows a
changed file's diff, stages and commits it, and the commit shows in its History and on the disk.

Discriminates: passes on v0.13.0 and dev 5f1c6384e; on a v0.13.0 build whose dropped tab keeps
its old place, whose layout toggle saves nothing and whose Git panel ignores Stage, each test
fails (the tab order, the side layout after the reload, no Unstage), while a file browser journey
passes on that build.

Since c24c3d914 each changed file's stage box and Stage all are checkboxes that say whether they
are ticked, Stage all reading mixed while only some files are staged; Discard changes also deletes
a file the last commit does not have, after a warning saying so; the list beside a diff can be
resized from the keyboard. The branch menu makes a branch that takes the next commit while main
stays as it was, and the sync button pushes a commit to the remote and publishes a new branch. Undo last commit in
History takes the last commit back and keeps its change staged.

Discriminates: on dev f659c0d12, in a frontend copy whose Stage all never reads mixed the stage
test fails, one that offers Discard only for tracked files (as v0.13.0) fails the discard test,
one whose divider ignores the keyboard fails the resize test, one whose branch pick checks
nothing out fails the branch test and one that offers Fetch in place of Push fails the push test;
the diff and undo tests stay green on that build as its controls.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import choose_terminal, open_new_terminal, pick_terminal
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


def _add_tab(page: Page, entry: str | re.Pattern) -> None:
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
    # the Terminal menu's button is named for the terminal picked in it
    tooltip_button(page.get_by_role("main"), name).click()
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
    # since 89b924aae a terminal picked in full-width tabs leaves the conversation in front
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=terminal).click()
    expect(tooltip_button(page.get_by_role("main"), terminal)).to_be_visible()
    expect(layout).to_have_attribute("aria-pressed", "true")
    _add_tab(page, "Files")
    expect(_tab_strip(page).get_by_role("tab")).to_have_count(2)
    _tab_strip(page).get_by_role("tab", name="Files").click()
    after_reload = _file_browser_width(page)
    assert after_reload > side_width * 1.5, (
        f"after a reload the file browser is back to {after_reload}px, the side layout"
    )


def _open_git_panel(page: Page, terminal: str, folder) -> Locator:
    """Pick the terminal, open `folder` in its file browser and expand the Git panel under it."""
    choose_terminal(page, terminal)
    # the conversation stays in front; Files opens from the New tab menu
    _add_tab(page, re.compile(r"^Files"))
    files = page.get_by_role("region", name="File browser")
    files.get_by_text(folder.name, exact=True).click()
    git = page.get_by_role("region", name="Git")
    # the panel shows once the file browser stands in the repository
    git.get_by_role("button", name="Expand Git panel").click()
    expect(git.get_by_role("button", name="Collapse Git panel")).to_be_visible()
    return git


def _staged(folder) -> set[str]:
    return set(_git(folder, "diff", "--cached", "--name-only").split())


def test_the_git_panel_shows_a_changed_files_diff_and_commits_it(terminal_page, repository):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    git.get_by_role("button", name="View changes tides.txt").first.click()
    expect(git.get_by_text("high water at seven")).to_be_visible()
    expect(git.get_by_text("high water at six")).to_be_visible()

    # since c24c3d914 a file's stage box is a checkbox that reads its state
    git.get_by_role("checkbox", name="Stage tides.txt").click()
    expect(git.get_by_role("checkbox", name="Unstage tides.txt")).to_have_attribute(
        "aria-checked", "true"
    )
    assert _staged(repository) == {"tides.txt"}
    git.get_by_label("Commit summary").fill("Move high water to seven")
    git.get_by_role("button", name="Commit 1 file to main").click()
    expect(git.get_by_text("No changes")).to_be_visible()
    assert _git(repository, "log", "-1", "--format=%s").strip() == "Move high water to seven"
    assert _git(repository, "status", "--porcelain") == "", "the commit left changes behind"

    git.get_by_role("button", name="History").click()
    expect(git.get_by_text("Move high water to seven")).to_be_visible()
    expect(git.get_by_text("Start the tide log")).to_be_visible()


def test_stage_boxes_stage_one_file_or_all_of_them(terminal_page, repository):
    page, terminal = terminal_page
    (repository / "moon.txt").write_text("full moon on friday\n")
    git = _open_git_panel(page, terminal, repository)
    every_file = git.get_by_role("checkbox", name="Stage all")
    expect(every_file).to_have_attribute("aria-checked", "false")

    git.get_by_role("checkbox", name="Stage moon.txt").click()
    expect(git.get_by_role("checkbox", name="Unstage moon.txt")).to_have_attribute(
        "aria-checked", "true"
    )
    expect(every_file).to_have_attribute("aria-checked", "mixed")
    assert _staged(repository) == {"moon.txt"}, "Stage staged other files than the one ticked"
    expect(git.get_by_role("button", name="Commit 1 file to main")).to_be_visible()

    every_file.click()
    every_file = git.get_by_role("checkbox", name="Unstage all")
    expect(every_file).to_have_attribute("aria-checked", "true")
    assert _staged(repository) == {"moon.txt", "tides.txt"}
    expect(git.get_by_role("button", name="Commit 2 files to main")).to_be_visible()

    every_file.click()
    expect(git.get_by_role("checkbox", name="Stage all")).to_have_attribute("aria-checked", "false")
    assert _staged(repository) == set(), "Unstage all left files staged"


def _branch_menu(git: Locator) -> Locator:
    git.get_by_role("button", name="Switch branch").click()
    search = git.page.get_by_role("textbox", name="Search branches")
    menu = git.page.get_by_role("menu").filter(has=search)
    expect(search).to_be_visible()
    return menu


def test_a_branch_made_in_the_branch_menu_takes_the_commit_and_main_stays_as_it_was(
    terminal_page, repository
):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)

    menu = _branch_menu(git)
    menu.get_by_role("menuitem", name="New branch").click()
    menu.get_by_role("textbox", name="New branch").fill("evening-tides")
    menu.get_by_role("menuitem", name="Create branch").click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("evening-tides")
    assert _git(repository, "branch", "--show-current").strip() == "evening-tides"

    git.get_by_role("checkbox", name="Stage tides.txt").click()
    git.get_by_label("Commit summary").fill("Tides in the evening")
    git.get_by_role("button", name="Commit 1 file to evening-tides").click()
    expect(git.get_by_text("No changes")).to_be_visible()

    menu = _branch_menu(git)
    main = menu.get_by_role("menuitem", name="main", exact=True)
    # the menu lists the branches it had until its reload marks main as no longer current
    expect(main).to_be_enabled()
    main.click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("main")
    assert _git(repository, "branch", "--show-current").strip() == "main"
    assert (repository / "tides.txt").read_text().startswith("high water at six"), (
        "switching back to main left the branch's version of the file on disk"
    )
    git.get_by_role("button", name="History").click()
    expect(git.get_by_text("Start the tide log")).to_be_visible()
    expect(git.get_by_text("Tides in the evening")).to_have_count(0)


def _discard(git: Locator, path: str) -> None:
    git.get_by_role("button", name=f"File actions {path}").click()
    git.page.get_by_role("menu").get_by_role("menuitem", name="Discard changes").click()
    dialog = git.page.get_by_role("dialog")
    expect(dialog).to_contain_text("Files not in the last commit will be deleted")
    dialog.get_by_role("button", name="Confirm").click()


def test_discarding_a_new_file_deletes_it_and_a_staged_change_goes_back_to_the_commit(
    terminal_page, repository
):
    page, terminal = terminal_page
    draft = repository / "draft.txt"
    draft.write_text("a draft nobody wants\n")
    _git(repository, "add", "tides.txt")
    git = _open_git_panel(page, terminal, repository)
    expect(git.get_by_role("checkbox", name="Unstage tides.txt")).to_be_visible()

    # since c24c3d914 a file not in the last commit can be discarded too
    _discard(git, "draft.txt")
    expect(git.get_by_role("checkbox", name="Stage draft.txt")).to_have_count(0)
    assert not draft.exists(), "discarding the new file left it on disk"

    _discard(git, "tides.txt")
    expect(git.get_by_text("No changes")).to_be_visible()
    assert (repository / "tides.txt").read_text() == "high water at six\nlow water at noon\n"
    assert _git(repository, "status", "--porcelain") == "", "the discard left changes behind"


@pytest.fixture
def remote(repository):
    """A bare repository the logbook pushes to, with main published and tracked."""
    bare = repository.parent / f"{repository.name}-remote.git"
    _git(repository.parent, "init", "-q", "--bare", "-b", "main", str(bare))
    _git(repository, "remote", "add", "origin", str(bare))
    _git(repository, "push", "-q", "-u", "origin", "main")
    yield bare
    shutil.rmtree(bare, ignore_errors=True)


def test_a_commit_is_pushed_and_a_new_branch_published_from_the_sync_button(
    terminal_page, repository, remote
):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    git.get_by_role("checkbox", name="Stage tides.txt").click()
    git.get_by_label("Commit summary").fill("Push the seven o'clock tide")
    git.get_by_role("button", name="Commit 1 file to main").click()
    expect(git.get_by_text("No changes")).to_be_visible()

    git.get_by_role("button", name="Push", exact=True).click()
    expect(git.get_by_role("button", name="Fetch", exact=True)).to_be_visible()
    assert _git(remote, "log", "-1", "--format=%s", "main").strip() == (
        "Push the seven o'clock tide"
    ), "the push did not reach the remote"

    menu = _branch_menu(git)
    menu.get_by_role("menuitem", name="New branch").click()
    menu.get_by_role("textbox", name="New branch").fill("spring-tides")
    menu.get_by_role("menuitem", name="Create branch").click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("spring-tides")
    git.get_by_role("button", name="Publish", exact=True).click()
    expect(git.get_by_role("button", name="Fetch", exact=True)).to_be_visible()
    assert "spring-tides" in _git(remote, "branch", "--format=%(refname:short)").split()


def test_the_list_beside_a_diff_is_resized_from_the_keyboard(terminal_page, repository):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    divider = git.get_by_role("separator", name="Resize Git list")
    expect(divider).to_have_attribute("aria-valuenow", re.compile(r"^\d+$"))
    start = int(divider.get_attribute("aria-valuenow"))

    divider.focus()
    page.keyboard.press("ArrowRight")
    expect(divider).to_have_attribute("aria-valuenow", str(start + 20))
    page.keyboard.press("End")
    expect(divider).to_have_attribute("aria-valuenow", divider.get_attribute("aria-valuemax"))
    widest = git.get_by_role("checkbox", name="Stage all").bounding_box()["width"]
    page.keyboard.press("Home")
    expect(divider).to_have_attribute("aria-valuenow", "160")
    narrowest = git.get_by_role("checkbox", name="Stage all").bounding_box()["width"]
    assert narrowest < widest, f"the list stayed {widest}px wide at its narrowest setting"


def test_undoing_the_last_commit_from_history_keeps_its_changes_staged(terminal_page, repository):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    git.get_by_role("checkbox", name="Stage tides.txt").click()
    git.get_by_label("Commit summary").fill("A tide too early")
    git.get_by_role("button", name="Commit 1 file to main").click()
    expect(git.get_by_text("No changes")).to_be_visible()

    git.get_by_role("button", name="History").click()
    expect(git.get_by_text("A tide too early")).to_be_visible()
    last = _git(repository, "log", "-1", "--format=%h").strip()
    git.get_by_text("A tide too early").hover()
    git.get_by_role("button", name=f"Commit actions {last}").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Undo last commit").click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()

    expect(git.get_by_text("A tide too early")).to_have_count(0)
    assert _git(repository, "log", "-1", "--format=%s").strip() == "Start the tide log"
    assert _staged(repository) == {"tides.txt"}, "the undone commit's change is not staged"
