"""Journey: the tabbed workspace of a chat, its split panes, its shortcuts and its Git panel.

Since v0.13.0 (2d7cf5ec9 and 58926cfe7) a chat has a workspace of tabs. A person who picked an
Open Terminal adds a shell and a new file from the workspace's New tab menu and drags the file's
tab in front of the shell. Since 617618c9e the conversation is always a tab of the workspace on a
large screen (the Use full-width tabs switch is gone), and the workspace splits into panes
(cd1aadd75, 0349240ce, 80ddf8e3b): Ctrl+Backslash or the Split screen button puts the
conversation and its tabs side by side and back, a tab dragged to the edge of a pane opens in a
pane of its own beside or below it, a tab dragged onto another pane's tab bar joins that pane,
closing a pane's only tab closes the pane, and the divider between two panes resizes them
between a fifth and four fifths, by mouse and by arrow keys. The layout of a saved chat (panes,
their tabs, the active tab, the divider) is saved for its person on the server (d4e8493cc), so
it comes back after leaving the chat and returning, after a reload and in another browser, also
when it was arranged before the first message; an admin opening the same chat gets the
conversation alone. Layout changes made after the browser reports the network back are still
saved. Since da850248d a new chat started from a chat keeps that chat's tabs beside it and takes
its place in the browser history, so the reload test returns to the chat through the sidebar
(it went back in the browser before, and expected the new chat's conversation alone).

`test_a_new_chat_started_beside_an_open_shell_sends_its_first_message` is red on dev c15f28649:
with a shell of the saved chat among the kept tabs, the new chat's first message is never sent
and the page says "Could not attach sessions to this chat" (finding "issue: [dev] The first
message of a new chat is not sent while a shell tab from the previous chat is open", from
da850248d). With only the Files tab kept the message is sent.

The workspace shortcuts (80ddf8e3b) open Files (Ctrl+Shift+E), a shell (Ctrl+`), a new file
(Ctrl+Alt+N) and a browser (Ctrl+Shift+B), step through a pane's tabs (Ctrl+Shift+] and [) and
close the active one (Ctrl+W, never the conversation). They do nothing while a menu or a dialog
is open or with Keyboard Shortcuts switched off; Desktop (Ctrl+Alt+D) does nothing on a terminal
without a desktop, New Chat Tab (Ctrl+Alt+O) needs the side chat permission and Comments
(Ctrl+Shift+Y) needs comments switched on. A tab closed while its session is still starting
closes once the start ends, so the test lets the Browser tab start first: where Open Terminal has
Chrome, the switch to Chrome mode takes up to its 15 s startup limit on a slow runner. When the
file browser is in a Git repository, the Git panel under the tabs shows a changed file's diff,
stages and commits it, and the commit shows in its History and on the disk.

Since c24c3d914 each changed file's stage box and Stage all are checkboxes that say whether they
are ticked, Stage all reading mixed while only some files are staged; Discard changes also deletes
a file the last commit does not have, after a warning saying so; the list beside a diff can be
resized from the keyboard. The branch menu makes a branch that takes the next commit while main
stays as it was, and the sync button pushes a commit to the remote and publishes a new branch.
Undo last commit in History takes the last commit back and keeps its change staged. Since
3797ce419 the panel says what it is doing while it stages or pushes (Staging…, Pushing…).

Discriminates: on dev f659c0d12, in a frontend copy whose Stage all never reads mixed the stage
test fails, one that offers Discard only for tracked files (as v0.13.0) fails the discard test,
one whose divider ignores the keyboard fails the resize test, one whose branch pick checks
nothing out fails the branch test and one that offers Fetch in place of Push fails the push test;
the diff and undo tests stay green on that build as its controls.

Retargeted for 617618c9e: `test_the_full_width_layout_is_kept_after_a_reload` saved the removed
full-width switch and is now the split layout test; the reorder test now counts the conversation
tab. Since 0b7e7b3f7 the Files tab is named after the folder it shows, and since 241c8c0e4
it reads Home in the terminal's home, where these tests open it.

`test_layout_changes_after_a_network_reconnect_are_still_saved` is red on dev f659c0d12: a save
started with nothing pending wedges the chat's layout saving until the next reload
(open-webui/open-webui#33892, from d4e8493cc). The same wedge can hit the reload test when a slow
divider drag outpaces the saves.

Discriminates for the workspace tests, in frontend builds of f659c0d12: with the layout never
sent to the server the reload, first message and reconnect tests fail; with the menu and dialog
guard gone the menu test fails, with the Keyboard Shortcuts switch ignored the switched off test
fails, with side chats open to everyone the side chat test fails and with edge drops never
splitting the drag test fails (the reorder, split, divider and shortcut tests pass there). With
the divider unclamped the divider test fails, with Ctrl+Backslash unbound the split test fails
and with Ctrl+W unbound the shortcut test fails; with the wedge fixed (no save started when
nothing is pending) the reconnect test passes. The reorder test fails on v0.13.0 builds whose
dropped tab keeps its old place.
The activity test passes on dev e8c0ad85e and fails on a copy of the dev e8c0ad85e build that labels
staging and pushing otherwise.
The reorder test also checks that no full-width or side tabs switch is offered; in an e8c0ad85e
build with a Use side tabs button put back it fails.

Over plain HTTP by address the New tab menu opens a New File and a New Terminal tab (PR
open-webui/open-webui#33847, fix b31ae40d2, issue #33845; the browser has no `crypto.randomUUID`
there and no tab opened). The test is red on dev e8c0ad85e because the chat page itself no longer
loads over plain HTTP since d4e8493cc (see e2e/chat/test_chat_plain_http.py); it passes on a build
of e8c0ad85e with that fixed and fails on the same build with b31ae40d2 reverted.

With two terminals connected, picking the other one with the screen split, in a new chat or in
a chat that has messages (allowed again since a518ff163), moves the open Files tab to it: its
files show and the tab reads Home again. A per-chat terminal picked in a new chat has no files
until the chat exists; after the first message the tab shows that chat's own empty home. All
three are red on dev 560f2b4c6 (the new chat ones also on b13d02e09), where the tab kept the first
terminal's folder and files since the workspace of tabs came in 0.13.0
(open-webui/open-webui#33951), and pass on a build of 560f2b4c6 with the fix, which is on dev
since eb2a3a6a2 (#33954). The per-chat test also fails on a build of the fix whose file browser
skips loading when the chat and the terminal arrive together (the tab stays on "/" and reads
Files). Since 6f2fc7bc7 a per-chat terminal is a terminal scoped to the chat on a Terminals
orchestrator, so that test runs against the real service on Docker (`harness.terminals`) and
skips without it; on the dev c15f28649 build it fails with the terminal list no longer telling
the browser a terminal's scope, in a backend copy (the file browser asks for the terminal's
files before the chat exists and fails to load them).

The Git panel's further tools: a split diff with Hide whitespace, the branch menu's search,
rename, copy and delete (a branch with commits of its own is kept and the panel says why), a
worktree opened in the Files tab, Fetch then Pull from the sync button, the History with a
commit's changes, its unpushed mark and Copy commit hash, the link to the repository's page
without credentials, Git unavailable with Retry, live updates followed and their loss shown, and
the diff of an open changed file from its toolbar. The New tab menu and the panel offer only
what the picked terminal says it has; a file opened again keeps its one tab, the arrow keys,
Home and End step through the tabs and Alt+Shift+Arrow moves one and a phone opens the tabs from
the Tabs button. A person's own terminal, let go when they lost the permission to use one, was
removed from Open WebUI in 44d55bd0f and 4720ff19e, and its test with it.

Discriminates for these, in frontend copies of the dev c15f28649 build: the split diff drawn
unified or Hide whitespace hiding nothing fails the diff test; a branch search that narrows
nothing, a Rename or a Delete that does nothing fails the branch test, and Delete doing nothing
the kept branch test; a new worktree not opened, or the current one offered and the list not
narrowed, the worktree test; commits to pull offered as a fetch, or not counted, the sync test;
no unpushed mark, a short hash copied or a commit's changes not shown, the history test;
credentials kept in the link, an SSH address not turned into a web address or a GitLab host
named by its address, the link test; New Terminal or Browser offered whatever the terminal says,
Git assumed where it says nothing, or a pull request view where it has none, the menu test; a
failed detection shown as nothing, or Retry not asking again, the Git unavailable test; lost
live updates not mentioned, a change not followed or the notice kept after they are back, the
live updates test; a second tab for a file opened again, or its crumbs browsing inside the file
tab, the file tab test; Alt+Shift+Arrow moving nothing or Home and End selecting nothing, the
keyboard test; the Tabs button missing for a picked terminal or the drawer opening without the
files, the phone test; a changed binary file read as unchanged or Show diff showing nothing, the
toolbar diff test.
`test_a_tab_moved_with_the_keyboard_keeps_the_focus_for_the_next_move` is red on that build: the
moved tab loses the keyboard focus, so the next Alt+Shift+Arrow does nothing.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.listener import json_answer
from harness.open_terminal import HOME_TAB, serving_open_terminal
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.terminals import create_template, share_template
from utils.browser_tab import browser_view
from utils.cached_chat import choose_terminal, open_new_terminal, pick_terminal
from utils.chat_ui import chat_input, close_menu, expect_reply, send
from utils.clipboard import copied, loopback_clipboard
from utils.git_panel import expand_git_panel
from utils.phone import PHONE
from utils.sidebar_chats import chat_row
from utils.tooltips import tooltip_button
from utils.user_terminals import listed_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FILES = HOME_TAB
# Open Terminal gives a Chrome that does not answer 15 s before it falls back to Proxy mode
BROWSER_START_MS = 45_000


@pytest.fixture
def terminal_account(make_user, admin, preserve, open_terminal) -> tuple[Actor, str]:
    """A fresh account with read access to the Open Terminal, and the terminal's name."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return person, connection["name"]


@pytest.fixture
def terminal_page(page_for, terminal_account) -> tuple[Page, str]:
    """A chat page of that account, and the terminal's name."""
    person, terminal = terminal_account
    return page_for(person), terminal


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


@pytest.fixture
def second_terminal():
    """Another real Open Terminal, with a file of its own in its home."""
    with serving_open_terminal() as terminal:
        (terminal.home / "tide-chart.csv").write_text("high,low\n")
        yield terminal


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


def test_tabs_added_from_the_new_tab_menu_can_be_dragged_into_a_new_order(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    open_new_terminal(page)
    _add_tab(page, "New File")
    strip = _tab_strip(page)
    expect(strip.get_by_role("tab", name="Untitled-1")).to_be_visible()
    assert _tab_names(page) == ["New Chat", FILES, "Terminal 1", "Untitled-1"], (
        f"the New tab menu added the tabs as {_tab_names(page)}"
    )
    # the chat is a tab of the full-width workspace, with no switch back to side tabs
    expect(
        page.get_by_role("button", name=re.compile(r"^Use (full-width|side) tabs$"))
    ).to_have_count(0)

    _drag(
        page,
        strip.get_by_role("tab", name="Untitled-1"),
        strip.get_by_role("tab", name="Terminal 1"),
    )
    expect(strip.get_by_role("tab").nth(2)).to_have_text("Untitled-1")
    assert _tab_names(page) == ["New Chat", FILES, "Untitled-1", "Terminal 1"], (
        f"dragging the new file's tab before the shell left the tabs as {_tab_names(page)}"
    )


def test_the_new_tab_menu_opens_tabs_over_plain_http(lan_page_for, terminal_account):
    """#33845: over plain HTTP by address the browser has no crypto.randomUUID, so no tab opened."""
    person, terminal = terminal_account
    page = lan_page_for(person)
    choose_terminal(page, terminal)
    strip = _tab_strip(page)

    _add_tab(page, re.compile(r"^New File\b"))
    expect(strip.get_by_role("tab", name="Untitled-1")).to_be_visible()
    _add_tab(page, re.compile(r"^New Terminal\b"))
    expect(strip.get_by_role("tab", name="Terminal 1")).to_be_visible()


def _panes(page: Page) -> list[list[str]]:
    """The tab names of each pane, the panes from left to right and then top to bottom."""
    strips = page.get_by_role("tablist", name="Tabs").all()
    placed = sorted(
        strips, key=lambda strip: (round(strip.bounding_box()["x"]), strip.bounding_box()["y"])
    )
    return [
        [name.strip() for name in strip.get_by_role("tab").all_inner_texts()] for strip in placed
    ]


def _expect_panes(page: Page, expected: list[list[str]], any_order: bool = False) -> None:
    """The panes hold these tabs; `any_order` where the order inside a pane is not the point."""
    expect(page.get_by_role("tablist", name="Tabs")).to_have_count(len(expected))

    def held() -> list:
        panes = _panes(page)
        return [sorted(names) for names in panes] if any_order else panes

    wanted = [sorted(names) for names in expected] if any_order else expected
    _wait_until(lambda: held() == wanted)
    assert held() == wanted, f"the panes hold {_panes(page)}, expected {expected}"


def _wait_until(condition, seconds: float = 15.0) -> None:
    """Wait for a state the page or the server reaches on its own; the caller asserts it."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.1)


def _selected(page: Page, name: str) -> None:
    expect(page.get_by_role("tab", name=name, exact=True)).to_have_attribute(
        "aria-selected", "true"
    )


def _divider(page: Page) -> Locator:
    return page.get_by_role("separator", name="Resize panels")


def _drag_to(page: Page, source: Locator, x: float, y: float) -> None:
    """Drag `source` by the mouse to the point, in steps as a hand moves."""
    start = source.bounding_box()
    page.mouse.move(start["x"] + start["width"] / 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(start["x"] + start["width"] / 2 + 10, start["y"] + start["height"] / 2, steps=5)
    page.mouse.move(x, y, steps=20)
    page.mouse.up()


def _answered_chat(page: Page, upstream, terminal: str) -> str:
    """Pick the terminal, ask one question with the conversation in front; returns the chat id."""
    pick_terminal(page, terminal)
    _tab_strip(page).get_by_role("tab", name="New Chat", exact=True).click()
    question = f"plan the harbour walk {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Start at the lighthouse.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Start at the lighthouse.")
    expect(page).to_have_url(re.compile(r"/c/"))
    return page.url.rsplit("/", 1)[-1]


def _open_from_sidebar(page: Page, title: str) -> None:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    chat_row(page.get_by_role("navigation", name="Chat history"), title).click()


def _saved_view(person, chat_id: str) -> dict | None:
    with person.client() as client:
        response = client.get(f"/api/v1/chats/{chat_id}/view")
    assert response.status_code == 200, response.text
    return response.json()["state"]


def test_ctrl_backslash_and_the_split_button_put_the_conversation_beside_its_tabs(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    _expect_panes(page, [["New Chat", FILES]])
    split = page.get_by_role("button", name="Split screen")
    expect(split).to_have_attribute("aria-pressed", "false")

    page.keyboard.press("Control+Backslash")
    _expect_panes(page, [["New Chat"], [FILES]])
    expect(split).to_have_attribute("aria-pressed", "true")
    # side by side: the message box and the file browser both show
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("region", name="File browser")).to_be_visible()
    page.keyboard.press("Control+Backslash")
    _expect_panes(page, [["New Chat", FILES]], any_order=True)
    expect(split).to_have_attribute("aria-pressed", "false")

    split.click()
    _expect_panes(page, [["New Chat"], [FILES]])
    split.click()
    _expect_panes(page, [["New Chat", FILES]], any_order=True)


def test_a_tab_dragged_to_a_panes_edge_opens_beside_it_and_onto_a_tab_bar_joins_it(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    open_new_terminal(page)
    _expect_panes(page, [["New Chat", FILES, "Terminal 1"]])
    shell = page.get_by_role("tabpanel", name="Terminal 1")

    area = shell.bounding_box()
    _drag_to(
        page,
        _tab_strip(page).get_by_role("tab", name="Terminal 1"),
        area["x"] + area["width"] - 20,
        area["y"] + area["height"] / 2,
    )
    _expect_panes(page, [["New Chat", FILES], ["Terminal 1"]])
    expect(_divider(page)).to_have_attribute("aria-orientation", "vertical")
    _selected(page, "Terminal 1")

    _drag_to(
        page,
        _tab_strip(page).get_by_role("tab", name="Terminal 1"),
        *_empty_part_of_tab_bar(page.get_by_role("tablist", name="Tabs").filter(has_text=FILES)),
    )
    _expect_panes(page, [["New Chat", FILES, "Terminal 1"]])
    expect(_divider(page)).to_have_count(0)

    _tab_strip(page).get_by_role("tab", name=FILES).click()
    area = page.get_by_role("region", name="File browser").bounding_box()
    _drag_to(
        page,
        _tab_strip(page).get_by_role("tab", name=FILES),
        area["x"] + area["width"] / 2,
        area["y"] + area["height"] - 20,
    )
    # a drop on the bottom edge stacks the panes, one tab bar above the other
    expect(_divider(page)).to_have_attribute("aria-orientation", "horizontal")
    _expect_panes(page, [["New Chat", "Terminal 1"], [FILES]])
    upper, lower = (
        strip.bounding_box() for strip in page.get_by_role("tablist", name="Tabs").all()
    )
    assert abs(upper["x"] - lower["x"]) < 2 and abs(upper["y"] - lower["y"]) > 100, (
        f"the two panes' tab bars are at {upper} and {lower}, not one above the other"
    )

    # closing a pane's only tab closes the pane
    _selected(page, FILES)
    page.keyboard.press("Control+W")
    _expect_panes(page, [["New Chat", "Terminal 1"]])
    expect(_divider(page)).to_have_count(0)


def _empty_part_of_tab_bar(strip: Locator) -> tuple[float, float]:
    """A point on the bar holding `strip`, past its tabs and New tab button, where a drop joins."""
    bar = strip.locator("xpath=ancestor::div[@data-panel-tab-bar]").bounding_box()
    tabs = strip.bounding_box()
    return (tabs["x"] + tabs["width"] + bar["x"] + bar["width"]) / 2, bar["y"] + bar["height"] / 2


def test_the_divider_resizes_the_panes_between_a_fifth_and_four_fifths(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    page.get_by_role("button", name="Split screen").click()
    _expect_panes(page, [["New Chat"], [FILES]])
    divider = _divider(page)
    expect(divider).to_have_attribute("aria-valuenow", "50")
    half = page.get_by_role("tabpanel", name="New Chat").bounding_box()["width"]

    _drag_divider_to(page, 0.3)
    _expect_ratio(divider, 30)
    narrow = page.get_by_role("tabpanel", name="New Chat").bounding_box()["width"]
    assert narrow < half * 0.7, f"the conversation is still {narrow}px of {half}px"

    _drag_divider_to(page, 0.02)
    expect(divider).to_have_attribute("aria-valuenow", "20")
    _drag_divider_to(page, 0.98)
    expect(divider).to_have_attribute("aria-valuenow", "80")
    divider.focus()
    page.keyboard.press("ArrowLeft")
    expect(divider).to_have_attribute("aria-valuenow", "75")


def _drag_divider_to(page: Page, fraction: float) -> None:
    """Drag the divider to that fraction of the width the two side-by-side panes share."""
    shown = [
        box for box in (panel.bounding_box() for panel in page.get_by_role("tabpanel").all()) if box
    ]
    start = min(box["x"] for box in shown)
    end = max(box["x"] + box["width"] for box in shown)
    middle = shown[0]["y"] + shown[0]["height"] / 2
    _drag_to(page, _divider(page), start + (end - start) * fraction, middle)


def _expect_ratio(divider: Locator, percent: int) -> None:
    expect(divider).to_have_attribute(
        "aria-valuenow", re.compile("|".join(str(percent + step) for step in (-2, -1, 0, 1, 2)))
    )


def test_the_split_layout_is_kept_after_a_reload_and_on_another_device(
    terminal_account, page_for, admin, upstream, config
):
    person, terminal = terminal_account
    page = page_for(person)
    chat_id = _answered_chat(page, upstream, terminal)
    title = _tab_strip(page).get_by_role("tab").first.inner_text().strip()
    page.keyboard.press("Control+Backslash")
    _tab_strip(page).get_by_role("tab", name=FILES).click()
    page.keyboard.press("Control+Backquote")
    layout = [[title], [FILES, "Terminal 1"]]
    _expect_panes(page, layout)
    _selected(page, "Terminal 1")
    _drag_divider_to(page, 0.3)
    ratio = _divider(page).get_attribute("aria-valuenow")
    assert ratio != "50", "dragging the divider left the panes at half and half"

    # wait for the server to hold the layout before reloading; a drag slower than the saves can
    # wedge the saving on dev f659c0d12 (open-webui/open-webui#33892)
    held = (int(ratio), "Terminal 1")
    _wait_until(lambda: _layout_of(_saved_view(person, chat_id)) == held)
    assert _layout_of(_saved_view(person, chat_id)) == held, _saved_view(person, chat_id)

    # the sidebar row under the pointer previews the chat, so the reply is looked for in main
    answer = page.get_by_role("main").get_by_text("Start at the lighthouse.")
    # leaving for a new chat and coming back through the sidebar brings the layout back too
    page.keyboard.press("Control+Shift+O")
    expect(page.get_by_role("tab", name="New Chat", exact=True)).to_be_visible()
    expect(answer).to_have_count(0)
    _open_from_sidebar(page, title)
    expect(answer).to_be_visible()
    _expect_panes(page, layout)
    _selected(page, "Terminal 1")
    expect(_divider(page)).to_have_attribute("aria-valuenow", ratio)

    page.reload()
    # a reload is a cold load, which page_for would otherwise wait out
    expect(answer).to_be_visible(timeout=config.navigation_timeout)
    _expect_panes(page, layout)
    _selected(page, "Terminal 1")
    expect(_divider(page)).to_have_attribute("aria-valuenow", ratio)

    elsewhere = page_for(person, path=f"/c/{chat_id}")
    _expect_panes(elsewhere, layout)
    _selected(elsewhere, "Terminal 1")
    expect(_divider(elsewhere)).to_have_attribute("aria-valuenow", ratio)

    # the layout is the person's own: an admin opening the chat gets the conversation alone
    overseer = page_for(admin, path=f"/c/{chat_id}")
    expect(overseer.get_by_text("Start at the lighthouse.")).to_be_visible()
    _expect_panes(overseer, [[title]])
    expect(_divider(overseer)).to_have_count(0)


def test_a_new_chat_started_beside_an_open_shell_sends_its_first_message(
    terminal_account, page_for, upstream
):
    """Red on dev c15f28649 (open-webui/open-webui#34088)."""
    person, terminal = terminal_account
    page = page_for(person)
    _answered_chat(page, upstream, terminal)
    # a shell opened in the saved chat
    page.keyboard.press("Control+Backquote")
    expect(page.get_by_role("tabpanel", name="Terminal 1")).to_be_visible()

    page.keyboard.press("Control+Shift+O")
    conversation = page.get_by_role("tab", name="New Chat", exact=True)
    conversation.click()
    expect(page.get_by_text("Start at the lighthouse.")).to_have_count(0)
    question = f"where do the ferries leave {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("From the east pier.", match=reply.answering(question)))
    send(page, question)

    def asked() -> bool:
        return any(map(reply.answering(question), upstream.chat_requests()))

    _wait_until(asked)
    assert asked(), (
        "the new chat's first message never reached the model; the page says "
        f"{page.get_by_role('alert').all_inner_texts()}"
    )
    expect_reply(page, "From the east pier.")


def test_a_layout_arranged_before_the_first_message_is_kept_with_the_chat(
    terminal_account, page_for, upstream, config
):
    person, terminal = terminal_account
    page = page_for(person)
    pick_terminal(page, terminal)
    page.keyboard.press("Control+Backslash")
    _expect_panes(page, [["New Chat"], [FILES]])
    question = f"where do the ferries leave {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("From the east pier.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "From the east pier.")
    expect(page).to_have_url(re.compile(r"/c/"))
    chat_id = page.url.rsplit("/", 1)[-1]

    # the conversation took the focus to send, so it is the active tab
    held = (50, "New Chat")
    _wait_until(lambda: _layout_of(_saved_view(person, chat_id)) == held)
    assert _layout_of(_saved_view(person, chat_id)) == held, _saved_view(person, chat_id)
    page.reload()
    expect(page.get_by_text("From the east pier.")).to_be_visible(timeout=config.navigation_timeout)
    title = _panes(page)[0][0]
    _expect_panes(page, [[title], [FILES]])


def test_layout_changes_after_a_network_reconnect_are_still_saved(
    terminal_account, page_for, upstream, config
):
    """Red on dev f659c0d12: a save with nothing pending, as the back-online event starts, wedges
    the chat's layout saving until the next reload (open-webui/open-webui#33892)."""
    person, terminal = terminal_account
    page = page_for(person)
    chat_id = _answered_chat(page, upstream, terminal)
    page.get_by_role("button", name="Split screen").click()
    expect(_divider(page)).to_have_attribute("aria-valuenow", "50")
    _wait_until(lambda: (_layout_of(_saved_view(person, chat_id)) or (0,))[0] == 50)
    assert (_layout_of(_saved_view(person, chat_id)) or (0,))[0] == 50, "the split was not saved"

    # the browser reports the network back, with the layout already saved
    page.evaluate("window.dispatchEvent(new Event('online'))")
    _divider(page).focus()
    for _ in range(3):
        page.keyboard.press("ArrowLeft")
    expect(_divider(page)).to_have_attribute("aria-valuenow", "35")
    _wait_until(lambda: (_layout_of(_saved_view(person, chat_id)) or (0,))[0] == 35)
    saved = _layout_of(_saved_view(person, chat_id))
    assert saved and saved[0] == 35, f"the divider moved to 35% but the server holds {saved}"


def _layout_of(view: dict | None) -> tuple[int, str] | None:
    if not view or view["panels"]["layout"]["type"] != "split":
        return None
    labels = {tab["id"]: tab["label"] for tab in view["panels"]["tabs"]}
    percent = round(view["panels"]["layout"]["ratio"] * 100)
    return percent, labels.get(view["panels"]["activeId"], "")


def test_the_workspace_shortcuts_open_step_through_and_close_tabs(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    _expect_panes(page, [["New Chat", FILES]])

    page.keyboard.press("Control+W")
    _expect_panes(page, [["New Chat"]])
    page.keyboard.press("Control+Shift+E")
    _expect_panes(page, [["New Chat", FILES]])
    _selected(page, FILES)
    page.keyboard.press("Control+Backquote")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1"]])
    _selected(page, "Terminal 1")
    page.keyboard.press("Control+Alt+N")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1"]])
    _selected(page, "Untitled-1")
    page.keyboard.press("Control+Shift+B")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1", "Browser"]])
    _selected(page, "Browser")
    # a close waits until the tab's session has started, so start it before Ctrl+W below
    expect(browser_view(page)).to_be_visible(timeout=BROWSER_START_MS)

    page.keyboard.press("Control+Shift+BracketLeft")
    _selected(page, "Untitled-1")
    page.keyboard.press("Control+Shift+BracketRight")
    _selected(page, "Browser")
    page.keyboard.press("Control+Shift+BracketRight")
    _selected(page, "New Chat")
    # the conversation never closes, and this terminal offers no desktop
    page.keyboard.press("Control+W")
    page.keyboard.press("Control+Alt+D")
    page.keyboard.press("Control+Shift+BracketLeft")
    _selected(page, "Browser")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1", "Browser"]])
    page.keyboard.press("Control+W")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1"]])


def test_workspace_shortcuts_do_nothing_while_a_menu_or_a_dialog_is_open(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)

    page.get_by_role("button", name="New tab", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name=re.compile("^New Terminal"))).to_be_visible()
    page.keyboard.press("Control+Backquote")
    page.keyboard.press("Control+Backslash")
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)
    # live again: a shell the menu had let through would make this one Terminal 2
    page.keyboard.press("Control+Backquote")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1"]])

    page.keyboard.press("Control+Period")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", name="General")).to_be_visible()
    page.keyboard.press("Control+Alt+N")
    page.keyboard.press("Control+W")
    page.keyboard.press("Escape")
    expect(settings).to_have_count(0)
    page.keyboard.press("Control+Alt+N")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1"]])


def test_workspace_shortcuts_do_nothing_with_keyboard_shortcuts_switched_off(
    terminal_account, page_for
):
    person, terminal = terminal_account
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"keyboardShortcuts": False}}
        )
        assert saved.status_code == 200, saved.text
    page = page_for(person)
    pick_terminal(page, terminal)

    page.keyboard.press("Control+Backquote")
    page.keyboard.press("Control+Backslash")
    page.keyboard.press("Control+W")
    # the menu's shell comes after the shortcuts, so it is Terminal 2 if one of them answered
    open_new_terminal(page)
    _expect_panes(page, [["New Chat", FILES, "Terminal 1"]])


def _open_git_panel(page: Page, terminal: str, folder) -> Locator:
    """Pick the terminal, open `folder` in its file browser and expand the Git panel under it."""
    choose_terminal(page, terminal)
    # the conversation stays in front; Files opens from the New tab menu
    _add_tab(page, re.compile(r"^Files"))
    # the panel shows once the file browser stands in the repository
    return expand_git_panel(page, folder.name)


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


def _hold(page: Page, operation: str) -> list:
    """Hold the page's requests for the Git `operation` until the test lets them go."""
    held = []
    page.route(re.compile(rf"/git/{operation}(\?|$)"), lambda route: held.append(route))
    return held


def _let_go(held: list) -> None:
    assert held, "the page sent no request to let go"
    for route in held:
        route.continue_()


def test_the_git_panel_says_what_it_is_doing_while_staging_and_pushing(
    terminal_page, repository, remote
):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    activity = git.get_by_role("status")

    held = _hold(page, "stage")
    git.get_by_role("checkbox", name="Stage tides.txt").click()
    expect(activity).to_have_text("Staging…")
    _let_go(held)
    expect(git.get_by_role("checkbox", name="Unstage tides.txt")).to_be_visible()
    expect(activity).not_to_have_text("Staging…")

    git.get_by_label("Commit summary").fill("Push it slowly")
    git.get_by_role("button", name="Commit 1 file to main").click()
    expect(git.get_by_text("No changes")).to_be_visible()
    held = _hold(page, "push")
    git.get_by_role("button", name="Push", exact=True).click()
    expect(activity).to_have_text("Pushing…")
    assert _git(remote, "log", "-1", "--format=%s", "main").strip() != "Push it slowly"
    _let_go(held)
    expect(git.get_by_role("button", name="Fetch", exact=True)).to_be_visible()
    expect(activity).not_to_have_text("Pushing…")
    assert _git(remote, "log", "-1", "--format=%s", "main").strip() == "Push it slowly"


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


def _set_comments(admin, enabled: bool) -> None:
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_COMMENTS": enabled}
        )
    assert saved.status_code == 200, saved.text


def _answered(page: Page, upstream) -> str:
    """Ask one question in the page's new chat; returns the conversation tab's name."""
    question = f"name a quiet beach {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Try the north cove.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Try the north cove.")
    expect(page).to_have_url(re.compile(r"/c/"))
    return _tab_strip(page).get_by_role("tab").first.inner_text().strip()


def test_side_chat_and_comments_shortcuts_answer_only_where_allowed(
    page_for, make_user, admin, preserve, upstream
):
    preserve("admin_config")
    preserve("permissions")
    _set_comments(admin, False)
    with admin.client() as client:
        defaults = client.get("/api/v1/users/default/permissions").json()
        defaults["chat"]["side_chat"] = False
        saved = client.post("/api/v1/users/default/permissions", json=defaults)
    assert saved.status_code == 200, saved.text
    page = page_for(make_user())
    title = _answered(page, upstream)
    page.keyboard.press("Control+Alt+O")
    page.keyboard.press("Control+Shift+Y")
    # the search opens after both chords were handled
    page.keyboard.press("Control+K")
    expect(page.get_by_role("dialog").get_by_placeholder("Search")).to_be_visible()
    _expect_panes(page, [[title]])

    _set_comments(admin, True)
    allowed = make_user()
    make_group(admin, [allowed], {"chat": {"side_chat": True}})
    page = page_for(allowed)
    title = _answered(page, upstream)
    page.keyboard.press("Control+Alt+O")
    _expect_panes(page, [[title, "New Chat 1"]])
    page.keyboard.press("Control+Shift+Y")
    _expect_panes(page, [[title, "New Chat 1", "Comments"]])
    _selected(page, "Comments")


def _files_in_a_folder_of_harbour(
    page_for, make_user, admin, open_terminal, other: dict, person: Actor | None = None
) -> tuple[Page, Locator, str]:
    """A new chat on Harbour of `person` (else a fresh account), with `other` open to it too, its
    Files tab in a folder of Harbour and split beside the conversation; returns the page, the
    file browser and the folder's name."""
    person = person or make_user()
    grants = [read_grant(person.id)]
    harbour = open_terminal.connection(name="Harbour", config={"access_grants": grants})
    other = {**other, "config": {**other["config"], "access_grants": grants}}
    with admin.client() as client:
        configure_terminals(client, harbour, other)
    dock = open_terminal.home / f"dock-{uuid.uuid4().hex[:6]}"
    dock.mkdir()
    (dock / "manifest.txt").write_text("two ropes\n")
    page = page_for(person)
    pick_terminal(page, "Harbour")
    files = page.get_by_role("region", name="File browser")
    files.get_by_text(dock.name, exact=True).click()
    expect(files.get_by_text("manifest.txt", exact=True)).to_be_visible()
    expect(_tab_strip(page).get_by_role("tab", name=dock.name, exact=True)).to_be_visible()
    page.keyboard.press("Control+Backslash")
    expect(chat_input(page)).to_be_visible()
    return page, files, dock.name


def _pick_another_terminal(page: Page, current: str, new: str) -> None:
    # once picked, the composer's terminal button is named after the terminal
    composer = page.get_by_role("main")
    composer.get_by_role("button", name=current, exact=True).click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=new).click()
    expect(composer.get_by_role("button", name=new, exact=True)).to_be_visible()


def test_picking_another_terminal_before_the_first_message_shows_its_files(
    page_for, make_user, admin, preserve, open_terminal, second_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    lighthouse = second_terminal.connection(name="Lighthouse")
    page, files, folder = _files_in_a_folder_of_harbour(
        page_for, make_user, admin, open_terminal, lighthouse
    )

    _pick_another_terminal(page, "Harbour", "Lighthouse")

    expect(files.get_by_text("tide-chart.csv", exact=True)).to_be_visible()
    expect(files.get_by_text("manifest.txt", exact=True)).to_have_count(0)
    expect(_tab_strip(page).get_by_role("tab", name=FILES, exact=True)).to_be_visible()
    expect(_tab_strip(page).get_by_role("tab", name=folder, exact=True)).to_have_count(0)


def test_picking_another_terminal_in_a_chat_with_messages_shows_its_files(
    page_for, make_user, admin, preserve, upstream, open_terminal, second_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    lighthouse = second_terminal.connection(name="Lighthouse")
    page, files, folder = _files_in_a_folder_of_harbour(
        page_for, make_user, admin, open_terminal, lighthouse
    )
    prompt = f"read the manifest {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Two ropes.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Two ropes.")

    _pick_another_terminal(page, "Harbour", "Lighthouse")

    expect(files.get_by_text("tide-chart.csv", exact=True)).to_be_visible()
    expect(files.get_by_text("manifest.txt", exact=True)).to_have_count(0)
    expect(_tab_strip(page).get_by_role("tab", name=FILES, exact=True)).to_be_visible()
    expect(_tab_strip(page).get_by_role("tab", name=folder, exact=True)).to_have_count(0)


def test_a_per_chat_terminal_picked_before_the_first_message_shows_its_files_once_the_chat_starts(
    page_for, make_user, admin, upstream, open_terminal, connect_terminals
):
    person = make_user()
    # Lighthouse is a terminal on a Terminals orchestrator that gives every chat its own machine
    orchestrator = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        lighthouse = create_template(client, orchestrator, name="Lighthouse", scope="chat")
        share_template(client, lighthouse["terminal_id"], read_grant(person.id))
    page, files, folder = _files_in_a_folder_of_harbour(
        page_for, make_user, admin, open_terminal, orchestrator, person
    )

    _pick_another_terminal(page, "Harbour", "Lighthouse")
    # a per-chat terminal has no files until its chat exists
    expect(page.get_by_text("Terminal connection unavailable")).to_be_visible()
    prompt = f"chart the coast {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Charted.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Charted.")

    # the chat's own machine starts with an empty home
    expect(_tab_strip(page).get_by_role("tab", name=FILES, exact=True)).to_be_visible()
    expect(files.get_by_text("This folder is empty")).to_be_visible()
    expect(files.get_by_text("manifest.txt", exact=True)).to_have_count(0)
    expect(_tab_strip(page).get_by_role("tab", name=folder, exact=True)).to_have_count(0)


def _diff_setting(git: Locator, name: str) -> Locator:
    """Open the Git panel's Diff settings and return the entry `name` of its menu."""
    git.get_by_role("button", name="Diff settings").click()
    entry = git.page.get_by_role("menu").get_by_role("menuitem", name=name)
    expect(entry).to_be_visible()
    return entry


def _place(line: Locator) -> tuple[float, float]:
    box = line.bounding_box()
    return box["x"], box["y"]


def test_a_diff_is_shown_side_by_side_and_hides_lines_that_differ_only_in_spacing(
    terminal_page, repository
):
    page, terminal = terminal_page
    (repository / "tides.txt").write_text("high water at seven\nlow  water at   noon\n")
    git = _open_git_panel(page, terminal, repository)
    git.get_by_role("button", name="View changes tides.txt").first.click()
    before = git.get_by_text("high water at six")
    after = git.get_by_text("high water at seven")
    respaced = git.get_by_text("low water at noon")
    expect(after).to_be_visible()
    expect(respaced).to_have_count(2)
    (old_x, old_y), (new_x, new_y) = _place(before), _place(after)
    assert new_y > old_y + 5 and abs(new_x - old_x) < 2, (
        f"the unified diff has the old line at {(old_x, old_y)} and the new one at {(new_x, new_y)}"
    )

    _diff_setting(git, "Split").click()
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    (old_x, old_y), (new_x, new_y) = _place(before), _place(after)
    assert abs(new_y - old_y) < 2 and new_x > old_x + 50, (
        f"the split diff has the old line at {(old_x, old_y)} and the new one at {(new_x, new_y)}"
    )

    hide = _diff_setting(git, "Hide whitespace")
    expect(hide).to_have_attribute("aria-pressed", "false")
    hide.click()
    expect(hide).to_have_attribute("aria-pressed", "true")
    expect(respaced).to_have_count(0)
    expect(after).to_be_visible()
    hide.click()
    expect(respaced).to_have_count(2)


def _branch_actions(menu: Locator, branch: str) -> Locator:
    """Open the actions of `branch` in the open branch menu; returns that menu."""
    more = menu.get_by_role("menuitem", name=f"Branch actions {branch}")
    more.click()
    actions = menu.page.get_by_role("menu").filter(
        has=menu.page.get_by_role("menuitem", name="Copy branch name")
    )
    expect(actions).to_be_visible()
    return actions


def _branches(folder) -> set[str]:
    return set(_git(folder, "branch", "--format=%(refname:short)").split())


def test_branches_are_searched_renamed_copied_and_deleted_from_the_branch_menu(
    terminal_account, page_for, repository
):
    person, terminal = terminal_account
    _git(repository, "branch", "spring-tides")
    _git(repository, "branch", "neap-tides")
    page = page_for(person)
    clipboard = loopback_clipboard(page, person)
    git = _open_git_panel(page, terminal, repository)

    menu = _branch_menu(git)
    expect(menu.get_by_role("menuitem", name="spring-tides", exact=True)).to_be_visible()
    menu.get_by_role("textbox", name="Search branches").fill("NEAP")
    expect(menu.get_by_role("menuitem", name="spring-tides", exact=True)).to_have_count(0)
    expect(menu.get_by_role("menuitem", name="main", exact=True)).to_have_count(0)
    expect(menu.get_by_role("menuitem", name="neap-tides", exact=True)).to_be_visible()
    menu.get_by_role("textbox", name="Search branches").fill("")

    _branch_actions(menu, "spring-tides").get_by_role("menuitem", name="Copy branch name").click()
    assert copied(clipboard) == "spring-tides", f"the clipboard holds {copied(clipboard)!r}"

    _branch_actions(menu, "spring-tides").get_by_role("menuitem", name="Rename").click()
    dialog = page.get_by_role("dialog").filter(has_text="Rename branch")
    name = dialog.get_by_role("textbox")
    expect(name).to_have_value("spring-tides")
    name.fill("autumn-tides")
    dialog.get_by_role("button", name="Confirm").click()
    expect(dialog).to_have_count(0)
    _wait_until(lambda: "autumn-tides" in _branches(repository))
    assert _branches(repository) == {"main", "autumn-tides", "neap-tides"}, (
        f"after the rename the repository has the branches {_branches(repository)}"
    )

    menu = _branch_menu(git)
    expect(menu.get_by_role("menuitem", name="autumn-tides", exact=True)).to_be_visible()
    # the branch that is checked out cannot be deleted
    expect(_branch_actions(menu, "main").get_by_role("menuitem", name="Delete")).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    menu = _branch_menu(git)
    _branch_actions(menu, "neap-tides").get_by_role("menuitem", name="Delete").click()
    dialog = page.get_by_role("dialog").filter(has_text="Delete branch")
    expect(dialog).to_contain_text("Delete branch neap-tides? Unmerged branches will be kept.")
    dialog.get_by_role("button", name="Confirm").click()
    expect(dialog).to_have_count(0)
    _wait_until(lambda: "neap-tides" not in _branches(repository))
    assert _branches(repository) == {"main", "autumn-tides"}, (
        f"after the delete the repository has the branches {_branches(repository)}"
    )
    menu = _branch_menu(git)
    expect(menu.get_by_role("menuitem", name="autumn-tides", exact=True)).to_be_visible()
    expect(menu.get_by_role("menuitem", name="neap-tides", exact=True)).to_have_count(0)


def test_a_branch_with_commits_of_its_own_is_kept_when_deleted_and_the_panel_says_why(
    terminal_page, repository
):
    page, terminal = terminal_page
    _git(repository, "stash", "-q")
    _git(repository, "switch", "-q", "-c", "storm-tides")
    (repository / "storm.txt").write_text("surge at midnight\n")
    _git(repository, "add", "storm.txt")
    _git(repository, "commit", "-q", "-m", "Log the storm surge")
    _git(repository, "switch", "-q", "main")
    git = _open_git_panel(page, terminal, repository)

    menu = _branch_menu(git)
    _branch_actions(menu, "storm-tides").get_by_role("menuitem", name="Delete").click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()

    expect(git.get_by_role("alert")).to_contain_text("not fully merged")
    assert "storm-tides" in _branches(repository), "the unmerged branch was deleted"


def _worktrees_menu(git: Locator) -> Locator:
    git.get_by_role("button", name="Worktrees").click()
    search = git.page.get_by_role("textbox", name="Search worktrees")
    expect(search).to_be_visible()
    return git.page.get_by_role("menu").filter(has=search)


def test_a_worktree_made_in_the_worktrees_menu_opens_in_the_files_tab(terminal_page, repository):
    page, terminal = terminal_page
    night = repository.parent / f"{repository.name}-night"
    git = _open_git_panel(page, terminal, repository)
    files = page.get_by_role("region", name="File browser")
    try:
        menu = _worktrees_menu(git)
        # the worktree the file browser stands in is listed and cannot be picked
        expect(menu.get_by_role("menuitem", name=re.compile(r"^main\b"))).to_be_disabled()
        menu.get_by_role("menuitem", name="New worktree").click()
        create = menu.get_by_role("menuitem", name="Create worktree")
        expect(create).to_be_disabled()
        menu.get_by_role("textbox", name="Worktree branch").fill("night-tides")
        menu.get_by_role("textbox", name="Worktree directory").fill(f"../{night.name}")
        create.click()

        # the new worktree opens in the Files tab, on its own branch and with the committed file
        expect(_tab_strip(page).get_by_role("tab", name=night.name, exact=True)).to_be_visible()
        expect(files.get_by_text("tides.txt", exact=True)).to_be_visible()
        expect(git.get_by_role("button", name="Switch branch")).to_have_text("night-tides")
        assert (night / "tides.txt").read_text() == "high water at six\nlow water at noon\n", (
            "the worktree does not hold the committed file"
        )
        assert _git(night, "branch", "--show-current").strip() == "night-tides"

        menu = _worktrees_menu(git)
        expect(menu.get_by_role("menuitem", name=re.compile(r"^night-tides\b"))).to_be_disabled()
        menu.get_by_role("textbox", name="Search worktrees").fill(night.name)
        expect(menu.get_by_role("menuitem", name=re.compile(r"^main\b"))).to_have_count(0)
        menu.get_by_role("textbox", name="Search worktrees").fill("")
        menu.get_by_role("menuitem", name=re.compile(r"^main\b")).click()
        expect(
            _tab_strip(page).get_by_role("tab", name=repository.name, exact=True)
        ).to_be_visible()
        expect(git.get_by_role("button", name="Switch branch")).to_have_text("main")
    finally:
        shutil.rmtree(night, ignore_errors=True)


def test_commits_on_the_remote_are_fetched_and_then_pulled_from_the_sync_button(
    terminal_page, repository, remote
):
    page, terminal = terminal_page
    elsewhere = repository.parent / f"{repository.name}-elsewhere"
    _git(repository.parent, "clone", "-q", str(remote), str(elsewhere))
    try:
        _git(elsewhere, "config", "user.name", "Ben Quay")
        _git(elsewhere, "config", "user.email", "ben@example.com")
        (elsewhere / "moon.txt").write_text("full moon on friday\n")
        _git(elsewhere, "add", "moon.txt")
        _git(elsewhere, "commit", "-q", "-m", "Note the full moon")
        _git(elsewhere, "push", "-q", "origin", "main")
    finally:
        shutil.rmtree(elsewhere, ignore_errors=True)
    git = _open_git_panel(page, terminal, repository)

    git.get_by_role("button", name="Fetch", exact=True).click()
    expect(git.get_by_role("button", name="Pull", exact=True)).to_be_visible()
    expect(git.get_by_text("↓1", exact=True)).to_be_visible()
    assert not (repository / "moon.txt").exists(), "Fetch already changed the working tree"

    git.get_by_role("button", name="Pull", exact=True).click()
    expect(git.get_by_role("button", name="Fetch", exact=True)).to_be_visible()
    expect(git.get_by_text("↓1", exact=True)).to_have_count(0)
    assert (repository / "moon.txt").read_text() == "full moon on friday\n", (
        "Pull did not bring the remote's commit into the working tree"
    )
    # the change that was not committed is still there
    assert (repository / "tides.txt").read_text().startswith("high water at seven")


def test_history_shows_a_commits_changes_marks_what_is_not_pushed_and_copies_its_hash(
    terminal_account, page_for, repository, remote
):
    person, terminal = terminal_account
    _git(repository, "commit", "-q", "-a", "-m", "Move high water to seven")
    newest = _git(repository, "log", "-1", "--format=%H").strip()
    page = page_for(person)
    clipboard = loopback_clipboard(page, person)
    git = _open_git_panel(page, terminal, repository)
    git.get_by_role("button", name="History").click()
    unpushed = git.get_by_title("Unpushed")
    expect(git.get_by_text("Start the tide log")).to_be_visible()
    expect(unpushed).to_have_count(1)

    git.get_by_text("Start the tide log").click()
    expect(git.get_by_text("low water at noon")).to_be_visible()
    expect(git.get_by_text("high water at seven")).to_have_count(0)
    git.get_by_text("Move high water to seven").click()
    expect(git.get_by_text(f"{newest[:7]} · Move high water to seven")).to_be_visible()
    expect(git.get_by_text("high water at seven")).to_be_visible()
    expect(git.get_by_text("high water at six")).to_be_visible()

    git.get_by_role("button", name=f"Commit actions {newest[:7]}").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Copy commit hash").click()
    assert copied(clipboard) == newest, f"the clipboard holds {copied(clipboard)!r}, not {newest}"

    git.get_by_role("button", name="Push", exact=True).click()
    expect(git.get_by_role("button", name="Fetch", exact=True)).to_be_visible()
    expect(unpushed).to_have_count(0)
    expect(git.get_by_text("Move high water to seven").first).to_be_visible()


REMOTES = [
    ("git@github.com:example/logbook.git", "GitHub", "https://github.com/example/logbook"),
    (
        "https://ada:hunter2@gitlab.example.com/harbour/logbook.git",
        "GitLab",
        "https://gitlab.example.com/harbour/logbook",
    ),
    (
        "ssh://git@bitbucket.org/example/logbook.git",
        "Bitbucket",
        "https://bitbucket.org/example/logbook",
    ),
    ("https://code.example.com/harbour/logbook", "code.example.com", None),
]


def test_the_panel_links_to_the_repositorys_page_on_its_host_without_credentials(
    terminal_page, repository
):
    page, terminal = terminal_page
    git = _open_git_panel(page, terminal, repository)
    link = git.get_by_role("link", name=re.compile("^View on "))
    expect(git.get_by_role("button", name="Diff settings")).to_be_visible()
    expect(link).to_have_count(0)

    for number, (remote_url, host, address) in enumerate(REMOTES):
        _git(repository, "remote", "set-url" if number else "add", "origin", remote_url)
        # the panel reads the repository again when it is opened
        git.get_by_role("button", name="Collapse Git panel").click()
        git.get_by_role("button", name="Expand Git panel").click()
        expect(link).to_have_accessible_name(f"View on {host}")
        expect(link).to_have_attribute("href", address or f"{remote_url}")
        expect(link).to_have_attribute("target", "_blank")


SCRIPTED_HOME = "/home/user"
SCRIPTED_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Open Terminal", "version": "1"},
    "paths": {},
}
NOT_A_REPOSITORY = {"is_repo": False, "branch": "", "files": [], "ahead": 0, "behind": 0}


def _repository_status(*changed: str) -> dict:
    files = [
        {
            "path": path,
            "status": "modified",
            "staged": False,
            "unstaged": True,
            "additions": 1,
            "deletions": 1,
        }
        for path in changed
    ]
    return {
        "is_repo": True,
        "root": SCRIPTED_HOME,
        "branch": "main",
        "head": "0" * 40,
        "upstream": "origin/main",
        "ahead": 0,
        "behind": 0,
        "files": files,
        "remote_url": "",
    }


@pytest.fixture
def scripted_terminal(admin, preserve):
    """`scripted_terminal(name, person, features=...)` connects a terminal that answers the file
    browser from a script, with one file in its home, for the person to use; returns the server.
    `features=None` makes its configuration fail."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connections = []
    with contextlib.ExitStack() as running:

        def connect(name: str, person: Actor, features: dict | None = None):
            server = running.enter_context(serving_terminal())
            config = json_answer({"features": features}) if features is not None else (500, {}, b"")
            server.route("GET", "/openapi.json", json_answer(SCRIPTED_SPEC))
            server.route("GET", "/api/config", config)
            server.route(
                "GET", "/files/cwd", json_answer({"cwd": SCRIPTED_HOME, "home": SCRIPTED_HOME})
            )
            listed = [{"name": f"{name.lower()}-notes.md", "type": "file", "size": 12}]
            server.route(
                "GET",
                "/files/list",
                json_answer({"dir": SCRIPTED_HOME, "entries": listed, "writable": True}),
            )
            server.route("GET", "/git/status", json_answer(_repository_status()))
            server.route("GET", "/git/pr/capabilities", json_answer({"available": False}))
            connections.append(
                server.connection(name=name, config={"access_grants": [read_grant(person.id)]})
            )
            with admin.client() as client:
                configure_terminals(client, *connections)
            return server

        yield connect


def _new_tab_entries(page: Page) -> list[str]:
    """What the open New tab menu offers, without the shortcut beside each entry."""
    entries = page.get_by_role("menu").get_by_role("menuitem").all_inner_texts()
    return [entry.strip().splitlines()[0].strip() for entry in entries]


def _expect_new_tab_entries(page: Page, expected: list[str]) -> None:
    # of the pane that holds the Files tab; the conversation's pane has a New tab button too
    files_view = page.get_by_role("tabpanel").filter(
        has=page.get_by_role("region", name="File browser")
    )
    files_tab = page.locator(f'[id="{files_view.get_attribute("aria-labelledby")}"]')
    files_pane = page.locator("[data-panel-tab-bar]").filter(has=files_tab)
    files_pane.get_by_role("button", name="New tab", exact=True).click()
    expect(page.get_by_role("menu").get_by_role("menuitem").first).to_be_visible()
    # the entries follow what the terminal answers about itself, shortly after it is picked
    _wait_until(lambda: _new_tab_entries(page) == expected)
    assert _new_tab_entries(page) == expected, (
        f"the New tab menu offers {_new_tab_entries(page)}, expected {expected}"
    )
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)


def _watched(terminal) -> list:
    """The live-update connections the page opened on the scripted terminal."""
    return [session for session in terminal.sessions if session.path.split("?")[0] == "/events"]


def test_the_new_tab_menu_and_the_git_panel_offer_what_the_picked_terminal_can_do(
    page_for, make_user, scripted_terminal
):
    person = make_user()
    everything = {"terminal": True, "browser": True, "desktop": True, "git": True, "events": True}
    plain = scripted_terminal("Plain", person, features={})
    workstation = scripted_terminal("Workstation", person, features=everything)
    locked = scripted_terminal(
        "Locked", person, features={"terminal": False, "git": False, "events": False}
    )
    scripted_terminal("Silent", person, features=None)
    page = page_for(person)
    files = page.get_by_role("region", name="File browser")
    git = page.get_by_role("region", name="Git")

    pick_terminal(page, "Plain")
    # side by side, so the chat input's Terminal menu stays in reach
    page.keyboard.press("Control+Backslash")
    expect(chat_input(page)).to_be_visible()
    expect(files.get_by_text("plain-notes.md", exact=True)).to_be_visible()
    _expect_new_tab_entries(page, ["Files", "New File", "New Terminal", "Voice Note"])
    # a terminal that does not say it has Git is never asked about a repository
    assert not plain.requests_to("/git/status"), "a terminal without Git was asked for its status"

    _pick_another_terminal(page, "Plain", "Workstation")
    expect(files.get_by_text("workstation-notes.md", exact=True)).to_be_visible()
    _expect_new_tab_entries(
        page, ["Files", "New File", "New Terminal", "Browser", "Desktop", "Voice Note"]
    )
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("main")
    _wait_until(lambda: _watched(workstation))
    assert _watched(workstation), "the terminal with live updates was not asked for them"

    _pick_another_terminal(page, "Workstation", "Locked")
    expect(files.get_by_text("locked-notes.md", exact=True)).to_be_visible()
    _expect_new_tab_entries(page, ["Files", "New File", "Voice Note"])
    expect(git).to_have_count(0)
    assert not locked.requests_to("/git/status"), "a terminal with Git off was asked for its status"
    assert not _watched(locked), "a terminal with live updates off was asked for them"

    # a terminal whose configuration cannot be read still shows its files
    _pick_another_terminal(page, "Locked", "Silent")
    expect(files.get_by_text("silent-notes.md", exact=True)).to_be_visible()
    _expect_new_tab_entries(page, ["Files", "New File", "Voice Note"])
    expect(git).to_have_count(0)


def test_a_repository_the_terminal_cannot_read_says_git_unavailable_until_a_retry_works(
    page_for, make_user, scripted_terminal
):
    person = make_user()
    terminal = scripted_terminal("Harbour", person, features={"git": True})
    terminal.route("GET", "/git/status", json_answer({"detail": "git: command not found"}, 500))
    page = page_for(person)
    pick_terminal(page, "Harbour")
    git = page.get_by_role("region", name="Git")
    unavailable = page.get_by_text("Git unavailable", exact=True)

    expect(unavailable).to_be_visible()
    expect(git).to_have_count(0)
    unavailable.hover()
    expect(page.get_by_role("tooltip").filter(has_text="git: command not found")).to_be_visible()

    # an Open Terminal from before the Git panel does not know the route
    terminal.route("GET", "/git/status", json_answer({"detail": "Not Found"}, 404))
    asked = len(terminal.requests_to("/git/status"))
    page.get_by_role("button", name="Retry Git detection").click()
    _wait_until(lambda: len(terminal.requests_to("/git/status")) > asked)
    unavailable.hover()
    expect(
        page.get_by_role("tooltip").filter(
            has_text="Git support is unavailable on this connection. "
            "Update and restart Open Terminal."
        )
    ).to_be_visible()

    terminal.route("GET", "/git/status", json_answer(_repository_status("tides.txt")))
    page.get_by_role("button", name="Retry Git detection").click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("main")
    expect(git.get_by_role("button", name="Git changes")).to_contain_text("1 changed")
    expect(unavailable).to_have_count(0)


def _settled(read, quiet: float = 1.5, limit: float = 15.0):
    """The value `read` gives once it has stopped changing for `quiet` seconds."""
    deadline = time.monotonic() + limit
    value, since = read(), time.monotonic()
    while time.monotonic() < deadline and time.monotonic() - since < quiet:
        time.sleep(0.1)
        if (current := read()) != value:
            value, since = current, time.monotonic()
    return value


def test_the_git_panel_follows_changes_as_they_happen_and_says_when_it_no_longer_hears_of_them(
    page_for, make_user, scripted_terminal
):
    person = make_user()
    terminal = scripted_terminal("Harbour", person, features={"git": True, "events": True})
    watching = []

    def watch(session) -> None:
        session.authenticated()
        session.receive()
        session.send(json.dumps({"type": "fs:ready", "path": SCRIPTED_HOME}))
        watching.append(session)

    terminal.script_sessions(watch, "/events")
    page = page_for(person)
    pick_terminal(page, "Harbour")
    git = page.get_by_role("region", name="Git")
    changes = git.get_by_role("button", name="Git changes")
    sync = git.get_by_role("button", name="Fetch", exact=True)
    expect(sync).to_be_visible()
    _wait_until(lambda: watching)
    assert watching, "the page never asked the terminal for live updates"
    # the page reads the repository once more when its folder is watched; let that pass first
    _settled(lambda: len(terminal.requests_to("/git/status")))
    expect(changes).not_to_contain_text("changed")

    terminal.route("GET", "/git/status", json_answer(_repository_status("tides.txt")))
    for session in list(watching):
        session.send(json.dumps({"type": "fs:change", "path": f"{SCRIPTED_HOME}/tides.txt"}))
    expect(changes).to_contain_text("1 changed")
    sync.hover()
    expect(page.get_by_role("tooltip").filter(has_text="Fetch")).to_be_visible()
    expect(page.get_by_role("tooltip").filter(has_text="Live updates unavailable")).to_have_count(0)
    changes.hover()

    # the terminal hangs up and takes no new connection
    terminal.refuse_sessions(503, "/events")
    for session in list(watching):
        session.drop()
    watching.clear()
    lost = page.get_by_role("tooltip").filter(has_text="Live updates unavailable")
    _wait_until(lambda: _hovering_shows(page, sync, changes, lost))
    expect(lost).to_be_visible()
    changes.hover()

    # it is back: the page connects again by itself and the notice goes
    del terminal.session_refusals["/events"]
    _wait_until(lambda: watching, seconds=40)
    assert watching, "the page did not ask for live updates again after losing them"
    _wait_until(lambda: not _hovering_shows(page, sync, changes, lost))
    sync.hover()
    expect(page.get_by_role("tooltip").filter(has_text="Fetch")).to_be_visible()
    expect(lost).to_have_count(0)


def _hovering_shows(page: Page, button: Locator, elsewhere: Locator, tooltip: Locator) -> bool:
    """Whether `tooltip` shows when the pointer comes onto `button` afresh."""
    elsewhere.hover()
    button.hover()
    try:
        tooltip.wait_for(state="visible", timeout=1_000)
    except PlaywrightTimeoutError:
        return False
    return True


def test_a_file_opened_again_keeps_its_one_tab_and_its_folder_crumb_leads_to_the_files_tab(
    terminal_page, repository
):
    page, terminal = terminal_page
    (repository / "moon.txt").write_text("full moon on friday\n")
    git = _open_git_panel(page, terminal, repository)
    strip = _tab_strip(page)
    folder_tab = strip.get_by_role("tab", name=repository.name, exact=True)
    expect(folder_tab).to_have_attribute("aria-selected", "true")

    git.get_by_role("button", name="Open file tides.txt").click()
    _selected(page, "tides.txt")
    expect(page.get_by_role("tabpanel", name="tides.txt")).to_contain_text("high water at seven")
    folder_tab.click()
    git.get_by_role("button", name="Open file moon.txt").click()
    _selected(page, "moon.txt")
    folder_tab.click()
    git.get_by_role("button", name="Open file tides.txt").click()
    _selected(page, "tides.txt")
    assert _tab_names(page) == ["New Chat", repository.name, "tides.txt", "moon.txt"], (
        f"opening a file that is already open left the tabs as {_tab_names(page)}"
    )

    # the file's tab names the folders above it; the home's crumb opens it in the Files tab
    page.get_by_role("tabpanel", name="tides.txt").get_by_role(
        "button", name="Home", exact=True
    ).click()
    expect(strip.get_by_role("tab", name=FILES, exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    files = page.get_by_role("region", name="File browser")
    expect(files.get_by_text(repository.name, exact=True)).to_be_visible()
    assert _tab_names(page) == ["New Chat", FILES, "tides.txt", "moon.txt"], (
        f"the crumb left the tabs as {_tab_names(page)}"
    )


def test_tabs_are_stepped_through_and_moved_with_the_keyboard(terminal_page):
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    open_new_terminal(page)
    _add_tab(page, "New File")
    strip = _tab_strip(page)
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1"]])

    strip.get_by_role("tab", name=FILES, exact=True).click()
    page.keyboard.press("ArrowRight")
    _selected(page, "Terminal 1")
    expect(strip.get_by_role("tab", name="Terminal 1")).to_be_focused()
    page.keyboard.press("End")
    _selected(page, "Untitled-1")
    # past the last tab the arrow comes round to the first
    page.keyboard.press("ArrowRight")
    _selected(page, "New Chat")
    page.keyboard.press("ArrowLeft")
    _selected(page, "Untitled-1")
    page.keyboard.press("Home")
    _selected(page, "New Chat")
    page.keyboard.press("ArrowRight")
    _selected(page, FILES)
    expect(page.get_by_role("region", name="File browser")).to_be_visible()

    page.keyboard.press("Alt+Shift+ArrowRight")
    _expect_panes(page, [["New Chat", "Terminal 1", FILES, "Untitled-1"]])
    _selected(page, FILES)
    # the moved tab is clicked again first: the move takes the focus off it (see the next test)
    strip.get_by_role("tab", name=FILES, exact=True).click()
    page.keyboard.press("Alt+Shift+ArrowLeft")
    _expect_panes(page, [["New Chat", FILES, "Terminal 1", "Untitled-1"]])
    _selected(page, FILES)


def test_a_tab_moved_with_the_keyboard_keeps_the_focus_for_the_next_move(terminal_page):
    """Red on dev c15f28649 (open-webui/open-webui#34099)."""
    page, terminal = terminal_page
    pick_terminal(page, terminal)
    open_new_terminal(page)
    _add_tab(page, "New File")
    strip = _tab_strip(page)
    files_tab = strip.get_by_role("tab", name=FILES, exact=True)
    files_tab.click()
    expect(files_tab).to_be_focused()

    page.keyboard.press("Alt+Shift+ArrowRight")
    _expect_panes(page, [["New Chat", "Terminal 1", FILES, "Untitled-1"]])
    focused = page.evaluate(
        "document.activeElement.getAttribute('title') || document.activeElement.tagName"
    )
    page.keyboard.press("Alt+Shift+ArrowRight")
    _wait_until(lambda: _panes(page) == [["New Chat", "Terminal 1", "Untitled-1", FILES]], 5)
    assert _panes(page) == [["New Chat", "Terminal 1", "Untitled-1", FILES]], (
        f"the second Alt+Shift+ArrowRight moved nothing: the tabs are {_panes(page)} and after "
        f"the first move the keyboard focus was on {focused!r}, not on the moved tab"
    )


def test_on_a_phone_the_tabs_button_shows_once_a_terminal_is_picked_and_opens_its_files(
    terminal_account, page_for, open_terminal
):
    person, terminal = terminal_account
    marker = open_terminal.home / f"pier-{uuid.uuid4().hex[:6]}.txt"
    marker.write_text("berth four\n")
    page = page_for(person, **PHONE)
    tabs_button = page.get_by_role("button", name="Tabs", exact=True)
    expect(chat_input(page)).to_be_visible()
    # nothing to show beside a new chat yet
    expect(tabs_button).to_have_count(0)

    tooltip_button(page.get_by_role("main"), "Terminal").tap()
    listed_terminal(page, terminal).tap()
    close_menu(page)
    expect(tabs_button).to_have_attribute("aria-expanded", "false")
    expect(page.get_by_role("region", name="File browser")).to_have_count(0)

    tabs_button.tap()
    drawer = page.get_by_role("dialog", name="Tabs")
    files = drawer.get_by_role("region", name="File browser")
    expect(files.get_by_text(marker.name, exact=True)).to_be_visible()
    expect(drawer.get_by_role("tab", name=FILES, exact=True)).to_have_attribute(
        "aria-selected", "true"
    )

    drawer.get_by_role("button", name="Hide tabs").tap()
    expect(drawer).to_have_count(0)
    expect(chat_input(page)).to_be_visible()
    expect(tabs_button).to_have_attribute("aria-expanded", "false")


def test_an_open_changed_file_shows_its_diff_from_the_toolbar_and_an_unchanged_one_has_none(
    terminal_page, repository
):
    page, terminal = terminal_page
    (repository / "notes.txt").write_text("nothing new here\n")
    (repository / "chart.bin").write_bytes(bytes(range(256)))
    _git(repository, "add", "notes.txt", "chart.bin")
    _git(repository, "commit", "-q", "-m", "Add the notes and the chart")
    (repository / "chart.bin").write_bytes(bytes(reversed(range(256))))
    _open_git_panel(page, terminal, repository)
    files = page.get_by_role("region", name="File browser")
    changes = page.get_by_role("region", name="File changes")
    show = files.get_by_role("button", name="Show diff")
    hide = files.get_by_role("button", name="Hide diff")

    files.get_by_text("tides.txt", exact=True).click()
    expect(show).to_have_attribute("aria-pressed", "false")
    expect(files.get_by_text("+1", exact=True)).to_be_visible()
    expect(changes).to_have_count(0)
    show.click()
    expect(hide).to_have_attribute("aria-pressed", "true")
    expect(changes.get_by_text("high water at six")).to_be_visible()
    expect(changes.get_by_text("high water at seven")).to_be_visible()
    hide.click()
    expect(changes).to_have_count(0)
    expect(show).to_be_visible()

    # a file with no change since the last commit has no diff to show
    files.get_by_role("button", name=repository.name, exact=True).click()
    files.get_by_text("notes.txt", exact=True).click()
    expect(files.get_by_role("button", name="Download")).to_be_visible()
    expect(show).to_have_count(0)

    files.get_by_role("button", name=repository.name, exact=True).click()
    files.get_by_text("chart.bin", exact=True).click()
    show.click()
    expect(changes).to_have_text("Binary file changed")
