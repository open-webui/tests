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
saved.

The workspace shortcuts (80ddf8e3b) open Files (Ctrl+Shift+E), a shell (Ctrl+`), a new file
(Ctrl+Alt+N) and a browser (Ctrl+Shift+B), step through a pane's tabs (Ctrl+Shift+] and [) and
close the active one (Ctrl+W, never the conversation). They do nothing while a menu or a dialog
is open or with Keyboard Shortcuts switched off; Desktop (Ctrl+Alt+D) does nothing on a terminal
without a desktop, New Chat Tab (Ctrl+Alt+O) needs the side chat permission and Comments
(Ctrl+Shift+Y) needs comments switched on. When the file browser is in a Git repository, the
Git panel under the tabs shows a changed file's diff, stages and commits it, and the commit shows
in its History and on the disk.

Since c24c3d914 each changed file's stage box and Stage all are checkboxes that say whether they
are ticked, Stage all reading mixed while only some files are staged; Discard changes also deletes
a file the last commit does not have, after a warning saying so; the list beside a diff can be
resized from the keyboard. The branch menu makes a branch that takes the next commit while main
stays as it was, and the sync button pushes a commit to the remote and publishes a new branch.
Undo last commit in History takes the last commit back and keeps its change staged.

Discriminates: on dev f659c0d12, in a frontend copy whose Stage all never reads mixed the stage
test fails, one that offers Discard only for tracked files (as v0.13.0) fails the discard test,
one whose divider ignores the keyboard fails the resize test, one whose branch pick checks
nothing out fails the branch test and one that offers Fetch in place of Push fails the push test;
the diff and undo tests stay green on that build as its controls.

Retargeted for 617618c9e: `test_the_full_width_layout_is_kept_after_a_reload` saved the removed
full-width switch and is now the split layout test; the reorder test now counts the conversation
tab. Since 0b7e7b3f7 the Files tab is named after the folder it shows, so the
tests name it by the terminal's home.

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
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.open_terminal import HOME_FOLDER
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import choose_terminal, open_new_terminal, pick_terminal
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# since 0b7e7b3f7 the Files tab is named after the folder it shows, here the terminal's home
FILES = HOME_FOLDER


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

    _drag(
        page,
        strip.get_by_role("tab", name="Untitled-1"),
        strip.get_by_role("tab", name="Terminal 1"),
    )
    expect(strip.get_by_role("tab").nth(2)).to_have_text("Untitled-1")
    assert _tab_names(page) == ["New Chat", FILES, "Untitled-1", "Terminal 1"], (
        f"dragging the new file's tab before the shell left the tabs as {_tab_names(page)}"
    )


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

    # leaving for a new chat and coming back brings the layout back too
    page.keyboard.press("Control+Shift+O")
    _expect_panes(page, [["New Chat"]])
    page.go_back()
    _expect_panes(page, layout)
    _selected(page, "Terminal 1")
    expect(_divider(page)).to_have_attribute("aria-valuenow", ratio)

    page.reload()
    # a reload is a cold load, which page_for would otherwise wait out
    expect(page.get_by_text("Start at the lighthouse.")).to_be_visible(
        timeout=config.navigation_timeout
    )
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
