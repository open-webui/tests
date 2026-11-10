"""Journey: a terminal watches a folder for each Files tab on screen and for no page that is gone.

The browser's side of invariant 2 of `integration/terminals/test_terminal_watchers.py`: pages
that are gone are cleaned up. A Files tab holds one live-update connection to its terminal
through Open WebUI, and through it a watcher on the folder it shows. A person closes the tab
and opens it again, enters a folder and goes back, opens the terminal in a second browser and
closes one page after the other. After every step the terminal watches for the Files tabs on
screen and no others (none without a tab, at most one for each), and once the pages are closed
it holds no connection and no watcher. Nothing here allows a number of leftovers: one watcher
left per closed tab is how a single person gets a terminal to the 40 watchers from which Open
Terminal 0.17.7 makes every file request wait (open-webui/open-terminal#230).

Each step waits until the tab was told that its folder is watched (one second on Open Terminal
0.17.7): a tab that leaves before that is let go of by the answer it missed, also by a proxy
that would keep it otherwise.

Not covered here: a browser that is killed and one that stops answering, which the integration
module covers with pages on plain sockets, and what a reload brings back on screen.

Discriminates: on dev c15f28649 with Open Terminal 0.17.7, a backend copy in which Open WebUI
ends a stream only when both of its directions have ended (what Terminals 0.3.1 did) fails it:
every closed tab and every folder change leaves one more watcher (1 with no tab on screen in
the first round, 7 with one tab by the end of the second).
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Locator, Page, WebSocket, expect

from harness.actors import Actor
from harness.open_terminal import (
    EventCounts,
    OpenTerminal,
    serving_open_terminal,
    wait_for_event_counts,
)
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOBODY = EventCounts(connections=0, watchers=0)
RELEASE_BOUND_SECONDS = 5
TOLD_BOUND_SECONDS = 10
ROUNDS = 3


@pytest.fixture(scope="module")
def watched_terminal() -> Iterator[OpenTerminal]:
    """An Open Terminal of this module's own, so that only its pages connect to it."""
    with serving_open_terminal() as terminal:
        yield terminal


@pytest.fixture
def its_person(make_user, admin, preserve, watched_terminal) -> tuple[Actor, str]:
    """A fresh account with read access to the terminal, and the terminal's name."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    grants = {"access_grants": [read_grant(person.id)]}
    connection = watched_terminal.connection(config=grants)
    with admin.client() as client:
        configure_terminals(client, connection)
    return person, connection["name"]


class LiveUpdates:
    """The live-update connections a page opened, and whether each was told that its folder is
    watched."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.told: list[bool] = []
        self.waited_for = 0
        page.on("websocket", self._opened)

    def _opened(self, socket: WebSocket) -> None:
        if not urlsplit(socket.url).path.endswith("/events"):
            return
        place = len(self.told)
        self.told.append(False)

        def received(frame: str | bytes) -> None:
            if "fs:ready" in str(frame):
                self.told[place] = True

        socket.on("framereceived", received)

    def wait_until_told(self) -> None:
        """Wait until the Files tab's newest connection, opened since the last wait, was told."""
        deadline = time.monotonic() + TOLD_BOUND_SECONDS
        while not (len(self.told) > self.waited_for and self.told[-1]):
            assert time.monotonic() < deadline, (
                f"the Files tab was not told within {TOLD_BOUND_SECONDS}s that its folder is "
                f"watched (its live-update connections: {self.told}, {self.waited_for} of them "
                f"from earlier steps)"
            )
            # the page's frames are only handed over while Playwright is waiting
            self.page.wait_for_timeout(100)
        self.waited_for = len(self.told)


def _files_tab(page: Page) -> Locator:
    return page.get_by_role("region", name="File browser")


def _watched_for(tabs: int):
    """Whether a terminal watches for `tabs` Files tabs: not at all without one, and with some
    at least once and at most once for each."""
    return lambda counts: min(tabs, 1) <= counts.watchers <= tabs


def test_a_terminal_watches_for_each_files_tab_on_screen_and_for_no_page_that_is_gone(
    page_for, its_person, watched_terminal
):
    """A higher limit is a stopgap: the count must return to baseline."""
    person, terminal = its_person
    folder = watched_terminal.home / f"dock-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    (folder / "manifest.txt").write_text("manifest\n")
    before = wait_for_event_counts(watched_terminal.event_counts, NOBODY, RELEASE_BOUND_SECONDS)
    assert before == NOBODY, f"before any page was open, the terminal held {before}"
    page = page_for(person)
    pages = {page: LiveUpdates(page)}
    # step: (Files tabs on screen, watchers on the terminal)
    seen: dict[str, tuple[int, int]] = {}

    def look(step: str, watching_anew: Page | None = None) -> None:
        if watching_anew:
            # a tab that leaves before it was told is let go of by the answer it missed
            pages[watching_anew].wait_until_told()
        on_screen = sum(_files_tab(open_page).count() for open_page in pages)
        held = wait_for_event_counts(
            watched_terminal.event_counts, _watched_for(on_screen), RELEASE_BOUND_SECONDS
        )
        seen[step] = (on_screen, held.watchers)

    pick_terminal(page, terminal)
    files = _files_tab(page)
    look("the Files tab opened", page)
    for round_number in range(1, ROUNDS + 1):
        page.keyboard.press("Control+W")
        expect(files).to_have_count(0)
        look(f"round {round_number}: the Files tab closed")
        page.keyboard.press("Control+Shift+E")
        expect(files).to_be_visible()
        look(f"round {round_number}: the Files tab opened again", page)
        files.get_by_text(folder.name, exact=True).click()
        expect(files.get_by_text("manifest.txt", exact=True)).to_be_visible()
        look(f"round {round_number}: a folder entered", page)
        files.get_by_role("button", name="Home", exact=True).click()
        expect(files.get_by_text(folder.name, exact=True)).to_be_visible()
        look(f"round {round_number}: back in the home folder", page)

    second = page_for(person)
    pages[second] = LiveUpdates(second)
    pick_terminal(second, terminal)
    look("the Files tab opened in a second browser", second)
    page.close()
    del pages[page]
    look("the first browser's page closed")
    second.close()
    at_the_end = wait_for_event_counts(watched_terminal.event_counts, NOBODY, RELEASE_BOUND_SECONDS)

    unequal = {
        step: (tabs, watchers)
        for step, (tabs, watchers) in seen.items()
        if not min(tabs, 1) <= watchers <= tabs
    }
    assert not unequal, (
        f"the terminal did not watch for just the Files tabs on screen (step: Files tabs, "
        f"watchers {RELEASE_BOUND_SECONDS}s later): {unequal}"
    )
    in_both_browsers = seen["the Files tab opened in a second browser"]
    assert seen["the Files tab opened"] == (1, 1) and in_both_browsers[0] == 2, (
        f"the Files tab did not show in each browser, so not all of it was watched: {seen}"
    )
    assert at_the_end == NOBODY, (
        f"{RELEASE_BOUND_SECONDS}s after both pages were closed the terminal still held "
        f"{at_the_end}"
    )
