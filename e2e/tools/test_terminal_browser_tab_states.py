"""Journey: what the Browser tab says when its terminal answers slowly, oddly or not at all.

The terminal here is scripted, so each state can be set: a terminal without live updates lists
its ports only when the tab reads them, so Refresh on the start page is what brings a new server
in and takes a stopped one out; a session the terminal fails to hand back shows the terminal's
words with Retry, which reads it again; a terminal whose Chrome does not start leaves the tab in
Proxy mode, and picking Chrome says "Chrome failed. Switched to Proxy." with the terminal's
reason; a tab whose session the terminal refuses to end is put back with the error, while one
the terminal no longer knows closes; and an open tab tells the terminal every minute that it is
still there. In Chrome mode a viewer the terminal refuses as expired says "Browser session
expired. Reopen this tab.", and one that keeps breaking off is tried three times before the tab
goes to Proxy mode with "Chrome connection failed.".

The Desktop tab on such a terminal: a desktop that does not start shows the terminal's reason
with Retry, a screen stream that ends is opened again after two seconds with "Reconnecting…",
and one the terminal closes as refused says "Desktop access denied. Reopen this tab." and is
not opened again.

Discriminates: passes on the dev c15f28649 build. Each of these edits of a frontend copy turns
its test red: Refresh doing nothing; Retry doing nothing; the fallback to Proxy not announced, a
failed Chrome not switched back; a tab whose session was not ended left closed, a session the
terminal no longer knows counted as a failed close, no ending asked for; no sign of life each
minute; a refused viewer tried again; the viewer given up after two tries; Retry on the Desktop
doing nothing, the terminal's reason not shown; a refused screen opened again, an ended one not.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

import pytest
from playwright.sync_api import Page, expect

from harness.backends import wait_until
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.browser_tab import (
    NO_SERVERS,
    add_tab,
    browser_view,
    choose_option,
    chosen_options,
    open_browser_tab,
    port_row,
    ports_heading,
    tab,
    tab_names,
)
from utils.cached_chat import choose_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHROME_FAILED = "Chrome failed. Switched to Proxy."
SESSION_EXPIRED = "Browser session expired. Reopen this tab."
DESKTOP_DENIED = "Desktop access denied. Reopen this tab."
VIEWER_TRIES = 3
KEEP_ALIVE_MS = 60_000


@dataclass
class ScriptedBrowser:
    """The browser side of a scripted terminal: one session, a list of ports, and what the
    terminal answers when the tab reads, changes or ends the session."""

    terminal: FakeTerminalServer
    tag: str
    chrome_available: bool = False
    ports: list[dict] = field(default_factory=list)
    # set to an answer to have the terminal fail that call
    read_answer: tuple | None = None
    chrome_answer: tuple | None = None
    end_answer: tuple | None = None
    mode: str = "proxy"
    changes: list[dict] = field(default_factory=list)

    @property
    def session_path(self) -> str:
        return f"/browser/sessions/tab-{self.tag}"

    def session(self) -> dict:
        return {
            "id": f"tab-{self.tag}",
            "token": f"token-{self.tag}",
            "url": "",
            "title": "",
            "mode": self.mode,
            "chrome_available": self.chrome_available,
        }

    def _changed(self, request) -> tuple:
        change = json.loads(request.body or b"{}")
        self.changes.append(change)
        if change.get("mode") == "chrome" and self.chrome_answer:
            return self.chrome_answer
        self.mode = change.get("mode", self.mode)
        return json_answer(self.session())

    def install(self) -> None:
        serve = self.terminal.route
        features = {"browser": True, "desktop": True}
        serve("GET", "/api/config", json_answer({"features": features}))
        serve("GET", "/ports", lambda _request: json_answer({"ports": self.ports}))
        serve("POST", "/browser/sessions", lambda _request: json_answer(self.session()))
        serve(
            "GET",
            self.session_path,
            lambda _request: self.read_answer or json_answer(self.session()),
        )
        serve("PATCH", self.session_path, self._changed)
        serve(
            "DELETE",
            self.session_path,
            lambda _request: self.end_answer or json_answer({"status": "closed"}),
        )

    def ends(self) -> int:
        return sum(
            entry.method == "DELETE" for entry in self.terminal.requests_to(self.session_path)
        )


@pytest.fixture
def scripted(admin, make_user, preserve, page_for):
    """`scripted(**settings)`: a page of a person who picked a scripted terminal, and its
    `ScriptedBrowser`."""
    with serving_terminal() as terminal:

        def open_page(**settings) -> tuple[Page, ScriptedBrowser]:
            tag = uuid.uuid4().hex[:8]
            browser = ScriptedBrowser(terminal, tag, **settings)
            browser.install()
            preserve(TERMINAL_SERVERS_CONFIG)
            person = make_user()
            connection = terminal.connection(name=f"Lab box {tag}")
            connection["config"]["access_grants"] = [read_grant(person.id)]
            with admin.client() as client:
                configure_terminals(client, connection)
            page = page_for(person)
            choose_terminal(page, connection["name"])
            return page, browser

        yield open_page


def test_refresh_reads_the_ports_again_on_a_terminal_without_live_updates(scripted):
    page, browser = scripted()
    view = open_browser_tab(page)
    expect(view.get_by_text(NO_SERVERS)).to_be_visible()
    reads = len(browser.terminal.requests_to("/ports"))
    assert reads >= 1, "the start page never asked the terminal for its ports"

    browser.ports = [{"port": 5173, "pid": 41, "process": "vite"}]
    refresh = view.get_by_role("button", name="Refresh").last
    refresh.click()
    expect(port_row(view, 5173)).to_be_visible()
    expect(port_row(view, 5173)).to_contain_text("vite")
    expect(ports_heading(view, 1)).to_be_visible()
    expect(view.get_by_text(NO_SERVERS)).to_have_count(0)

    browser.ports.append({"port": 8000, "pid": 42, "process": "uvicorn"})
    refresh.click()
    expect(port_row(view, 8000)).to_be_visible()
    expect(ports_heading(view, 2)).to_be_visible()

    browser.ports = []
    refresh.click()
    expect(view.get_by_text(NO_SERVERS)).to_be_visible()
    expect(port_row(view, 5173)).to_have_count(0)
    assert len(browser.terminal.requests_to("/ports")) == reads + 3, (
        "each Refresh did not read the ports once: "
        f"{len(browser.terminal.requests_to('/ports')) - reads} reads for three clicks"
    )


def test_a_session_the_terminal_fails_to_hand_back_offers_retry(scripted):
    page, browser = scripted()
    browser.read_answer = json_answer({"detail": f"the browser is restarting {browser.tag}"}, 503)
    page.get_by_role("button", name="New tab", exact=True).click()
    page.get_by_role("menu").get_by_role("menuitem", name="Browser").click()

    failed = page.get_by_role("tabpanel", name="Browser", exact=True)
    expect(failed).to_contain_text(f"the browser is restarting {browser.tag}")
    expect(failed.get_by_role("button", name="Reopen browser")).to_have_count(0)
    expect(browser_view(page)).to_have_count(0)

    browser.read_answer = None
    failed.get_by_role("button", name="Retry").click()
    expect(browser_view(page)).to_be_visible()
    expect(browser_view(page).get_by_text(NO_SERVERS)).to_be_visible()
    expect(page.get_by_text(f"the browser is restarting {browser.tag}")).to_have_count(0)


def test_a_chrome_that_does_not_start_leaves_the_tab_in_proxy_mode_and_says_why(scripted):
    page, browser = scripted(chrome_available=True)
    browser.chrome_answer = json_answer({"detail": f"Chromium crashed {browser.tag}"}, 409)
    view = open_browser_tab(page)
    expect(view.get_by_text(NO_SERVERS)).to_be_visible()
    assert chosen_options(page, view) == ["Proxy"], (
        f"with a Chrome that did not start these options are ticked: {chosen_options(page, view)}"
    )
    expect(view.get_by_role("alert")).to_have_count(0)
    assert [change for change in browser.changes if "mode" in change] == [
        {"mode": "chrome"},
        {"mode": "proxy"},
    ], f"opening the tab asked the terminal for these modes: {browser.changes}"

    choose_option(page, view, "Chrome")
    expect(view.get_by_role("alert")).to_have_text(
        f"{CHROME_FAILED} Chromium crashed {browser.tag}"
    )
    assert chosen_options(page, view) == ["Proxy"], (
        f"after Chrome failed these options are ticked: {chosen_options(page, view)}"
    )
    assert browser.mode == "proxy", "the terminal was left in Chrome mode"

    browser.chrome_answer = None
    choose_option(page, view, "Chrome")
    expect(view.get_by_role("alert")).to_have_count(0)
    assert chosen_options(page, view)[0] == "Chrome", (
        f"with Chrome starting these options are ticked: {chosen_options(page, view)}"
    )


def test_a_tab_whose_session_the_terminal_does_not_end_is_put_back(scripted):
    page, browser = scripted()
    open_browser_tab(page)
    close = page.get_by_role("button", name="Close Browser", exact=True)

    browser.end_answer = json_answer({"detail": f"the session is busy {browser.tag}"}, 500)
    close.click()
    expect(page.get_by_text(f"the session is busy {browser.tag}")).to_be_visible()
    expect(tab(page, "Browser")).to_be_visible()
    expect(browser_view(page)).to_be_visible()
    assert browser.ends() == 1, f"the tab asked the terminal {browser.ends()} times to end it"

    # a session the terminal no longer knows needs no ending
    browser.end_answer = json_answer({"detail": "Browser session not found"}, 404)
    close.click()
    expect(tab(page, "Browser")).to_have_count(0)
    assert tab_names(page) == ["New Chat"], f"closing the tab left {tab_names(page)}"
    assert browser.ends() == 2, f"the tab asked the terminal {browser.ends()} times to end it"


def test_an_open_browser_tab_tells_the_terminal_every_minute_that_it_is_still_there(scripted):
    page, browser = scripted()
    # the tab's timers are started after this, so the test moves their clock
    page.clock.install()
    view = open_browser_tab(page)
    expect(view.get_by_text(NO_SERVERS)).to_be_visible()
    assert {} not in browser.changes, f"the tab said it is still there at once: {browser.changes}"

    for minute in (1, 2):
        page.clock.run_for(KEEP_ALIVE_MS)
        assert wait_until(lambda: browser.changes.count({}) == minute, 10), (
            f"after {minute} min the terminal heard {browser.changes.count({})} times that the "
            f"tab is still there: {browser.changes}"
        )


def _streams(browser: ScriptedBrowser, ending: str) -> list:
    """The streams the terminal was asked for whose path, without the query, ends so."""
    return [
        session
        for session in browser.terminal.sessions
        if session.path.split("?")[0].endswith(ending)
    ]


def _viewer_streams(browser: ScriptedBrowser) -> list:
    return _streams(browser, "/stream")


def test_a_chrome_viewer_the_terminal_refuses_as_expired_says_so(scripted):
    page, browser = scripted(chrome_available=True)
    browser.terminal.refuse_sessions(404, "/browser/view/*")
    view = open_browser_tab(page)

    expect(view.get_by_role("alert")).to_have_text(SESSION_EXPIRED)
    assert chosen_options(page, view)[0] == "Chrome", (
        f"a refused viewer left these options ticked: {chosen_options(page, view)}"
    )
    refused = [
        entry for entry in browser.terminal.received if entry.path.startswith("/browser/view/")
    ]
    assert len(refused) == 1, f"the refused viewer was tried {len(refused)} times"


def test_a_chrome_viewer_that_keeps_breaking_off_is_tried_three_times_then_the_tab_goes_to_proxy(
    scripted,
):
    page, browser = scripted(chrome_available=True)

    def break_off(session) -> None:
        session.drop()

    browser.terminal.script_sessions(break_off, "/browser/view/*")
    view = open_browser_tab(page)

    expect(view.get_by_role("alert")).to_have_text(
        f"{CHROME_FAILED} Chrome connection failed.", timeout=20_000
    )
    assert len(_viewer_streams(browser)) == VIEWER_TRIES, (
        f"the viewer was opened {len(_viewer_streams(browser))} times before the tab gave up"
    )
    assert chosen_options(page, view) == ["Proxy"], (
        f"after the viewer failed these options are ticked: {chosen_options(page, view)}"
    )
    assert browser.mode == "proxy", "the terminal was left in Chrome mode"


def _desktop(page: Page):
    return page.get_by_role("tabpanel", name="Desktop", exact=True)


def _screen_streams(browser: ScriptedBrowser) -> list:
    return _streams(browser, "/desktop/stream")


def test_a_desktop_that_does_not_start_shows_the_terminals_reason_and_retry_asks_again(scripted):
    page, browser = scripted()
    reason = f"Desktop is not installed on this terminal {browser.tag}"
    browser.terminal.route("GET", "/desktop", json_answer({"detail": reason}, 503))
    add_tab(page, "Desktop")

    expect(_desktop(page).get_by_text(reason)).to_be_visible()
    assert not _screen_streams(browser), "the tab opened a screen without a desktop"

    browser.terminal.route(
        "GET", "/desktop", json_answer({"width": 1280, "height": 800, "transport": "vnc"})
    )
    _desktop(page).get_by_role("button", name="Retry").click()
    expect(_desktop(page).get_by_text(reason)).to_have_count(0)
    assert wait_until(lambda: len(_screen_streams(browser)) == 1, 10), (
        f"Retry opened {len(_screen_streams(browser))} screen streams"
    )


def test_a_desktop_stream_that_ends_is_opened_again_unless_the_terminal_refused_it(scripted):
    page, browser = scripted()
    browser.terminal.route(
        "GET", "/desktop", json_answer({"width": 1280, "height": 800, "transport": "vnc"})
    )
    closes = iter((1000, 4003))

    def end(session) -> None:
        session.authenticated()
        session.close(next(closes))

    browser.terminal.script_sessions(end, "/desktop/stream")
    add_tab(page, "Desktop")

    # the first stream ended as any connection can; the second was refused
    expect(_desktop(page).get_by_text(DESKTOP_DENIED)).to_be_visible(timeout=20_000)
    streams = _screen_streams(browser)
    assert len(streams) == 2, f"the screen was opened {len(streams)} times"
    assert not [stream.failure for stream in streams if stream.failure], (
        f"the scripted streams failed: {[stream.failure for stream in streams]}"
    )
    expect(_desktop(page).get_by_role("button", name="Retry")).to_be_visible()
    # a refused screen is not asked for again: the next try would come two seconds later
    page.wait_for_timeout(4000)
    assert len(_screen_streams(browser)) == 2, (
        f"a refused screen was opened again: {len(_screen_streams(browser))} streams"
    )
