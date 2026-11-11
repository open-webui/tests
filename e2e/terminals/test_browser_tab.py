"""Journey: the Browser tab of a chat shows the web servers of the picked terminal.

A person who picked a terminal opens a Browser tab from the New tab menu. Its start page lists
the ports that listen in the terminal: "No servers detected" at first, and a server started in
the terminal shows by itself, with the number of servers beside Ports. A click on the port opens
it in the tab, through a browser session the terminal keeps for the tab; the tab is then named
after the page's title and the address bar reads the page's own address. The address bar takes
web addresses only (a bare `localhost` gets `http`, an address with a sign-in or another scheme
is refused in words), Back and Forward walk the pages visited, and Refresh loads the page again.
A link that asks for a new window opens a second Browser tab; the terminal knows which of a
chat's tabs is on screen, which is the one the model's browser tools read, and a closed tab's
session is gone from it. Once the chat is saved its tabs come back after a reload; a tab whose
session the terminal no longer has says "Session no longer available" and "Reopen browser"
starts it again. Open WebUI serves the page in the tab, so its frame is sandboxed without its
origin, also with the person's Terminal Preview Allow Same Origin on: the page cannot read the
person's sign-in. Open WebUI's frame and Open Terminal's own sandbox header each keep it out.

The terminal is a template on a real Terminals orchestrator, so each test gets a machine of its
own (the slim Open Terminal image: no Chromium, so the tab works in Proxy mode); the tests skip
without the images or Docker.

`test_a_port_opens_in_a_new_browser_tab_from_the_start_page` is red on dev c15f28649 and
e791cbdde (#34108): Open in new tab beside a port opens `/proxy/<port>/` of Open WebUI itself,
which answers "404: Not Found". Its Per chat case also fails on a frontend copy that opens the
terminal's own port address, which such a terminal refuses without the chat ("This terminal
requires a saved chat"); it passes with the tab's preview address handed to the Browser tab.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7,
except the red test, which passes in both cases on the e791cbdde build with the tab's preview
address handed to the Browser tab (fails in both on e791cbdde; the Per chat case fails on a copy
whose port row opens the terminal's own port address). Each of these edits of a frontend copy
turns its test red: live port updates dropped,
no number beside Ports, a port click that opens nothing, the tab named after the host, the
terminal not told the tab's page, Desktop offered on every terminal; any scheme or a sign-in
accepted, a bare localhost sent over https, a leading slash not read against the page on screen,
Back and Forward or Refresh doing nothing; a new window's request dropped, the terminal not told
which tab is on screen, a closed tab's session left on the terminal; a saved chat's tab not
handed to the chat, Reopen browser doing nothing, a gone session shown as there. The sign-in
test is red only with both sandboxes gone (a frontend copy that gives the frame its origin while
the tab has a session and a backend copy that drops the terminal's sandbox header), green with
either alone.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.backends import wait_until
from utils.browser_tab import (
    NO_SERVERS,
    SESSION_GONE,
    WEB_ADDRESSES_ONLY,
    address_bar,
    arrow,
    browser_view,
    open_browser_tab,
    open_tabs,
    port_row,
    ports_heading,
    rewrite_page,
    serve_site,
    shown_page,
    tab,
    tab_names,
    visit,
)
from utils.cached_chat import ask, choose_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# each test's machine is its own container, so the port is free
PORT = 8765
HARBOUR = f"http://localhost:{PORT}/"


def _page(title: str, heading: str, more: str = "") -> str:
    return f"<html><head><title>{title}</title></head><body><h1>{heading}</h1>{more}</body></html>"


def _open_workbench(
    page_for, make_user, shared_terminal, interface: dict | None = None, scope: str = "user"
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Workbench {tag}"
    terminal_id = shared_terminal(person, name=name, scope=scope)
    if interface:
        with person.client() as client:
            saved = client.post("/api/v1/users/user/settings/update", json={"ui": interface})
        assert saved.status_code == 200, f"saving the settings failed: {saved.text}"
    page = page_for(person)
    choose_terminal(page, name)
    return page, person, terminal_id, tag


@pytest.fixture
def workbench(page_for, make_user, shared_terminal):
    """A person with a terminal of their own machine picked in a new chat: the page, the
    person, the terminal's id and a tag for what the test names."""
    return _open_workbench(page_for, make_user, shared_terminal)


def _serve(person, terminal_id: str, tag: str, headers: dict | None = None, **pages: str) -> None:
    with person.client() as client:
        serve_site(client, terminal_id, f"site-{tag}", PORT, pages, headers)


def _heading(page: Page, name: str):
    return shown_page(browser_view(page)).get_by_role("heading", name=name, exact=True)


def test_a_server_started_in_the_terminal_shows_under_ports_and_opens_in_the_tab(workbench):
    page, person, terminal_id, tag = workbench
    page.get_by_role("button", name="New tab", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name=re.compile(r"^Browser\b"))).to_be_visible()
    # the slim image has no desktop, and the terminal says so in the same answer
    expect(menu.get_by_role("menuitem", name=re.compile(r"^Desktop\b"))).to_have_count(0)
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)

    view = open_browser_tab(page)
    expect(view.get_by_text(NO_SERVERS)).to_be_visible()
    with person.client() as client:
        held = open_tabs(client, terminal_id)
    assert [(held_tab["url"], held_tab["mode"]) for held_tab in held] == [("", "proxy")], (
        f"the terminal holds these Browser tabs for the new tab: {held}"
    )

    _serve(person, terminal_id, tag, **{"index.html": _page(f"Harbour {tag}", f"Keeper {tag}")})
    expect(port_row(view, PORT)).to_be_visible()
    expect(port_row(view, PORT)).to_contain_text("python")
    expect(ports_heading(view, 1)).to_be_visible()
    expect(view.get_by_text(NO_SERVERS)).to_have_count(0)

    port_row(view, PORT).click()
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(HARBOUR)
    expect(tab(page, f"Harbour {tag}")).to_be_visible()
    assert tab_names(page) == ["New Chat", f"Harbour {tag}"], (
        f"opening the port left the tabs as {tab_names(page)}"
    )
    with person.client() as client:
        assert wait_until(
            lambda: (
                [
                    (held_tab["url"], held_tab["title"])
                    for held_tab in open_tabs(client, terminal_id)
                ]
                == [(HARBOUR, f"Harbour {tag}")]
            ),
            15,
        ), f"the terminal was not told the tab's page: {open_tabs(client, terminal_id)}"


@pytest.mark.parametrize("scope", ["user", "chat"])
def test_a_port_opens_in_a_new_browser_tab_from_the_start_page(
    scope, page_for, make_user, shared_terminal, upstream
):
    """Red on dev c15f28649 and on v0.13.3 (open-webui/open-webui#34108). A Per chat terminal
    serves the port only under the chat's own address."""
    page, person, terminal_id, tag = _open_workbench(
        page_for, make_user, shared_terminal, scope=scope
    )
    # a signed-in browser carries the session as a cookie too, which a new tab's request needs
    page.context.add_cookies([{"name": "token", "value": person.token, "url": person.base_url}])
    chat = None
    if scope == "chat":
        # a Per chat terminal runs only for a saved chat
        ask(page, upstream, f"keep the harbour open {tag}", reply.text(f"Kept {tag}."))
        page.wait_for_url(re.compile(r"/c/[0-9a-f-]+$"))
        chat = {"X-Session-Id": page.url.rsplit("/", 1)[1]}
    view = open_browser_tab(page)
    _serve(
        person, terminal_id, tag, chat, **{"index.html": _page(f"Harbour {tag}", f"Keeper {tag}")}
    )
    row = port_row(view, PORT)
    expect(row).to_be_visible()

    with page.context.expect_page() as opening:
        row.locator("xpath=..").get_by_role("button", name="Open in new tab").click()
    opened = opening.value

    expect(
        opened.get_by_role("heading", name=f"Keeper {tag}"),
        f"the new browser tab at {opened.url} does not show the server's page",
    ).to_be_visible()


def test_the_address_bar_takes_web_addresses_and_the_arrows_walk_the_pages_visited(workbench):
    page, person, terminal_id, tag = workbench
    link = '<a href="/quay.html">to the quay</a>'
    _serve(
        person,
        terminal_id,
        tag,
        **{
            "index.html": _page(f"Harbour {tag}", f"Keeper {tag}", link),
            "quay.html": _page(f"Quay {tag}", f"Mooring {tag}"),
            "tide.html": _page(f"Tide {tag}", f"Low water {tag}"),
        },
    )
    view = open_browser_tab(page)
    expect(port_row(view, PORT)).to_be_visible()
    expect(arrow(view, "Back")).to_be_disabled()
    expect(arrow(view, "Refresh")).to_be_disabled()

    # a bare localhost address is opened over http
    visit(view, f"localhost:{PORT}")
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(HARBOUR)
    expect(arrow(view, "Back")).to_be_disabled()
    expect(arrow(view, "Forward")).to_be_disabled()

    shown_page(view).get_by_role("link", name="to the quay").click()
    expect(_heading(page, f"Mooring {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(f"{HARBOUR}quay.html")
    expect(tab(page, f"Quay {tag}")).to_be_visible()

    # an address that starts with a slash is a page of the site on screen
    visit(view, "/tide.html")
    expect(_heading(page, f"Low water {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(f"{HARBOUR}tide.html")

    arrow(view, "Back").click()
    expect(_heading(page, f"Mooring {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(f"{HARBOUR}quay.html")
    arrow(view, "Back").click()
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()
    expect(tab(page, f"Harbour {tag}")).to_be_visible()
    expect(arrow(view, "Back")).to_be_disabled()
    arrow(view, "Forward").click()
    expect(_heading(page, f"Mooring {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(f"{HARBOUR}quay.html")

    for refused in ("ftp://example.com/", f"http://keeper:secret@localhost:{PORT}/tide.html"):
        visit(view, refused)
        expect(view.get_by_role("alert")).to_have_text(WEB_ADDRESSES_ONLY)
        expect(_heading(page, f"Mooring {tag}")).to_be_visible()
        # an address the bar takes clears the refusal
        visit(view, f"{HARBOUR}quay.html")
        expect(view.get_by_role("alert")).to_have_count(0)

    with person.client() as client:
        rewrite_page(
            client,
            terminal_id,
            f"site-{tag}/quay.html",
            _page(f"Quay {tag}", f"Mooring rebuilt {tag}"),
        )
    arrow(view, "Refresh").click()
    expect(_heading(page, f"Mooring rebuilt {tag}")).to_be_visible()


def test_a_link_for_a_new_window_opens_a_second_tab_and_the_terminal_knows_the_one_on_screen(
    workbench,
):
    page, person, terminal_id, tag = workbench
    link = '<a href="/quay.html" target="_blank">the quay in a new window</a>'
    _serve(
        person,
        terminal_id,
        tag,
        **{
            "index.html": _page(f"Harbour {tag}", f"Keeper {tag}", link),
            "quay.html": _page(f"Quay {tag}", f"Mooring {tag}"),
        },
    )
    view = open_browser_tab(page)
    visit(view, HARBOUR)
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()

    def held() -> dict[str, bool]:
        with person.client() as client:
            return {
                held_tab["title"]: held_tab["active"] for held_tab in open_tabs(client, terminal_id)
            }

    def expect_held(expected: dict[str, bool], when: str) -> None:
        assert wait_until(lambda: held() == expected, 15), (
            f"{when} the terminal holds these tabs (title: on screen): {held()}"
        )

    expect_held({f"Harbour {tag}": True}, "with one Browser tab")

    shown_page(view).get_by_role("link", name="the quay in a new window").click()
    expect(tab(page, f"Quay {tag}")).to_be_visible()
    expect(tab(page, f"Quay {tag}")).to_have_attribute("aria-selected", "true")
    expect(_heading(page, f"Mooring {tag}")).to_be_visible()
    assert tab_names(page) == ["New Chat", f"Harbour {tag}", f"Quay {tag}"], (
        f"the link for a new window left the tabs as {tab_names(page)}"
    )
    expect_held({f"Harbour {tag}": False, f"Quay {tag}": True}, "after the second tab opened")

    tab(page, f"Harbour {tag}").click()
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()
    expect_held({f"Harbour {tag}": True, f"Quay {tag}": False}, "back on the first tab")

    page.get_by_role("button", name=f"Close Quay {tag}", exact=True).click()
    expect(tab(page, f"Quay {tag}")).to_have_count(0)
    expect_held({f"Harbour {tag}": True}, "after the second tab was closed")


def test_a_saved_chats_browser_tab_comes_back_after_a_reload_and_reopens_once_its_session_is_gone(
    workbench, upstream
):
    page, person, terminal_id, tag = workbench
    _serve(person, terminal_id, tag, **{"index.html": _page(f"Harbour {tag}", f"Keeper {tag}")})
    view = open_browser_tab(page)
    visit(view, HARBOUR)
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()

    tab(page, "New Chat").click()
    ask(page, upstream, f"keep the harbour open {tag}", reply.text(f"Kept {tag}."))
    page.wait_for_url(re.compile(r"/c/[0-9a-f-]+$"))
    chat_id = page.url.rsplit("/", 1)[1]
    with person.client() as client:
        # the tab now belongs to the saved chat on the terminal
        assert wait_until(
            lambda: (
                [held_tab["url"] for held_tab in open_tabs(client, terminal_id, chat_id)]
                == [HARBOUR]
            ),
            15,
        ), (
            f"the saved chat's tabs on the terminal: {open_tabs(client, terminal_id, chat_id)}; "
            f"without a chat: {open_tabs(client, terminal_id)}"
        )
        session_id = client.get(
            f"/api/v1/terminals/{terminal_id}/browser/tabs", headers={"X-Session-Id": chat_id}
        ).json()["tabs"][0]["session_id"]

    page.reload()
    expect(tab(page, f"Harbour {tag}")).to_be_visible()
    tab(page, f"Harbour {tag}").click()
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()
    expect(address_bar(browser_view(page))).to_have_value(HARBOUR)

    with person.client() as client:
        closed = client.delete(
            f"/api/v1/terminals/{terminal_id}/browser/sessions/{session_id}",
            headers={"X-Session-Id": chat_id},
        )
        assert closed.status_code == 200, f"ending the tab's session failed: {closed.text}"
    page.reload()
    expect(tab(page, f"Harbour {tag}")).to_be_visible()
    tab(page, f"Harbour {tag}").click()
    gone = page.get_by_role("tabpanel").filter(has_text=SESSION_GONE)
    expect(gone).to_be_visible()
    expect(_heading(page, f"Keeper {tag}")).to_have_count(0)

    gone.get_by_role("button", name="Reopen browser").click()
    reopened = browser_view(page)
    expect(reopened).to_be_visible()
    expect(page.get_by_text(SESSION_GONE)).to_have_count(0)
    with person.client() as client:
        assert wait_until(lambda: len(open_tabs(client, terminal_id, chat_id)) == 1, 15), (
            f"Reopen browser left the terminal with {open_tabs(client, terminal_id, chat_id)}"
        )
        new_session = open_tabs(client, terminal_id, chat_id)[0]["session_id"]
    assert new_session != session_id, "Reopen browser did not start a new session"
    visit(reopened, HARBOUR)
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()


def test_a_page_in_the_tab_cannot_reach_open_webui_even_with_same_origin_previews_on(
    page_for, make_user, shared_terminal
):
    page, person, terminal_id, tag = _open_workbench(
        page_for, make_user, shared_terminal, {"terminalPreviewAllowSameOrigin": True}
    )
    probe = (
        '<p id="reach">checking</p><script>'
        "let reached;"
        "try { reached = localStorage.getItem('token') ? 'token read' : 'no token'; }"
        "catch (error) { reached = 'storage refused'; }"
        "document.getElementById('reach').textContent = reached;"
        "</script>"
    )
    _serve(
        person, terminal_id, tag, **{"index.html": _page(f"Harbour {tag}", f"Keeper {tag}", probe)}
    )
    view = open_browser_tab(page)
    visit(view, HARBOUR)
    expect(_heading(page, f"Keeper {tag}")).to_be_visible()

    # the frame is served by Open WebUI, so only its sandbox keeps the page from the account
    expect(shown_page(view).locator("#reach")).to_have_text("storage refused")
