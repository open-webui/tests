"""Journey: on a terminal with Chromium and a desktop, the Browser tab shows a real Chrome and the
Desktop tab the terminal's screen.

A terminal from the full Open Terminal image reports a desktop and has Chromium. A Browser tab
on it starts in Chrome mode: the page is rendered by a Chrome inside the terminal and its screen
is drawn in the tab, so the tab shows the page's own colours, is named after its title, and the
model's browser tools read the same page. Clicks and keys in the tab reach that page. Browser
options switch the tab to Proxy (the page in a frame) and back, pick the stream quality and turn
on Mobile view, in which the page gets a phone's width. The Desktop tab, one per terminal, shows
the terminal's 1280 by 800 desktop: the pointer moved in the tab moves on the desktop, Ctrl+V
hands the clipboard's text to the desktop, the speaker button plays the desktop's sound and the
full screen button fills the screen with it. A person who loses the terminal while the Desktop
tab is open is told "Desktop access denied. Reopen this tab.".

What the tab draws is a canvas, so the tests read it by its pixels and check the other side in
the terminal: the page's own title (it changes on a click, a key and a resize), the pointer and
the clipboard of the desktop, the model's snapshot of the page.

The terminal is the full image in a container (`full_open_terminal`); the tests skip where
`OPEN_TERMINAL_FULL_IMAGE` is not set.

Discriminates: passes on the dev c15f28649 build with Open Terminal 0.17.7. Each of these edits
of a frontend copy turns its test red: the Chrome's screen never opened, keys not passed on, the
picked mode not applied; Mobile view or a picked stream quality not passed on; each ask for the
Desktop opening another tab, pointer and keys not passed on, the speaker giving up at once,
muting leaving the sound open, the full screen button doing nothing; Ctrl+V handing nothing over;
a refused screen opened again like a broken one.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.terminals import execute, output_of
from utils.browser_tab import (
    add_tab,
    address_bar,
    choose_option,
    chosen_options,
    chrome_screen,
    open_browser_tab,
    open_tabs,
    serve_site,
    shown_page,
    tab,
    tab_names,
    visit,
)
from utils.cached_chat import choose_terminal
from utils.clipboard import write_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PAGE_COLOUR = (200, 30, 30)
# the stream packs the screen, which shifts a colour by a few steps
COLOUR_SLACK = 16
DESKTOP_SIZE = (1280, 800)
WEBM_HEADER = b"\x1a\x45\xdf\xa3"
ACCESS_DENIED = "Desktop access denied. Reopen this tab."
# Open WebUI checks an open stream's access every 10 s, whatever the browser's speed
RECHECK_TIMEOUT_MS = 30_000

REPORTING_PAGE = """<html><head><title>Harbour {tag}</title></head>
<body style="margin:0;background:rgb(200,30,30)">
<h1>Keeper {tag}</h1>
<script>
let typed = '';
addEventListener('keydown', (event) => {{
    if (event.key.length !== 1) return;
    typed += event.key;
    document.title = 'typed ' + typed;
}});
addEventListener('click', () => {{ document.title = 'clicked {tag}'; }});
</script>
</body></html>"""

WIDTH_PAGE = """<html><head><meta name="viewport" content="width=device-width">
<title>measuring</title></head><body><h1>Gauge {tag}</h1>
<script>
const report = () => {{ document.title = innerWidth + ' wide {tag}'; }};
addEventListener('resize', report);
report();
</script>
</body></html>"""


@pytest.fixture
def full_workbench(page_for, make_user, admin, preserve, full_open_terminal):
    """A person with the full terminal picked in a new chat: the page, the person, the
    terminal's id and a tag. The person's Browser tabs on the terminal end with the test, and
    their Chrome with them."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    connection = full_open_terminal.connection(
        name=f"Studio {tag}", config={"access_grants": [read_grant(person.id)]}
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    choose_terminal(page, connection["name"])
    yield page, person, connection, tag
    with admin.client() as client:
        configure_terminals(client, connection)
    with person.client() as client:
        for held in open_tabs(client, connection["id"]):
            client.delete(
                f"/api/v1/terminals/{connection['id']}/browser/sessions/{held['session_id']}"
            )


def _free_port() -> int:
    # the terminal is shared by the session's tests
    return 20000 + uuid.uuid4().int % 20000


def _page_colour(canvas: Locator) -> tuple[int, ...]:
    # away from the middle, where the test clicks and the screen draws the pointer
    return tuple(
        canvas.evaluate(
            "canvas => Array.from(canvas.getContext('2d')"
            ".getImageData(canvas.width * 3 / 4, canvas.height * 3 / 4, 1, 1).data.slice(0, 3))"
        )
    )


def _shows_the_pages_colour(canvas: Locator) -> bool:
    if canvas.count() != 1:
        return False
    return all(
        abs(seen - expected) <= COLOUR_SLACK
        for seen, expected in zip(_page_colour(canvas), PAGE_COLOUR)
    )


def _expect_the_pages_colour(view: Locator, when: str) -> None:
    canvas = chrome_screen(view).locator("canvas")
    assert wait_until(lambda: _shows_the_pages_colour(canvas), 20), (
        f"{when} the tab's screen does not show the page's colour {PAGE_COLOUR}: "
        f"{_page_colour(canvas) if canvas.count() == 1 else 'no screen'}"
    )


def _expect_the_title_in_the_terminal(person, terminal_id: str, title: str, when: str) -> None:
    """The page of the person's one Browser tab has `title` in the terminal's Chrome. The tab's
    own name follows the page only when it loads, so a title a script sets is read there."""

    def titles() -> list[str]:
        with person.client() as client:
            return [held["title"] for held in open_tabs(client, terminal_id)]

    assert wait_until(lambda: titles() == [title], 15), (
        f"{when} the page in the terminal's Chrome is titled {titles()}, not {title!r}"
    )


def _click_the_middle(page: Page, target: Locator) -> None:
    box = target.bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_a_browser_tab_shows_the_page_in_the_terminals_chrome_and_passes_clicks_and_keys_on(
    full_workbench,
):
    page, person, connection, tag = full_workbench
    terminal_id, port = connection["id"], _free_port()
    with person.client() as client:
        serve_site(
            client, terminal_id, f"site-{tag}", port, {"index.html": REPORTING_PAGE.format(tag=tag)}
        )
    view = open_browser_tab(page)
    assert chosen_options(page, view) == ["Chrome", "Balanced"], (
        f"a new Browser tab on a terminal with Chromium has these options ticked: "
        f"{chosen_options(page, view)}"
    )

    visit(view, f"localhost:{port}")
    expect(tab(page, f"Harbour {tag}")).to_be_visible()
    expect(address_bar(view)).to_have_value(f"http://localhost:{port}/")
    _expect_the_pages_colour(view, "in Chrome mode")
    expect(view.get_by_role("status")).to_have_count(0)
    expect(view.get_by_role("alert")).to_have_count(0)
    with person.client() as client:
        snapshot = client.get(f"/api/v1/terminals/{terminal_id}/browser/snapshot")
    assert snapshot.status_code == 200, f"the model's snapshot of the tab failed: {snapshot.text}"
    assert f'heading "Keeper {tag}"' in snapshot.json()["snapshot"], (
        f"the Chrome behind the tab does not show the page: {snapshot.text}"
    )

    _expect_the_title_in_the_terminal(person, terminal_id, f"Harbour {tag}", "before a click")
    _click_the_middle(page, chrome_screen(view).locator("canvas"))
    _expect_the_title_in_the_terminal(person, terminal_id, f"clicked {tag}", "after a click")
    chrome_screen(view).focus()
    page.keyboard.type("tide")
    _expect_the_title_in_the_terminal(person, terminal_id, "typed tide", "after typing tide")

    choose_option(page, view, "Proxy")
    expect(shown_page(view).get_by_role("heading", name=f"Keeper {tag}")).to_be_visible()
    expect(chrome_screen(view)).to_have_count(0)
    assert chosen_options(page, view) == ["Proxy"], (
        f"in Proxy mode these options are ticked: {chosen_options(page, view)}"
    )
    choose_option(page, view, "Chrome")
    _expect_the_pages_colour(view, "back in Chrome mode")
    expect(view.locator("iframe")).to_have_count(0)


def test_mobile_view_and_the_stream_quality_reach_the_chrome_in_the_terminal(full_workbench):
    page, person, connection, tag = full_workbench
    terminal_id, port = connection["id"], _free_port()
    with person.client() as client:
        serve_site(
            client, terminal_id, f"site-{tag}", port, {"index.html": WIDTH_PAGE.format(tag=tag)}
        )
    sent: list[str] = []

    def note_stream(socket) -> None:
        if socket.url.endswith("/stream"):
            socket.on(
                "framesent", lambda frame: sent.append(frame) if isinstance(frame, str) else None
            )

    page.on("websocket", note_stream)
    view = open_browser_tab(page)
    visit(view, f"localhost:{port}")
    expect(tab(page, re.compile(rf"^\d+ wide {tag}$"))).to_be_visible()

    def width_in_the_terminal() -> int:
        """How wide the page is in the terminal's Chrome, which the page writes into its title;
        0 before it has."""
        with person.client() as client:
            titles = [held["title"] for held in open_tabs(client, terminal_id)]
        measured = re.match(rf"^(\d+) wide {tag}$", titles[0]) if len(titles) == 1 else None
        return int(measured.group(1)) if measured else 0

    def expect_width(fits, when: str) -> None:
        assert wait_until(lambda: fits(width_in_the_terminal()), 15), (
            f"{when} the page in the terminal's Chrome is {width_in_the_terminal()} wide"
        )

    expect_width(lambda width: width > 600, "before Mobile view")
    choose_option(page, view, "Mobile view", role="menuitemcheckbox")
    expect_width(lambda width: width == 390, "in Mobile view")
    choose_option(page, view, "Mobile view", role="menuitemcheckbox")
    expect_width(lambda width: width > 600, "with Mobile view off again")

    assert '{"type":"quality","quality":"balanced"}' in sent, (
        f"the tab did not ask the terminal for the balanced stream it starts with: {sent}"
    )
    choose_option(page, view, "Crisp")
    assert chosen_options(page, view) == ["Chrome", "Crisp"], (
        f"after picking Crisp these options are ticked: {chosen_options(page, view)}"
    )
    assert wait_until(lambda: '{"type":"quality","quality":"crisp"}' in sent, 10), (
        f"picking Crisp did not reach the terminal: {[frame for frame in sent if 'quali' in frame]}"
    )


def _desktop(page: Page) -> Locator:
    return page.get_by_role("tabpanel", name="Desktop", exact=True)


def _open_desktop(page: Page) -> Locator:
    """Open the Desktop tab and wait until the desktop's screen is drawn; returns the screen."""
    add_tab(page, "Desktop")
    desktop = _desktop(page)
    expect(desktop).to_be_visible()
    screen = desktop.locator("canvas")
    expect(screen).to_have_count(1)
    expect(desktop.get_by_text(re.compile("Connecting|Reconnecting"))).to_have_count(0)
    return screen


def _colours(screen: Locator) -> int:
    """How many different colours a grid of points over the screen shows."""
    return screen.evaluate(
        """canvas => {
            const context = canvas.getContext('2d');
            const seen = new Set();
            for (let x = 0; x < 16; x++)
                for (let y = 0; y < 16; y++) {
                    const point = [(x + 0.5) * canvas.width / 16, (y + 0.5) * canvas.height / 16];
                    seen.add(context.getImageData(point[0], point[1], 1, 1).data.join(','));
                }
            return seen.size;
        }"""
    )


def _pointer_on_the_desktop(person, terminal_id: str) -> tuple[int, int]:
    with person.client() as client:
        located = output_of(execute(client, terminal_id, "xdotool getmouselocation"))
    found = re.search(r"x:(\d+) y:(\d+)", located)
    assert found, f"the desktop's pointer could not be read: {located!r}"
    return int(found.group(1)), int(found.group(2))


def test_the_desktop_tab_shows_the_terminals_desktop_and_takes_the_pointer_sound_and_full_screen(
    full_workbench,
):
    page, person, connection, tag = full_workbench
    terminal_id = connection["id"]
    sound: list = []
    page.on("websocket", lambda socket: sound.append(socket) if "/audio" in socket.url else None)

    screen = _open_desktop(page)
    assert tuple(screen.evaluate("canvas => [canvas.width, canvas.height]")) == DESKTOP_SIZE, (
        "the Desktop tab's screen is not the size of the terminal's desktop"
    )
    assert wait_until(lambda: _colours(screen) > 1, 20), (
        "the Desktop tab's screen stayed one flat colour"
    )
    add_tab(page, "Desktop")
    assert tab_names(page) == ["New Chat", "Desktop"], (
        f"asking for the Desktop a second time left the tabs as {tab_names(page)}"
    )

    box = screen.bounding_box()
    for across, down in ((0.25, 0.5), (0.75, 0.25)):
        page.mouse.move(box["x"] + box["width"] * across, box["y"] + box["height"] * down, steps=4)
        expected = (round(DESKTOP_SIZE[0] * across), round(DESKTOP_SIZE[1] * down))
        assert wait_until(
            lambda: all(
                abs(seen - wanted) <= 3
                for seen, wanted in zip(_pointer_on_the_desktop(person, terminal_id), expected)
            ),
            15,
        ), (
            f"the pointer moved in the tab to {expected} of the desktop is at "
            f"{_pointer_on_the_desktop(person, terminal_id)} there"
        )

    desktop = _desktop(page)
    speaker = desktop.get_by_role("button", name="Desktop audio")
    expect(speaker).to_have_attribute("aria-pressed", "false")
    assert not sound, "the desktop's sound was asked for before the speaker button"
    speaker.click()
    expect(speaker).to_have_attribute("aria-pressed", "true")
    assert wait_until(lambda: len(sound) == 1, 10), f"sound streams opened: {len(sound)}"
    first = sound[0].wait_for_event("framereceived", timeout=15000)
    assert isinstance(first, bytes) and first[:4] == WEBM_HEADER, (
        f"the desktop's sound started with {first[:16]!r}"
    )
    for _ in range(5):
        sound[0].wait_for_event("framereceived", timeout=15000)
    # the button springs back and a toast shows when the browser cannot play the stream
    expect(speaker).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_text("Desktop audio unavailable.")).to_have_count(0)
    speaker.click()
    expect(speaker).to_have_attribute("aria-pressed", "false")
    assert wait_until(sound[0].is_closed, 10), "muting left the desktop's sound stream open"

    desktop.get_by_role("button", name="Open in full screen").click()
    leave = desktop.get_by_role("button", name="Exit full screen")
    expect(leave).to_be_visible()
    assert page.evaluate("document.fullscreenElement !== null"), "nothing fills the screen"
    leave.click()
    expect(desktop.get_by_role("button", name="Open in full screen")).to_be_visible()
    assert page.evaluate("document.fullscreenElement === null"), (
        "the desktop still fills the screen"
    )


# the browser lets a page read the clipboard only in a secure context
@pytest.mark.secure_context
def test_ctrl_v_in_the_desktop_tab_hands_the_clipboards_text_to_the_desktop(full_workbench):
    page, person, connection, tag = full_workbench
    terminal_id = connection["id"]
    copied = f"moor at berth {tag}"
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    screen = _open_desktop(page)

    def desktop_clipboard() -> str:
        with person.client() as client:
            read = execute(client, terminal_id, "xclip -o -selection clipboard 2>&1")
        return output_of(read).strip()

    assert copied not in desktop_clipboard(), "the desktop already holds the text"
    write_text(page, copied)
    page.bring_to_front()
    _click_the_middle(page, screen)
    page.keyboard.press("Control+v")

    assert wait_until(lambda: desktop_clipboard() == copied, 15), (
        f"after Ctrl+V the desktop's clipboard holds {desktop_clipboard()!r}"
    )


@pytest.mark.slow
def test_a_person_who_loses_the_terminal_is_told_so_on_the_open_desktop_tab(full_workbench, admin):
    page, person, connection, tag = full_workbench
    screen = _open_desktop(page)
    assert wait_until(lambda: _colours(screen) > 1, 20), "the desktop never showed"
    expect(_desktop(page).get_by_text(ACCESS_DENIED)).to_have_count(0)

    with admin.client() as client:
        configure_terminals(client, {**connection, "config": {"access_grants": []}})

    expect(_desktop(page).get_by_text(ACCESS_DENIED)).to_be_visible(timeout=RECHECK_TIMEOUT_MS)
    expect(_desktop(page).get_by_role("button", name="Retry")).to_be_visible()
