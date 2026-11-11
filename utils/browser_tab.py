"""The Browser tab of a chat's workspace: opening it, its address bar, its Ports list and the
page it shows, plus a site served inside the terminal for it to open.

The tab asks the terminal for a browser session and shows pages through it: in Proxy mode in a
frame, in Chrome mode (a terminal with Chromium) as the screen of a Chrome in the terminal,
drawn on a canvas. A page the tab opens has to be served inside the terminal, so `serve_site`
writes its files there and starts a web server through Open WebUI, as the person.
"""

from __future__ import annotations

import re

import httpx
from playwright.sync_api import FrameLocator, Locator, Page, expect

NO_SERVERS = "No servers detected"
WEB_ADDRESSES_ONLY = "Enter an HTTP or HTTPS URL without credentials."
SESSION_GONE = "Session no longer available"


def tab_strip(page: Page) -> Locator:
    return page.get_by_role("tablist", name="Tabs")


def tab(page: Page, name: str | re.Pattern) -> Locator:
    return tab_strip(page).get_by_role("tab", name=name, exact=True)


def tab_names(page: Page) -> list[str]:
    return [name.strip() for name in tab_strip(page).get_by_role("tab").all_inner_texts()]


def add_tab(page: Page, entry: str) -> None:
    """Pick `entry` in the New tab menu; the entry shows once the terminal said it has it."""
    page.get_by_role("button", name="New tab", exact=True).click()
    # the entry's name carries its shortcut hint
    page.get_by_role("menu").get_by_role("menuitem", name=re.compile(rf"^{entry}\b")).click()
    expect(page.get_by_role("menu")).to_have_count(0)


def browser_view(page: Page) -> Locator:
    """The Browser tab on screen: the panel with the address bar."""
    return page.get_by_role("tabpanel").filter(has=page.get_by_role("textbox", name="URL"))


def open_browser_tab(page: Page) -> Locator:
    """Open a Browser tab of the picked terminal from the New tab menu; returns its panel."""
    add_tab(page, "Browser")
    view = browser_view(page)
    expect(view).to_be_visible()
    return view


def address_bar(view: Locator) -> Locator:
    return view.get_by_role("textbox", name="URL")


def visit(view: Locator, address: str) -> None:
    """Type `address` into the address bar and press Enter."""
    address_bar(view).fill(address)
    address_bar(view).press("Enter")


def arrow(view: Locator, name: str) -> Locator:
    """Back, Forward or Refresh beside the address bar."""
    return view.get_by_role("button", name=name, exact=True).first


def shown_page(view: Locator) -> FrameLocator:
    """The page a Proxy mode tab shows."""
    return view.frame_locator('iframe[title="Browser"]')


def chrome_screen(view: Locator) -> Locator:
    """The screen of the Chrome a Chrome mode tab shows."""
    return view.get_by_role("application", name="Browser page")


def port_row(view: Locator, port: int) -> Locator:
    """The row of a listening port on the tab's start page; a click opens the port in the tab."""
    return view.get_by_role("button", name=re.compile(rf"^:{port}\b"))


def ports_heading(view: Locator, count: int) -> Locator:
    """The Ports heading with the number of servers beside it."""
    return view.get_by_text(re.compile(rf"^Ports\s*{count}$"))


def open_options(page: Page, view: Locator) -> Locator:
    """Open the tab's Browser options; returns the menu."""
    view.get_by_role("button", name="Browser options").click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def choose_option(page: Page, view: Locator, name: str, role: str = "menuitemradio") -> None:
    """Pick a mode or a stream quality, or flip Mobile view (`role="menuitemcheckbox"`)."""
    open_options(page, view).get_by_role(role, name=name, exact=True).click()
    if role == "menuitemcheckbox":
        page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)


def chosen_options(page: Page, view: Locator) -> list[str]:
    """The ticked entries of Browser options, top to bottom."""
    menu = open_options(page, view)
    ticked = menu.locator('[aria-checked="true"]').all_inner_texts()
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    return [name.strip() for name in ticked]


def serve_site(
    client: httpx.Client,
    terminal_id: str,
    folder: str,
    port: int,
    pages: dict[str, str],
    headers: dict[str, str] | None = None,
) -> None:
    """Write `pages` (file name to HTML) into `folder` of the caller's terminal and serve the
    folder on `port` there; returns once the server answers inside the terminal."""
    terminal = f"/api/v1/terminals/{terminal_id}"
    for name, html in pages.items():
        written = client.post(
            f"{terminal}/files/write",
            headers=headers,
            json={"path": f"{folder}/{name}", "content": html},
        )
        assert written.status_code == 200, f"writing {name} failed: {written.text}"
    started = client.post(
        f"{terminal}/execute",
        params={"wait": 1},
        headers=headers,
        json={"command": f"cd {folder} && exec python3 -m http.server {port}"},
    )
    assert started.status_code == 200, f"starting the site's server failed: {started.text}"
    probe = (
        "import sys, time, urllib.request\n"
        "for _ in range(100):\n"
        "    try:\n"
        f"        urllib.request.urlopen('http://127.0.0.1:{port}/', timeout=1)\n"
        "        sys.exit(0)\n"
        "    except OSError:\n"
        "        time.sleep(0.1)\n"
        "sys.exit(1)\n"
    )
    answered = client.post(
        f"{terminal}/execute",
        params={"wait": 30},
        headers=headers,
        json={"command": f'python3 -c "{probe}"'},
    )
    assert answered.status_code == 200 and answered.json().get("exit_code") == 0, (
        f"the site's server on port {port} never answered: {answered.text}"
    )


def rewrite_page(
    client: httpx.Client, terminal_id: str, path: str, html: str, headers: dict | None = None
) -> None:
    written = client.post(
        f"/api/v1/terminals/{terminal_id}/files/write",
        headers=headers,
        json={"path": path, "content": html},
    )
    assert written.status_code == 200, f"rewriting {path} failed: {written.text}"


def open_tabs(client: httpx.Client, terminal_id: str, chat_id: str | None = None) -> list[dict]:
    """The Browser tabs the terminal holds for the caller in that chat (a new chat without),
    as the model's `browser_tabs` tool reads them: `url`, `title`, `mode`, `active`."""
    listed = client.get(
        f"/api/v1/terminals/{terminal_id}/browser/tabs",
        headers={"X-Session-Id": chat_id} if chat_id else None,
    )
    assert listed.status_code == 200, f"listing the Browser tabs failed: {listed.text}"
    return listed.json()["tabs"]
