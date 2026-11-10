"""A terminal's shell in a tab of the chat's workspace, and reading its screen.

The shell draws on a WebGL canvas where the browser has one, which no test can read; a page
opened with `page_for(person, init_script=READABLE_SHELL)` has none, so the shell keeps its
plain text rows. `open_shell` adds a shell from the New tab menu, `expect_on_screen` waits for
text on its screen and `type_into` types into it. A shell that is not connected shows a notice
with one button in place of its screen (`shell_notice`): Retry, or Start new terminal.
"""

from __future__ import annotations

import re
import time

from playwright.sync_api import Locator, Page, expect

READABLE_SHELL = """
const getContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function (type, ...options) {
    return /webgl/i.test(type) ? null : getContext.call(this, type, ...options);
};
"""
RECONNECTING = "Connection interrupted. Reconnecting to the same terminal…"


def shell_tab(page: Page, number: int = 1) -> Locator:
    return page.get_by_role("tab", name=f"Terminal {number}", exact=True)


def shell_panel(page: Page, number: int = 1) -> Locator:
    return page.get_by_role("tabpanel", name=f"Terminal {number}", exact=True)


def open_shell(page: Page, number: int = 1) -> Locator:
    """Add a shell of the picked terminal from the New tab menu; returns its panel."""
    page.get_by_role("button", name="New tab", exact=True).first.click()
    # the entry's name carries its shortcut hint once the shortcuts have loaded
    page.get_by_role("menu").get_by_role("menuitem", name=re.compile(r"^New Terminal\b")).click()
    panel = shell_panel(page, number)
    expect(panel).to_be_visible()
    return panel


def screen_of(shell: Locator) -> str:
    """What the shell's screen shows, its rows joined: a long line wraps onto the next row.
    Nothing while the shell shows no screen."""
    rows = shell.locator(".xterm-rows").locator("visible=true")
    return rows.inner_text().replace("\n", "") if rows.count() else ""


def expect_on_screen(shell: Locator, text: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    shown = ""
    while time.monotonic() < deadline:
        shown = screen_of(shell)
        if text in shown:
            return
        time.sleep(0.1)
    raise AssertionError(f"the shell never showed {text!r}; its screen reads {shown!r}")


def type_into(shell: Locator, text: str) -> None:
    shell.get_by_role("textbox", name="Terminal input").press_sequentially(text)


def shell_notice(shell: Locator) -> Locator:
    """What a shell that is not connected says in place of its screen."""
    return shell.get_by_role("status")
