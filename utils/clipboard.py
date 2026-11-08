"""The clipboard of a browser page, read and written by the test.

Every page of the suite runs over plain HTTP, where the browser offers no `navigator.clipboard`
(see `harness/plain_http.py`). The test reaches the clipboard from a page on the server's loopback
address in the same browser context, which shares it, so the app page under test stays as a
plain HTTP user has it.

`loopback_clipboard(page, actor)` returns such a page with the clipboard filled with a marker, so a
copy that never happens reads back as `NOTHING_COPIED`; `copied(reader)` reads it.
"""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Page

from harness.actors import Actor
from harness.backends import wait_until
from harness.plain_http import loopback_page

COPIED = "navigator.clipboard.readText().then(text => text !== '')"
NOTHING_COPIED = "nothing copied yet"


def on_clipboard(page: Page, script: str, arg: Any = None) -> Any:
    """`script` evaluated beside `page` where `navigator.clipboard` exists, with `arg`."""
    helper = loopback_page(page)
    try:
        return helper.evaluate(script, arg)
    finally:
        helper.close()


def read_text(page: Page) -> str:
    return on_clipboard(page, "navigator.clipboard.readText()")


def write_text(page: Page, text: str) -> None:
    on_clipboard(page, "text => navigator.clipboard.writeText(text)", text)


def wait_for_text(page: Page) -> str:
    """The clipboard's text, once something was copied there."""
    helper = loopback_page(page)
    try:
        helper.wait_for_function(COPIED)
        return helper.evaluate("navigator.clipboard.readText()")
    finally:
        helper.close()


def loopback_clipboard(page: Page, actor: Actor) -> Page:
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=actor.base_url)
    reader = loopback_page(page)
    reader.evaluate("text => navigator.clipboard.writeText(text)", NOTHING_COPIED)
    page.bring_to_front()
    return reader


def _read(reader: Page) -> str:
    return reader.evaluate("navigator.clipboard.readText()")


def copied(reader: Page) -> str:
    """The clipboard's text once a copy has replaced the marker, else the marker."""
    wait_until(lambda: _read(reader) != NOTHING_COPIED, timeout=5)
    return _read(reader)
