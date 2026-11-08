"""The browser clipboard read from a page that may read it, beside a plain-HTTP page that may not.

A page opened over plain HTTP by address (`lan_page_for`) is not a secure context, so it cannot
read the clipboard itself. `loopback_clipboard(page, actor)` opens a page of the same browser on
the actor's loopback address, a secure context, and returns it with the clipboard filled with a
marker, so a copy that never happens reads back as `NOTHING_COPIED`. `copied(reader)` reads it.
"""

from __future__ import annotations

from playwright.sync_api import Page

from harness.actors import Actor
from harness.backends import wait_until

NOTHING_COPIED = "nothing copied yet"


def loopback_clipboard(page: Page, actor: Actor) -> Page:
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=actor.base_url)
    reader = page.context.new_page()
    reader.goto(f"{actor.base_url}/health")
    reader.evaluate("text => navigator.clipboard.writeText(text)", NOTHING_COPIED)
    page.bring_to_front()
    return reader


def _read(reader: Page) -> str:
    return reader.evaluate("navigator.clipboard.readText()")


def copied(reader: Page) -> str:
    """The clipboard's text once a copy has replaced the marker, else the marker."""
    wait_until(lambda: _read(reader) != NOTHING_COPIED, timeout=5)
    return _read(reader)
