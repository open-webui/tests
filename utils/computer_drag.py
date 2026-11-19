"""Files and folders of this computer dragged onto the page and dropped, as from a file manager."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Locator, Page


def drag_from_the_computer(page: Page, onto: Locator, *paths: Path) -> dict:
    """Bring files or folders of this computer over `onto` and hold them there; returns what
    `drop` needs."""
    box = onto.bounding_box()
    assert box, "the place to drop on is not on the page"
    session = page.context.new_cdp_session(page)
    held = {
        "x": box["x"] + box["width"] / 2,
        "y": box["y"] + box["height"] / 2,
        "data": {"items": [], "files": [str(path) for path in paths], "dragOperationsMask": 1},
    }
    # the page offers no way to start a drag of the computer's own files
    session.send("Input.dispatchDragEvent", {"type": "dragEnter", **held})
    session.send("Input.dispatchDragEvent", {"type": "dragOver", **held})
    return {"session": session, **held}


def drop(held: dict) -> None:
    session = held.pop("session")
    session.send("Input.dispatchDragEvent", {"type": "drop", **held})
    session.detach()
