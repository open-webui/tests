"""A folder's row in the sidebar, located the way a person sees it.

The row holds the expand toggle (its first button), the folder's name that opens the folder (a
double click renames it, hovering shows a preview of the folder) and, on hover, the "More" menu
button (its last button). `folder_link(scope, name)` is the name to click; `folder_row(scope,
name)` is the whole row, for hovering, dragging and its buttons. `scope` is the sidebar or the
page. The name button also carries the folder's machine after its name (7a797bee4) and an unread
count when there is one, so the folder is found by the text of its name alone.
`open_advanced(dialog)` unfolds the folder dialog's Advanced section, which holds the default
model, the system prompt and the knowledge (70e4b9a52). `drag(page, source, target)` drops a chat
or folder row onto a folder or a section.

Retargeted for b704d0613: the folder's name is the trigger of its hover preview, so it is a
button that opens a dialog, still an anchor to the folder page. Chat rows are buttons as well,
so the name is looked up inside folder rows only.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

_ROWS = "[id^='folder-'][id$='-button']"


def _name_text(scope: Locator | Page, name: str) -> Locator:
    page = scope if isinstance(scope, Page) else scope.page
    return page.get_by_text(name, exact=True)


def folder_link(scope: Locator | Page, name: str) -> Locator:
    return scope.locator(_ROWS).get_by_role("button").filter(has=_name_text(scope, name))


def folder_row(scope: Locator | Page, name: str) -> Locator:
    return scope.locator(_ROWS).filter(has=_name_text(scope, name))


def click_folder_menu_button(row: Locator) -> None:
    """Click the row's "More" button, shown on hover, hovering the expand toggle to show it.

    Hovering the name instead opens its preview after a moment, and the preview covers the button.
    """
    toggle, menu_button = row.get_by_role("button").first, row.get_by_role("button").last
    # a folder above still unfolding moves the row from under the mouse, which hides the button
    for _ in range(3):
        toggle.hover()
        try:
            menu_button.click(timeout=3_000)
            return
        except PlaywrightTimeoutError:
            continue
    toggle.hover()
    menu_button.click()


def open_advanced(dialog: Locator) -> None:
    toggle = dialog.get_by_role("button", name="Advanced", exact=True)
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")


def drag(page: Page, source: Locator, target: Locator) -> None:
    """Drag `source` onto `target` with the events a browser fires for a drag and drop.

    A mouse drag across the sidebar rows sometimes never fires the drop. As in a browser the drag
    ends on the element that was dragged, which the drop may already have moved out of `source`'s
    reach (a chat dragged out of its folder), so that element is held from the start. The drop
    lands on the middle of `target`: near a folder row's top or bottom edge a dragged folder is put
    before or after it instead of into it (20c413371).
    """
    transfer = page.evaluate_handle("() => new DataTransfer()")
    dragged = source.element_handle()
    dragged.dispatch_event("dragstart", {"dataTransfer": transfer})
    box = target.bounding_box()
    assert box, "the drop target is not on screen"
    at_middle = {
        "dataTransfer": transfer,
        "clientX": box["x"] + box["width"] / 2,
        "clientY": box["y"] + box["height"] / 2,
    }
    target.dispatch_event("dragover", at_middle)
    target.dispatch_event("drop", at_middle)
    dragged.dispatch_event("dragend", {"dataTransfer": transfer})
