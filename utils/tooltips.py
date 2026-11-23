"""Finding an icon-only button the way a person does: by the tooltip it shows."""

from __future__ import annotations

import time

from playwright.sync_api import Locator

# as long as an `expect` waits for a button to show
TOOLTIP_WAIT_SECONDS = 5.0


def tooltip_button(scope: Locator, tooltip: str, role: str = "button") -> Locator:
    """The button (or other `role`) in `scope` whose tooltip reads `tooltip`; hover first when it
    shows on hover. Waits for the button, as a control the page draws once a list loads (the
    machine menu) can come after the chat input."""
    buttons = scope.get_by_role(role)
    deadline = time.monotonic() + TOOLTIP_WAIT_SECONDS
    while True:
        index = buttons.evaluate_all(
            "(buttons, tooltip) => buttons.findIndex("
            "(button) => button.parentElement?._tippy?.props.content === tooltip)",
            tooltip,
        )
        if index >= 0 or time.monotonic() > deadline:
            break
        scope.page.wait_for_timeout(100)
    assert index >= 0, f"no button shows the tooltip {tooltip!r}"
    return buttons.nth(index)
