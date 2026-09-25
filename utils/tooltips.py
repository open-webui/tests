"""Finding an icon-only button the way a person does: by the tooltip it shows."""

from __future__ import annotations

from playwright.sync_api import Locator


def tooltip_button(scope: Locator, tooltip: str) -> Locator:
    """The button in `scope` whose tooltip reads `tooltip`; hover first when it shows on hover."""
    buttons = scope.get_by_role("button")
    index = buttons.evaluate_all(
        "(buttons, tooltip) => buttons.findIndex("
        "(button) => button.parentElement?._tippy?.props.content === tooltip)",
        tooltip,
    )
    assert index >= 0, f"no button shows the tooltip {tooltip!r}"
    return buttons.nth(index)
