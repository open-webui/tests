"""The hover toolbar of a channel message, found the way a person does: hover the message, then pick
the button the pointer can reach there by the tooltip it shows.

Since open-webui 08f197ad8 the toolbar floats in the page outside the message it acts on, so it is
found by where it shows (on the hovered message, reachable by the pointer), not inside the message.
"""

from __future__ import annotations

from playwright.sync_api import Locator

REACHABLE_TOOLTIPS = """(buttons, message) => {
    const box = message.getBoundingClientRect();
    return buttons.map((button) => {
        const rect = button.getBoundingClientRect();
        const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
        const reachable = button.contains(hit) && rect.bottom > box.top && rect.top < box.bottom;
        return reachable ? (button.parentElement?._tippy?.props.content ?? null) : null;
    });
}"""


def message_tooltips(message: Locator) -> list[str | None]:
    """Each page button's tooltip, or None where a person cannot reach it on the hovered message."""
    message.hover()
    return message.page.get_by_role("button").evaluate_all(
        REACHABLE_TOOLTIPS, message.element_handle()
    )


def message_action(message: Locator, tooltip: str) -> Locator:
    """The button on the hovered `message` whose tooltip reads `tooltip`."""
    tooltips = message_tooltips(message)
    assert tooltip in tooltips, (
        f"hovering the message shows no button with the tooltip {tooltip!r}, "
        f"only {[shown for shown in tooltips if shown]}"
    )
    return message.page.get_by_role("button").nth(tooltips.index(tooltip))
