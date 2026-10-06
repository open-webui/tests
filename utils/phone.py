"""A phone in the browser: a 390 by 844 touch screen, taps and swipes, and what fits on it.

`PHONE` is the browser context of a phone (`page_for(actor, **PHONE)`): a narrow touch screen with
a phone's user agent, where the sidebar becomes a drawer and Enter in the message box starts a new
line. `expect_on_screen` asserts an element lies inside the screen and a tap on its centre reaches
it, with nothing scrolled first (a tall area only needs its top on the screen). `expect_reachable`
scrolls it into view first, the way a finger scrolls a list or a tab strip, and fails when that
took scrolling a box a finger cannot scroll. `expect_off_screen` asserts it is hidden or pushed
off the screen. `swipe` drags one finger across the screen and `send_by_tapping` types a message
and taps send.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

SCREEN = {"width": 390, "height": 844}

PHONE = {
    "viewport": SCREEN,
    "is_mobile": True,
    "has_touch": True,
    "device_scale_factor": 3,
    "user_agent": (
        "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Mobile Safari/537.36"
    ),
}

SETTLE_MS = 5_000

# half a pixel of slack for fractional layout; an area taller than half the screen may run past
# its foot, and is tapped in the middle of the part that shows
ON_SCREEN = """(element) => {
    const box = element.getBoundingClientRect();
    const tall = box.height > window.innerHeight / 2;
    const inside = box.width > 0 && box.height > 0 && box.left >= -0.5 && box.top >= -0.5
        && box.right <= window.innerWidth + 0.5
        && (tall ? box.top < window.innerHeight : box.bottom <= window.innerHeight + 0.5);
    const bottom = Math.min(box.bottom, window.innerHeight);
    const hit = document.elementFromPoint(box.left + box.width / 2, (box.top + bottom) / 2);
    return inside && element.contains(hit);
}"""

OFF_SCREEN = """(element) => {
    const box = element.getBoundingClientRect();
    return box.width === 0 || box.right <= 0.5 || box.left >= window.innerWidth - 0.5
        || box.bottom <= 0.5 || box.top >= window.innerHeight - 0.5;
}"""

# a container a finger cannot scroll, yet scrolled to bring the element into view
LOCKED_SCROLLER = """(element) => {
    const page = [document.body, document.documentElement];
    for (let box = element.parentElement; box && !page.includes(box); box = box.parentElement) {
        const style = getComputedStyle(box);
        const sideways = box.scrollLeft > 0 && !/(auto|scroll)/.test(style.overflowX);
        const downwards = box.scrollTop > 0 && !/(auto|scroll)/.test(style.overflowY);
        if (sideways || downwards) return box.outerHTML.slice(0, 120);
    }
    return null;
}"""

WHERE = """(element) => {
    const box = element.getBoundingClientRect();
    const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    const place = [box.left, box.top, box.right, box.bottom].map(Math.round).join(', ');
    return `at ${place} on a ${window.innerWidth}x${window.innerHeight} screen, a tap on its `
        + `centre lands on ${hit ? hit.outerHTML.slice(0, 120) : 'nothing'}`;
}"""


def _wait_until(locator: Locator, check: str, failure: str) -> None:
    try:
        locator.page.wait_for_function(check, arg=locator.element_handle(), timeout=SETTLE_MS)
    except PlaywrightTimeoutError:
        raise AssertionError(f"{failure}: {locator} is {locator.evaluate(WHERE)}") from None


def expect_on_screen(locator: Locator) -> None:
    expect(locator).to_be_visible()
    _wait_until(locator, ON_SCREEN, "not on the screen or covered")


def expect_reachable(locator: Locator) -> None:
    expect(locator).to_be_visible()
    locator.scroll_into_view_if_needed()
    _wait_until(locator, ON_SCREEN, "not reachable by scrolling")
    locked = locator.evaluate(LOCKED_SCROLLER)
    assert locked is None, f"{locator} only shows by scrolling what a finger cannot: {locked}"


def expect_off_screen(locator: Locator) -> None:
    if locator.count() == 0 or not locator.is_visible():
        return
    _wait_until(locator, OFF_SCREEN, "still on the screen")


def tap_on_screen(locator: Locator) -> None:
    expect_on_screen(locator)
    locator.tap()


def swipe(
    page: Page, start: tuple[float, float], end: tuple[float, float], steps: int = 10
) -> None:
    """One finger pressed at `start`, dragged to `end` and lifted."""
    session = page.context.new_cdp_session(page)
    try:
        session.send(
            "Input.dispatchTouchEvent",
            {"type": "touchStart", "touchPoints": [{"x": start[0], "y": start[1]}]},
        )
        for step in range(1, steps + 1):
            x = start[0] + (end[0] - start[0]) * step / steps
            y = start[1] + (end[1] - start[1]) * step / steps
            session.send(
                "Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y}]}
            )
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    finally:
        session.detach()


def send_button(page: Page) -> Locator:
    return tooltip_button(page.locator("form"), "Send message")


def send_by_tapping(page: Page, text: str) -> None:
    tap_on_screen(chat_input(page))
    page.keyboard.type(text)
    tap_on_screen(send_button(page))
