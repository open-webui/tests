"""The calendar's month view, located the way a person sees it: each day is a square named by its
full date ("Friday, October 30, 2026"), showing the day's number and the chips of the events on
that day. The day's own button fills the square beside the chips, so the chips are not inside it.
"""

from __future__ import annotations

import datetime as dt
import re

from playwright.sync_api import Locator, Page

DAY_NAME = re.compile(r"^\w+, \w+ \d{1,2}, \d{4}$")


def day_button(page: Page, day: dt.date) -> Locator:
    """The button that fills the day's square and starts an event on that day."""
    return page.get_by_role("button", name=f"{day:%A}, {day:%B} {day.day}, {day.year}", exact=True)


def day_squares(page: Page) -> Locator:
    """Every day square of the month view, in grid order."""
    return page.get_by_role("button", name=DAY_NAME).locator("xpath=..")


def days_showing(page: Page, chip: Locator) -> list[int]:
    """The day numbers of the month-view squares holding the chip, in grid order."""
    squares = day_squares(page).filter(has=chip)
    return [int(text.split()[0]) for text in squares.all_inner_texts()]
