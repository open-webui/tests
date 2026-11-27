"""The calendar's week and day views, located the way a person sees them: each day's column is a
stack of hour slots, each a button named by its day and hour ("Tuesday, November 24 at 9 AM")
that starts an event at that hour.
"""

from __future__ import annotations

import datetime as dt
import re

from playwright.sync_api import Locator, Page


def hour_slot(page: Page, day: dt.date, hour: int) -> Locator:
    """The button of the day's hour, 0 to 23, in the week or day view."""
    clock = dt.time(hour).strftime("%-I %p").replace(" ", r"\s?")
    name = re.compile(rf"^{day:%A}, {day:%B} {day.day} at {clock}$")
    return page.get_by_role("button", name=name)
