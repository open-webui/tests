"""The admin analytics dashboard, read the way an admin does.

Retargeted for 06c17834e, dc2ac8606 and 3539ad632, which rebuilt the dashboard: the totals are a
row of labelled figures (Tokens, Input, Output, Messages, Chats, Users), Tokens and Messages
switch what the chart and the share columns count (Tokens first), the chart's bucket size is the
"Chart interval" picker and each bucket is a point named by its time, whose hover box lists each
model's count and share.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def leaderboard_tab(settings: Locator, name: str, selected: bool | None = None) -> Locator:
    """The Models / Users switch above the leaderboard, shown once the dashboard has loaded."""
    return settings.get_by_role("button", name=name, exact=True, pressed=selected)


def period_picker(settings: Locator) -> Locator:
    return settings.get_by_role("combobox", name="Date range")


def group_picker(settings: Locator) -> Locator:
    return settings.get_by_role("combobox", name="User group")


def dashboard(page: Page, group_name: str | None = None, period: str | None = None) -> Locator:
    """The dashboard of a page opened on /admin/analytics, loaded and narrowed as given."""
    settings = page.get_by_role("dialog")
    expect(leaderboard_tab(settings, "Models", selected=True)).to_be_visible()
    if group_name:
        group_picker(settings).select_option(label=group_name)
    if period:
        period_picker(settings).select_option(label=period)
    return settings


def summary_figure(settings: Locator, label: str) -> Locator:
    """The value beside `label` in the row of totals above the chart."""
    term = settings.page.get_by_role("term").filter(has_text=re.compile(f"^{re.escape(label)}$"))
    return settings.locator("dl > div").filter(has=term).get_by_role("definition")


def count_by(settings: Locator, metric: str) -> None:
    """Switch the chart and the share columns to `metric`, Tokens or Messages."""
    switch = settings.get_by_role("button", name=metric, exact=True)
    with settings.page.expect_response(
        lambda response: (
            "/api/v1/analytics/daily?" in response.url
            and f"metric={metric.lower()}" in response.url
        )
    ):
        switch.click()
    expect(switch).to_have_attribute("aria-pressed", "true")


def chart_interval(settings: Locator) -> Locator:
    """The bucket size picked for the chart."""
    return settings.get_by_role("combobox", name="Chart interval").locator("option:checked")


def _reload_chart(settings: Locator, change, *query: str) -> None:
    """Make `change` and wait for the chart's data asked for with every `query` part."""
    with settings.page.expect_response(
        lambda response: (
            "/api/v1/analytics/daily?" in response.url
            and all(part in response.url for part in query)
        )
    ):
        change()


def pick_interval(settings: Locator, label: str, granularity: str) -> None:
    """Pick the chart's bucket size, which reloads the chart with `granularity`."""
    picker = settings.get_by_role("combobox", name="Chart interval")
    _reload_chart(settings, lambda: picker.select_option(label=label), f"granularity={granularity}")
    expect(chart_interval(settings)).to_have_text(label)


def compare_switch(settings: Locator) -> Locator:
    return settings.get_by_role("button", name="Compare", exact=True)


def compare_to_previous_period(settings: Locator) -> None:
    _reload_chart(settings, compare_switch(settings).click, "compare=true")
    expect(compare_switch(settings)).to_have_attribute("aria-pressed", "true")


def chart_view(settings: Locator, name: str) -> None:
    """Switch the chart between Volume (stacked bars) and Activity (a lane per model)."""
    switch = settings.get_by_role("button", name=name, exact=True)
    switch.click()
    expect(switch).to_have_attribute("aria-pressed", "true")


def legend_entry(settings: Locator, name: str) -> Locator:
    """The model's entry in the chart legend, which isolates the model when clicked."""
    return settings.get_by_role("button", name=f"{name} — Select to isolate model")


def activity_lane(settings: Locator, name: str) -> Locator:
    """The model's lane in the Activity view, named by the model and the day under focus."""
    return settings.get_by_role("button", name=re.compile(f"^{re.escape(name)}, "))


def chart_point(settings: Locator, label: str) -> Locator:
    """The chart's bar for the bucket the axis writes as `label`."""
    return settings.get_by_role("button", name=re.compile(f"^{re.escape(label)}: "))


def hour_label(hour) -> str:
    """How the chart names an hour's bucket (dayjs `MMM D, HH:mm`)."""
    return f"{MONTHS[hour.month - 1]} {hour.day}, {hour:%H:%M}"


def day_label(day) -> str:
    """How the chart names a day's bucket (dayjs `MMM D`)."""
    return f"{MONTHS[day.month - 1]} {day.day}"


def point_details(page: Page, point: Locator, heading: str) -> Locator:
    """Hover `point` and return its hover box, whose first line starts with `heading`."""
    point.hover()
    details = page.get_by_role("tooltip").filter(has_text=heading)
    expect(details).to_be_visible()
    return details


def model_row(settings: Locator, name: str) -> Locator:
    return settings.get_by_role("row").filter(has_text=name)


def model_details(page: Page) -> Locator:
    """The model's details; Settings also has an Overview switch, but tabs as well."""
    return (
        page.get_by_role("dialog")
        .filter(has=page.get_by_role("button", name="Overview", exact=True))
        .filter(has_not=page.get_by_role("tab"))
    )


def open_model_details(settings: Locator, name: str) -> Locator:
    """Click the model's row and return its details, opened on the Overview tab."""
    model_row(settings, name).click()
    details = model_details(settings.page)
    expect(details.get_by_text("Feedback Activity")).to_be_visible()
    return details


def close_model_details(details: Locator) -> None:
    details.get_by_role("button", name="Close").last.click()
    expect(details).to_be_hidden()
