"""Journey: the analytics dashboard over replies spread across days, read the way an admin does.

A fresh user has replies on two presets of the scripted model: three on aster (two of them two
days ago, one today), one on birch two days ago and one on aster ten days ago, each with its own
token usage, and two of the chats carry a tag. Narrowed to that user's group, the dashboard counts
seven days or thirty depending on the period picked, the Daily Messages chart shows each model's
count for the day the pointer is on, the last 24 hours switch it to Hourly Messages, the tokens
figure opens the input and output split, the model table sorts by tokens and gives each model's
share of the messages, and a model's row opens an overview with its chats' tags and a chat list
that keeps to the period picked.

Discriminates: passes on dev ebc6add67. In a frontend build whose chart tooltip leaves out the
counts, whose tokens tooltip shows the total as the input, whose model table ignores the Tokens
sort and gives every model 100%, whose model details ignore the period and whose last 24 hours and
last 30 days both ask for seven days, every test but the tag one fails; in a backend copy whose
model overview counts no tags the tag test fails.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.actors import Actor
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
DAY = 24 * 3600


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@pytest.fixture
def presets(admin) -> Iterator[tuple[str, str]]:
    """Two presets only this module chats with, so their counts are the test's own."""
    names = tuple(f"history-{word}-{uuid.uuid4().hex[:6]}" for word in ("aster", "birch"))
    with admin.client() as client:
        for name in names:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": name,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": name,
                    "meta": {},
                    "params": {},
                    "access_grants": PUBLIC_READ,
                },
            )
            assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        yield names
        for name in names:
            client.post("/api/v1/models/model/delete", json={"id": name})


@dataclass
class History:
    asker: Actor
    group_name: str
    tag: str


def seed_reply(
    client, model_id: str, prompt: str, days_ago: int, tokens: dict, chat_tag: str | None = None
) -> str:
    stamp = int(time.time()) - days_ago * DAY
    chat_id, _ = seed_chat(
        client,
        [
            {"role": "user", "content": prompt, "timestamp": stamp},
            {"role": "assistant", "content": "noted", "timestamp": stamp, "usage": tokens},
        ],
        model=model_id,
    )
    renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": prompt}})
    assert renamed.status_code == 200, renamed.text
    if chat_tag:
        tagged = client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": chat_tag})
        assert tagged.status_code == 200, tagged.text
    return chat_id


@pytest.fixture
def history(admin, make_user, presets) -> History:
    """Aster: 2 replies two days ago, 1 today, 1 ten days ago; birch: 1 two days ago."""
    aster, birch = presets
    asker = make_user(name=f"History Asker {uuid.uuid4().hex[:6]}")
    group_id = make_group(admin, [asker])
    tag = f"harbour{uuid.uuid4().hex[:6]}"
    with asker.client() as client:
        seed_reply(client, aster, "aster earlier one", 2, usage(10, 5), chat_tag=tag)
        seed_reply(client, aster, "aster earlier two", 2, usage(10, 5), chat_tag=tag)
        seed_reply(client, aster, "aster today", 0, usage(10, 5))
        seed_reply(client, aster, "aster long ago", 10, usage(10, 5))
        seed_reply(client, birch, "birch earlier", 2, usage(300, 200))
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    return History(asker, group_name, tag)


def open_dashboard(page: Page, group_name: str, period: str) -> Locator:
    page.goto("/admin/analytics")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("User Activity", exact=True)).to_be_visible()
    settings.get_by_role("combobox").filter(has_text="All Users").select_option(label=group_name)
    settings.get_by_role("combobox").filter(has_text="Custom range").select_option(label=period)
    return settings


@pytest.fixture
def admin_page(page_for, make_user) -> Page:
    """A fresh admin, so the period the dashboard remembers is this test's own."""
    return page_for(make_user(role="admin"))


def model_row(settings: Locator, name: str) -> Locator:
    return settings.get_by_role("row").filter(has_text=name)


def test_the_period_decides_whether_a_reply_from_ten_days_ago_counts(history, admin_page, presets):
    aster, _ = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 7 days")
    expect(settings.get_by_text("4 messages")).to_be_visible()
    expect(model_row(settings, aster).get_by_role("cell").nth(2)).to_have_text("3")

    settings.get_by_role("combobox").filter(has_text="Last 7 days").select_option(
        label="Last 30 days"
    )

    expect(settings.get_by_text("5 messages")).to_be_visible()
    expect(model_row(settings, aster).get_by_role("cell").nth(2)).to_have_text("4")


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def chart_tooltip(settings: Locator, day: date) -> Locator:
    """The chart's hover box for `day`, which dayjs labels `MMM D, YYYY`."""
    label = f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
    return settings.locator("div.pointer-events-none").filter(has_text=label)


def hover_chart_point(settings: Locator, index: int, points: int) -> None:
    """Move the pointer over the chart's `index`-th point of `points`."""
    chart = settings.locator("svg[preserveAspectRatio=none]")
    box = chart.bounding_box()
    assert box is not None
    chart.hover(position={"x": box["width"] * index / (points - 1), "y": box["height"] / 2})


def test_the_daily_chart_shows_each_models_count_for_the_day_under_the_pointer(
    history, admin_page, presets
):
    aster, birch = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 7 days")
    expect(settings.get_by_text("Daily Messages")).to_be_visible()

    # the 7-day chart has a point for each of the eight days it touches, today last
    hover_chart_point(settings, 5, 8)

    tooltip = chart_tooltip(settings, date.fromtimestamp(time.time() - 2 * DAY))
    expect(tooltip).to_be_visible()
    expect(tooltip.locator("div").filter(has_text=aster).last).to_contain_text(
        re.compile(r"2\s*\(67%\)")
    )
    expect(tooltip.locator("div").filter(has_text=birch).last).to_contain_text(
        re.compile(r"1\s*\(33%\)")
    )


def test_the_last_24_hours_chart_counts_by_the_hour(history, admin_page, presets):
    aster, birch = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 24 hours")

    expect(settings.get_by_text("Hourly Messages")).to_be_visible()
    expect(settings.get_by_text("Daily Messages")).to_have_count(0)
    expect(settings.get_by_text("1 messages")).to_be_visible()
    expect(model_row(settings, birch)).to_have_count(0)
    expect(model_row(settings, aster)).to_have_count(1)


def test_the_tokens_figure_opens_the_input_and_output_split(history, admin_page):
    settings = open_dashboard(admin_page, history.group_name, "Last 30 days")
    tokens = settings.get_by_role("button").filter(has_text="tokens")
    expect(tokens).to_contain_text("560")

    tokens.hover()

    split = admin_page.get_by_role("tooltip").filter(has_text="Token counts are estimates")
    expect(split).to_be_visible()
    expect(split).to_contain_text(re.compile(r"Input\s*340"))
    expect(split).to_contain_text(re.compile(r"Output\s*220"))


def test_the_model_table_sorts_by_tokens_and_shows_each_models_share(history, admin_page, presets):
    aster, birch = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 7 days")
    rows = settings.get_by_role("row").filter(has_text="history-")
    expect(rows.first).to_contain_text(aster)
    expect(model_row(settings, aster).get_by_role("cell").last).to_have_text("75.0%")
    expect(model_row(settings, birch).get_by_role("cell").last).to_have_text("25.0%")

    settings.get_by_role("columnheader", name="Tokens").first.click()

    expect(rows.first).to_contain_text(birch)
    expect(model_row(settings, birch).get_by_role("cell").nth(5)).to_have_text("500")
    expect(model_row(settings, aster).get_by_role("cell").nth(5)).to_have_text("45")


def open_model_details(page: Page, settings: Locator, name: str) -> Locator:
    model_row(settings, name).click()
    details = page.get_by_role("dialog").filter(has=page.get_by_role("button", name="Overview"))
    expect(details.get_by_text("Feedback Activity")).to_be_visible()
    return details


def test_a_models_overview_counts_the_tags_of_its_chats(history, admin_page, presets):
    aster, birch = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 30 days")

    details = open_model_details(admin_page, settings, aster)

    expect(details.get_by_text(f"{history.tag} 2")).to_be_visible()
    details.get_by_role("button", name="Close").last.click()
    other = open_model_details(admin_page, settings, birch)
    expect(other.get_by_text(history.tag)).to_have_count(0)


def test_a_models_chat_list_keeps_to_the_period_picked(history, admin_page, presets):
    aster, _ = presets
    settings = open_dashboard(admin_page, history.group_name, "Last 7 days")
    details = open_model_details(admin_page, settings, aster)
    details.get_by_role("button", name="Chats", exact=True).click()

    chats = details.get_by_role("link")
    expect(chats).to_have_count(3)
    expect(chats.filter(has_text="aster earlier one")).to_have_count(1)
    expect(chats.filter(has_text="aster today")).to_have_count(1)
    expect(chats.filter(has_text="aster long ago")).to_have_count(0)

    details.get_by_role("button", name="Close").last.click()
    settings.get_by_role("combobox").filter(has_text="Last 7 days").select_option(label="All time")
    details = open_model_details(admin_page, settings, aster)
    details.get_by_role("button", name="Chats", exact=True).click()
    expect(details.get_by_role("link")).to_have_count(4)
    expect(details.get_by_role("link").filter(has_text="aster long ago")).to_have_count(1)
