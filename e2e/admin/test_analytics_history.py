"""Journey: the analytics dashboard over replies spread across days, read the way an admin does.

A fresh user has replies on two presets of the scripted model: three on aster (two of them two
days ago, one today), one on birch two days ago and one on aster ten days ago, each with its own
token usage, and two of the chats carry a tag. Narrowed to that user's group, the dashboard counts
seven days or thirty depending on the period picked, the messages chart of the last 7 days counts
by the hour and shows each model's count and share for the hour the pointer is on, the last 24
hours switch it to 10-minute buckets, the token total is split into input and output, the model
table sorts by tokens and gives each model's share of the messages and a model's row opens an
overview with its chats' tags and a chat list that keeps to the period picked.

Discriminates: passes on dev 21f770fb7. In a frontend build of it whose chart hover box gives
every model the whole share, whose Input figure shows the total, whose model table ignores the
Total sort, whose model details ignore the period and whose last 24 hours and last 30 days both
ask for seven days, every test but the tag one fails; in a backend copy whose model overview
counts no tags the tag test fails (dev fbeee59e9).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from harness.access import make_group
from harness.actors import Actor
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from utils.analytics import (
    chart_interval,
    chart_point,
    close_model_details,
    count_by,
    dashboard,
    day_label,
    hour_label,
    model_row,
    open_model_details,
    period_picker,
    point_details,
    summary_figure,
)

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
    group_id: str
    group_name: str
    tag: str
    two_days_ago: int


def seed_reply(
    client, model_id: str, prompt: str, stamp: int, tokens: dict, chat_tag: str | None = None
) -> str:
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
    # one clock reading, so the replies of two days ago share the chart's hour
    now = int(time.time())
    two_days_ago = now - 2 * DAY
    with asker.client() as client:
        seed_reply(client, aster, "aster earlier one", two_days_ago, usage(10, 5), chat_tag=tag)
        seed_reply(client, aster, "aster earlier two", two_days_ago, usage(10, 5), chat_tag=tag)
        seed_reply(client, aster, "aster today", now, usage(10, 5))
        seed_reply(client, aster, "aster long ago", now - 10 * DAY, usage(10, 5))
        seed_reply(client, birch, "birch earlier", two_days_ago, usage(300, 200))
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    return History(asker, group_id, group_name, tag, two_days_ago)


@pytest.fixture
def admin_page(page_for, make_user) -> Page:
    """A fresh admin, so the period the dashboard remembers is this test's own."""
    return page_for(make_user(role="admin"), path="/admin/analytics")


def test_the_period_decides_whether_a_reply_from_ten_days_ago_counts(history, admin_page, presets):
    aster, _ = presets
    settings = dashboard(admin_page, history.group_name, "Last 7 days")
    expect(summary_figure(settings, "Messages")).to_have_text("4")
    expect(model_row(settings, aster).get_by_role("cell").nth(2)).to_have_text("3")

    period_picker(settings).select_option(label="Last 30 days")

    expect(summary_figure(settings, "Messages")).to_have_text("5")
    expect(model_row(settings, aster).get_by_role("cell").nth(2)).to_have_text("4")


def test_the_hourly_chart_shows_each_models_count_for_the_hour_under_the_pointer(
    history, admin_page, presets
):
    aster, birch = presets
    settings = dashboard(admin_page, history.group_name, "Last 7 days")
    expect(chart_interval(settings)).to_have_text("Hourly")
    count_by(settings, "Messages")

    hour = datetime.fromtimestamp(history.two_days_ago).replace(minute=0, second=0)
    heading = f"{day_label(hour)}, {hour:%Y} · {hour:%H:%M}"
    details = point_details(admin_page, chart_point(settings, hour_label(hour)), heading)
    expect(details.locator("div").filter(has_text=aster).last).to_contain_text(
        re.compile(r"2\s*\(66\.7%\)")
    )
    expect(details.locator("div").filter(has_text=birch).last).to_contain_text(
        re.compile(r"1\s*\(33\.3%\)")
    )


def test_the_last_24_hours_chart_counts_in_ten_minute_buckets(history, admin_page, presets):
    aster, birch = presets
    settings = dashboard(admin_page, history.group_name, "Last 24 hours")

    expect(chart_interval(settings)).to_have_text("10 min")
    expect(summary_figure(settings, "Messages")).to_have_text("1")
    expect(model_row(settings, birch)).to_have_count(0)
    expect(model_row(settings, aster)).to_have_count(1)


def test_the_token_total_is_split_into_input_and_output(history, admin_page):
    settings = dashboard(admin_page, history.group_name, "Last 30 days")

    expect(summary_figure(settings, "Tokens")).to_have_text("560")
    expect(summary_figure(settings, "Input")).to_have_text("340")
    expect(summary_figure(settings, "Output")).to_have_text("220")


def test_the_model_table_sorts_by_tokens_and_shows_each_models_share(history, admin_page, presets):
    aster, birch = presets
    settings = dashboard(admin_page, history.group_name, "Last 7 days")
    count_by(settings, "Messages")
    rows = settings.get_by_role("row").filter(has_text="history-")
    expect(rows.first).to_contain_text(aster)
    expect(model_row(settings, aster).get_by_role("cell").last).to_have_text("75.0%")
    expect(model_row(settings, birch).get_by_role("cell").last).to_have_text("25.0%")

    settings.get_by_role("columnheader", name="Total", exact=True).click()

    expect(rows.first).to_contain_text(birch)
    expect(model_row(settings, birch).get_by_role("cell").nth(7)).to_have_text("500")
    expect(model_row(settings, aster).get_by_role("cell").nth(7)).to_have_text("45")


def test_a_models_overview_counts_the_tags_of_its_chats(history, admin_page, presets):
    aster, birch = presets
    settings = dashboard(admin_page, history.group_name, "Last 30 days")

    details = open_model_details(settings, aster)

    expect(details.get_by_text(f"{history.tag} 2")).to_be_visible()
    close_model_details(details)
    other = open_model_details(settings, birch)
    expect(other.get_by_text(history.tag)).to_have_count(0)


def test_a_models_chat_list_keeps_to_the_period_picked(history, admin_page, presets):
    aster, _ = presets
    settings = dashboard(admin_page, history.group_name, "Last 7 days")
    details = open_model_details(settings, aster)
    details.get_by_role("button", name="Chats", exact=True).click()

    chats = details.get_by_role("link")
    expect(chats).to_have_count(3)
    expect(chats.filter(has_text="aster earlier one")).to_have_count(1)
    expect(chats.filter(has_text="aster today")).to_have_count(1)
    expect(chats.filter(has_text="aster long ago")).to_have_count(0)

    close_model_details(details)
    period_picker(settings).select_option(label="All time")
    details = open_model_details(settings, aster)
    details.get_by_role("button", name="Chats", exact=True).click()
    expect(details.get_by_role("link")).to_have_count(4)
    expect(details.get_by_role("link").filter(has_text="aster long ago")).to_have_count(1)
