"""Journey: the admin analytics dashboard counts what two users did with two models.

Two fresh users chat with two presets of the scripted model, each reply carrying a token count.
The dashboard's model table then shows each preset's messages, users, chats and tokens, the user
table (narrowed to the two users' group) shows each user's messages and tokens with the totals
above it, a custom date range from before the chats empties the dashboard and the last day
brings it back, a model row opens the model's chats with the user who had each and, with a
group picked, only the chats of that group's users (#33207), and a plain user is sent back to the
chat by the analytics page and refused its API.

Discriminates: passes on dev 21f770fb7 (the user table and date range mutations proven there, the
rest on 176d31d1d); in a backend copy, with `/api/v1/analytics/models` reporting `unique_users=0`
the model table test fails, with `/api/v1/analytics/users` reporting `total_tokens=0` the user
table test fails, with `/api/v1/analytics/models` ignoring the dates the date range test fails
(the row survives a range before the chats), with the model chats route dropping `user_name` the
drill-down test fails, and with `/api/v1/analytics/summary` open to verified users the access test
fails (HTTP 200). On dev 8889e8d6a, with the dashboard no longer passing the group to the model's
details (02821c505's frontend half reverted) the group drill-down test fails (both chats listed).
"""

from __future__ import annotations

import time
import uuid
from datetime import date, timedelta
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID
from utils.analytics import (
    group_picker,
    leaderboard_tab,
    open_model_details,
    period_picker,
    summary_figure,
)
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@pytest.fixture
def presets(admin) -> Iterator[tuple[str, str]]:
    """Two presets only this module chats with, so their counts are the test's own."""
    names = tuple(f"analytics-{word}-{uuid.uuid4().hex[:6]}" for word in ("aster", "birch"))
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


class Activity:
    """What the two users did, and the group holding just them."""

    def __init__(self, first: Actor, second: Actor, group_name: str, chat_ids: dict[str, str]):
        self.first = first
        self.second = second
        self.group_name = group_name
        self.chat_ids = chat_ids


@pytest.fixture
def activity(admin, make_user, upstream, presets) -> Activity:
    """Four replies: the first user two on aster in one chat and one on birch, the second one."""
    aster, birch = presets
    first = make_user(name=f"Analytics First {uuid.uuid4().hex[:6]}")
    second = make_user(name=f"Analytics Second {uuid.uuid4().hex[:6]}")
    group_id = make_group(admin, [first, second])
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]

    chat_ids = {}
    with first.client() as client:
        upstream.queue(
            reply.text("aster one", usage=_usage(100, 10), match=reply.answering("aster first"))
        )
        turn, _ = ask(client, "aster first", model=aster)
        chat_ids["aster"] = turn.chat_id
        upstream.queue(
            reply.text("aster two", usage=_usage(100, 20), match=reply.answering("aster again"))
        )
        ask(
            client,
            "aster again",
            model=aster,
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
            history=[
                {"role": "user", "content": "aster first"},
                {"role": "assistant", "content": "aster one"},
            ],
        )
        upstream.queue(
            reply.text("birch one", usage=_usage(100, 30), match=reply.answering("birch first"))
        )
        turn, _ = ask(client, "birch first", model=birch)
        chat_ids["birch by first"] = turn.chat_id
    with second.client() as client:
        upstream.queue(
            reply.text("birch two", usage=_usage(100, 40), match=reply.answering("birch second"))
        )
        turn, _ = ask(client, "birch second", model=birch)
        chat_ids["birch by second"] = turn.chat_id
    return Activity(first, second, group_name, chat_ids)


@pytest.fixture
def dashboard(page_for, make_user) -> tuple[Page, Locator]:
    """A fresh admin on the analytics dashboard, loaded."""
    page = page_for(make_user(role="admin"), path="/admin/analytics")
    settings = page.get_by_role("dialog")
    expect(leaderboard_tab(settings, "Models", selected=True)).to_be_visible()
    return page, settings


def _row(settings: Locator, name: str) -> Locator:
    return settings.get_by_role("row").filter(has_text=name)


def _cells(row: Locator, *expected: str) -> None:
    """The row's cells after the rank and name, in table order."""
    for offset, value in enumerate(expected):
        expect(row.get_by_role("cell").nth(2 + offset)).to_have_text(value)


def test_the_model_table_counts_each_presets_messages_users_chats_and_tokens(
    activity, dashboard, presets
):
    _, settings = dashboard
    aster, birch = presets
    expect(_row(settings, aster)).to_have_count(1)
    # messages, users, chats, then input, output and total tokens
    _cells(_row(settings, aster), "2", "1", "1")
    _cells(_row(settings, birch), "2", "2", "2")
    expect(_row(settings, aster).get_by_role("cell").nth(7)).to_have_text("230")
    expect(_row(settings, birch).get_by_role("cell").nth(7)).to_have_text("270")


def test_the_user_table_and_summary_count_the_groups_messages_and_tokens(activity, dashboard):
    _, settings = dashboard
    group_picker(settings).select_option(label=activity.group_name)
    leaderboard_tab(settings, "Users").click()
    expect(_row(settings, activity.first.name)).to_have_count(1)
    # messages, then input, output and total tokens
    _cells(_row(settings, activity.first.name), "3")
    _cells(_row(settings, activity.second.name), "1")
    expect(_row(settings, activity.first.name).get_by_role("cell").nth(5)).to_have_text("360")
    expect(_row(settings, activity.second.name).get_by_role("cell").nth(5)).to_have_text("140")
    expect(summary_figure(settings, "Messages")).to_have_text("4")
    expect(summary_figure(settings, "Tokens")).to_have_text("500")
    expect(summary_figure(settings, "Chats")).to_have_text("3")
    expect(summary_figure(settings, "Users")).to_have_text("2")


def test_a_date_range_before_the_chats_empties_the_dashboard(activity, dashboard, presets):
    _, settings = dashboard
    aster, _ = presets
    group_picker(settings).select_option(label=activity.group_name)
    expect(_row(settings, aster)).to_have_count(1)

    period_picker(settings).select_option(label="Custom range")
    start, end = settings.get_by_label("Start date"), settings.get_by_label("End date")
    # a whole day of margin either side, so no time zone puts the chats inside the range
    start.fill((date.today() - timedelta(days=9)).isoformat())
    end.fill((date.today() - timedelta(days=3)).isoformat())
    expect(summary_figure(settings, "Messages")).to_have_text("0")
    expect(_row(settings, aster)).to_have_count(0)
    expect(settings.get_by_text("No data")).to_have_count(1)
    leaderboard_tab(settings, "Users").click()
    expect(settings.get_by_text("No data")).to_have_count(1)
    leaderboard_tab(settings, "Models").click()

    period_picker(settings).select_option(label="Last 24 hours")
    expect(_row(settings, aster)).to_have_count(1)
    expect(summary_figure(settings, "Messages")).to_have_text("4")


def test_a_model_row_opens_the_models_chats_with_who_had_them(activity, dashboard, presets):
    page, settings = dashboard
    _, birch = presets
    details = open_model_details(settings, birch)
    expect(details.get_by_text(birch)).to_be_visible()
    details.get_by_role("button", name="Chats", exact=True).click()
    chats = details.get_by_role("link")
    expect(chats).to_have_count(2)
    expect(chats.filter(has_text="birch first")).to_have_attribute(
        "href", f"/s/{activity.chat_ids['birch by first']}"
    )
    expect(chats.filter(has_text="birch second")).to_have_attribute(
        "href", f"/s/{activity.chat_ids['birch by second']}"
    )
    expect(details.get_by_text(activity.first.name)).to_be_visible()
    expect(details.get_by_text(activity.second.name)).to_be_visible()
    expect(details.get_by_text("aster first")).to_have_count(0)


@pytest.fixture
def first_users_group(admin, activity) -> str:
    """A group holding only the first user, made before the dashboard loads its groups."""
    group_id = make_group(admin, [activity.first])
    with admin.client() as client:
        return client.get(f"/api/v1/groups/id/{group_id}").json()["name"]


@pytest.mark.regression
def test_a_models_chats_keep_to_the_user_group_picked(
    activity, first_users_group, dashboard, presets
):
    """#33207 (fix 02821c505): the Chats tab listed every user's chats with a group picked."""
    _, settings = dashboard
    _, birch = presets
    group_picker(settings).select_option(label=first_users_group)
    # the table follows the group before the row is opened
    expect(_row(settings, birch).get_by_role("cell").nth(3)).to_have_text("1")

    details = open_model_details(settings, birch)
    details.get_by_role("button", name="Chats", exact=True).click()
    chats = details.get_by_role("link")
    expect(chats.filter(has_text="birch first")).to_have_count(1)
    expect(chats).to_have_count(1)
    expect(details.get_by_text(activity.second.name)).to_have_count(0)


def test_a_user_is_sent_back_to_the_chat_and_refused_the_api(page_for, make_user):
    user = make_user()
    page = page_for(user, path="/admin/analytics")
    expect(chat_input(page)).to_be_visible()
    expect(page).to_have_url(f"{user.base_url}/")
    expect(page.get_by_role("dialog")).to_have_count(0)

    page.goto("/?settings=admin%3Aanalytics")
    expect(page.get_by_role("dialog").get_by_role("tab", selected=True)).to_be_visible()
    expect(leaderboard_tab(page.get_by_role("dialog"), "Models")).to_have_count(0)

    now = int(time.time())
    with user.client() as client:
        refused = client.get(
            "/api/v1/analytics/summary", params={"start_date": now - 3600, "end_date": now}
        )
    assert refused.status_code == 401, refused.text
