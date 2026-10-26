"""Regression: one message without a timestamp broke the analytics chart for All time.

Fix f25708484 (PR #31888, issue #27316). A chat stored through the API with a null message
`timestamp` wrote the first such message with no time and failed on the rest, so the replies after
it never reached analytics. A message row left without a time then failed
`GET /api/v1/analytics/daily` without a date range (the dashboard's All time) with a 500, as every
row's time is turned into a date. Since dev 7b303cf49 such a message is stored with its time
unknown: the chart, which places every message on a day, leaves it out, while the All time totals
per model and in the summary still count it, and any dated window leaves it out too. A row already
stored with no time at all is treated the same.

Discriminates: passes on dev 8889e8d6a; in a backend copy, with the chart's day bucket taking the
raw time unchecked and no longer skipping rows of unknown time (the #27316 shape) both All time
tests fail (500 on the row with no time, a day of 1970 for the untimed reply), with only the skip
dropped they fail with the reply put on today, and with an untimed message stored at the time it
was saved the All time and dated-window tests fail (the reply counts in the last day).
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Iterator

import pytest

from harness.access import make_group
from harness.actors import Actor
from harness.backends import write_rows
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
DAY = 24 * 3600


@pytest.fixture
def counted_model(admin) -> Iterator[str]:
    """A preset only this test's chats name, so its counts are the test's own."""
    model_id = f"untimed-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {},
                "access_grants": PUBLIC_READ,
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def owner(make_user) -> Iterator[Actor]:
    """A fresh account whose chats are deleted afterwards, untimed rows included."""
    account = make_user()
    yield account
    with account.client() as client:
        client.delete("/api/v1/chats/")


def seed_answered(owner: Actor, model_id: str, timestamp: int | None) -> str:
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "a question", "timestamp": timestamp},
                {"role": "assistant", "content": "an answer", "timestamp": timestamp},
            ],
            model=model_id,
        )
    return chat_id


def analytics(admin: Actor, path: str, **params) -> dict:
    with admin.client() as client:
        response = client.get(f"/api/v1/analytics/{path}", params=params)
    assert response.status_code == 200, f"{path} failed to load: {response.text}"
    return response.json()


def daily_counts(admin: Actor, model_id: str, **window) -> dict[str, int]:
    return {
        entry["date"]: entry["models"][model_id]
        for entry in analytics(admin, "daily", **window)["data"]
        if model_id in entry["models"]
    }


def model_count(admin: Actor, model_id: str, **window) -> int:
    models = analytics(admin, "models", **window)["models"]
    return sum(entry["count"] for entry in models if entry["model_id"] == model_id)


def last_day() -> dict[str, int]:
    now = int(time.time())
    return {"start_date": now - DAY, "end_date": now + 3600}


def test_all_time_loads_and_counts_a_message_saved_without_a_timestamp(admin, owner, counted_model):
    seed_answered(owner, counted_model, timestamp=None)

    assert daily_counts(admin, counted_model) == {}, "a reply of unknown time was put on a day"
    assert model_count(admin, counted_model) == 1

    group_id = make_group(admin, [owner])
    summary = analytics(admin, "summary", group_id=group_id)
    assert (summary["total_messages"], summary["total_chats"]) == (1, 1), summary


def test_a_message_saved_without_a_timestamp_stays_out_of_the_last_day(admin, owner, counted_model):
    seed_answered(owner, counted_model, timestamp=None)

    assert daily_counts(admin, counted_model, **last_day()) == {}
    assert model_count(admin, counted_model, **last_day()) == 0
    # the same reply is there once no window applies
    assert model_count(admin, counted_model) == 1


def test_all_time_loads_with_a_message_already_stored_without_a_time(
    instance, admin, owner, counted_model
):
    chat_id = seed_answered(owner, counted_model, timestamp=int(time.time()))
    write_rows(
        instance,
        "UPDATE chat_message SET created_at = NULL WHERE chat_id = :chat_id",
        [{"chat_id": chat_id}],
    )

    assert daily_counts(admin, counted_model) == {}, "a reply with no time was put on a day"
    assert model_count(admin, counted_model) == 1


# ---------------------------------------------------------------- nearby


def test_a_timestamped_message_still_counts_on_its_own_day(admin, owner, counted_model):
    three_days_ago = int(time.time()) - 3 * DAY
    seed_answered(owner, counted_model, timestamp=three_days_ago)

    expected_day = date.fromtimestamp(three_days_ago).isoformat()
    assert daily_counts(admin, counted_model) == {expected_day: 1}
    assert daily_counts(admin, counted_model, **last_day()) == {}
