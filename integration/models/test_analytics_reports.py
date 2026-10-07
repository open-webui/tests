"""Journey: the admin analytics API reports what real chats put behind it, number for number.

Replies are stored with a chosen day, model and token usage, and the admin reads them back
through every analytics report, always narrowed to the test's own accounts or presets. The token
reports split input and output per model and per user (both usage namings, a reply without usage
adds no tokens), the daily series puts each reply on its day and fills the empty days, a window
that starts after the oldest reply drops it from every report, the message list filters by model,
user and chat, the model drill-down lists, orders and pages the chats inside a window, the model
overview counts the chats' tags and the rated thumbs per day and a plain user is refused every
route.

Discriminates: passes on dev ebc6add67; in a backend copy, with the input and output token columns
swapped the token test fails, with the daily series not filling empty days the daily test fails,
with the model chats route ignoring the start date the drill-down window test fails, with the tag
counts doubled the overview tag test fails, with the lost count of the feedback history dropped
both overview history tests fail and with the daily route open to verified users the refusal test
fails for that route only. With every route open to verified users each refusal case fails, and
with the summary ignoring the start date, the chat id filter answering nothing, ordering by user
name dropped and paging ignoring `skip`, the window, message filter, ordering and paging tests fail.
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Callable, Iterator

import pytest

from harness.access import make_group
from harness.actors import Actor
from harness.backends import write_rows
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
DAY = 24 * 3600


@pytest.fixture
def make_preset(admin) -> Iterator[Callable[[], str]]:
    """Presets only this test's chats name, so their numbers are the test's own."""
    created: list[str] = []

    def make() -> str:
        model_id = f"report-{uuid.uuid4().hex[:8]}"
        with admin.client() as client:
            response = client.post(
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
        assert response.status_code == 200, response.text
        created.append(model_id)
        return model_id

    yield make
    with admin.client() as client:
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def accounts(make_user) -> Iterator[Callable[..., Actor]]:
    """Fresh accounts whose chats are deleted afterwards."""
    made: list[Actor] = []

    def make(**options) -> Actor:
        account = make_user(**options)
        made.append(account)
        return account

    yield make
    for account in made:
        with account.client() as client:
            client.delete("/api/v1/chats/")


def seed_reply(
    owner: Actor,
    model_id: str,
    answer: str,
    days_ago: int = 0,
    usage: dict | None = None,
) -> str:
    """One question and its answer stored on a day in the past; returns the chat id."""
    stamp = int(time.time()) - days_ago * DAY
    reply: dict = {"role": "assistant", "content": answer, "timestamp": stamp}
    if usage is not None:
        reply["usage"] = usage
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": f"asking {answer}", "timestamp": stamp}, reply],
            model=model_id,
        )
    return chat_id


def report(admin: Actor, path: str, **params) -> dict | list:
    with admin.client() as client:
        response = client.get(f"/api/v1/analytics/{path}", params=params)
    assert response.status_code == 200, (
        f"{path} failed: HTTP {response.status_code} {response.text}"
    )
    return response.json()


def window(start_days_ago: float, end_days_ago: float = 0) -> dict[str, int]:
    now = int(time.time())
    return {
        "start_date": int(now - start_days_ago * DAY),
        "end_date": int(now - end_days_ago * DAY) + 3600,
    }


def day_of(days_ago: int) -> str:
    return date.fromtimestamp(time.time() - days_ago * DAY).isoformat()


# ---------------------------------------------------------------- token usage


def test_token_reports_split_input_and_output_per_model_and_per_user(admin, accounts, make_preset):
    first, second = accounts(), accounts()
    aster, birch = make_preset(), make_preset()
    group_id = make_group(admin, [first, second])
    seed_reply(first, aster, "one", usage={"prompt_tokens": 100, "completion_tokens": 10})
    seed_reply(first, birch, "two", usage={"input_tokens": 200, "output_tokens": 20})
    seed_reply(second, aster, "three", usage={"input_tokens": 50, "output_tokens": 5})
    seed_reply(second, aster, "four")

    tokens = report(admin, "tokens", group_id=group_id)
    by_model = {entry["model_id"]: entry for entry in tokens["models"]}
    # the reply without usage is one of the three messages but adds no tokens
    assert by_model[aster] == {
        "model_id": aster,
        "input_tokens": 150,
        "output_tokens": 15,
        "total_tokens": 165,
        "message_count": 3,
    }
    assert by_model[birch]["input_tokens"] == 200
    assert by_model[birch]["output_tokens"] == 20
    assert set(by_model) == {aster, birch}
    assert (tokens["total_input_tokens"], tokens["total_output_tokens"]) == (350, 35)
    assert tokens["total_tokens"] == 385

    users = {
        entry["user_id"]: entry for entry in report(admin, "users", group_id=group_id)["users"]
    }
    assert (users[first.id]["count"], users[first.id]["input_tokens"]) == (2, 300)
    assert (users[first.id]["output_tokens"], users[first.id]["total_tokens"]) == (30, 330)
    # a reply without usage is a message but adds no tokens
    assert (users[second.id]["count"], users[second.id]["input_tokens"]) == (2, 50)
    assert (users[second.id]["output_tokens"], users[second.id]["total_tokens"]) == (5, 55)


# ---------------------------------------------------------------- daily series


def test_daily_series_puts_each_reply_on_its_day_and_fills_the_empty_days(
    admin, accounts, make_preset
):
    owner = accounts()
    aster, birch = make_preset(), make_preset()
    group_id = make_group(admin, [owner])
    seed_reply(owner, aster, "today")
    seed_reply(owner, aster, "two days", days_ago=2)
    seed_reply(owner, birch, "two days b", days_ago=2)
    seed_reply(owner, birch, "five days", days_ago=5)

    series = report(admin, "daily", granularity="daily", group_id=group_id, **window(6))["data"]

    assert [entry["date"] for entry in series] == [day_of(n) for n in range(6, -1, -1)]
    assert {entry["date"]: entry["models"] for entry in series} == {
        day_of(6): {},
        day_of(5): {birch: 1},
        day_of(4): {},
        day_of(3): {},
        day_of(2): {aster: 1, birch: 1},
        day_of(1): {},
        day_of(0): {aster: 1},
    }


def test_a_window_after_the_oldest_reply_leaves_it_out_of_every_report(
    admin, accounts, make_preset
):
    owner = accounts()
    model_id = make_preset()
    group_id = make_group(admin, [owner])
    seed_reply(owner, model_id, "today")
    seed_reply(owner, model_id, "five days", days_ago=5)
    narrow = {"group_id": group_id, **window(3.5)}

    def counted(**params) -> list:
        series = report(admin, "daily", **params)["data"]
        return [
            sum(entry["models"].get(model_id, 0) for entry in series),
            report(admin, "summary", **params)["total_messages"],
            {m["model_id"]: m["count"] for m in report(admin, "models", **params)["models"]}[
                model_id
            ],
            report(admin, "users", **params)["users"][0]["count"],
        ]

    assert counted(group_id=group_id) == [2, 2, 2, 2]
    assert counted(**narrow) == [1, 1, 1, 1]


# ---------------------------------------------------------------- messages


def test_messages_are_filtered_by_model_user_and_chat(admin, accounts, make_preset):
    owner, other = accounts(), accounts()
    aster, birch = make_preset(), make_preset()
    new_chat = seed_reply(owner, aster, "fresh aster")
    seed_reply(owner, aster, "old aster", days_ago=3)
    seed_reply(owner, birch, "birch")
    seed_reply(other, aster, "other aster")

    def contents(**params) -> set[str]:
        return {message["content"] for message in report(admin, "messages", **params)}

    assert contents(model_id=aster, **window(1)) == {
        "asking fresh aster",
        "fresh aster",
        "asking other aster",
        "other aster",
    }
    assert contents(model_id=aster) >= {"old aster", "fresh aster", "other aster"}
    assert "birch" not in contents(model_id=aster)
    assert contents(user_id=other.id) == {"asking other aster", "other aster"}
    assert contents(chat_id=new_chat) == {"asking fresh aster", "fresh aster"}
    assert report(admin, "messages") == []


# ---------------------------------------------------------------- model drill-down


@pytest.fixture
def drilldown(admin, accounts, make_preset, instance) -> dict:
    """One chat per user on a preset: Bob's the newest and on today, Ann's in the middle."""
    suffix = uuid.uuid4().hex[:6]
    names = {key: f"{key}-{suffix}" for key in ("Ann", "Bob", "Cy")}
    people = {key: accounts(name=name) for key, name in names.items()}
    model_id = make_preset()
    chats = {
        "Ann": seed_reply(people["Ann"], model_id, "ann chat", days_ago=3),
        "Bob": seed_reply(people["Bob"], model_id, "bob chat", days_ago=0),
        "Cy": seed_reply(people["Cy"], model_id, "cy chat", days_ago=6),
    }
    now = int(time.time())
    # updated_at order (oldest first): Cy, Ann, Bob
    for key, age in (("Cy", 300), ("Ann", 200), ("Bob", 100)):
        write_rows(
            instance,
            "UPDATE chat SET updated_at = :stamp WHERE id = :id",
            [{"stamp": now - age, "id": chats[key]}],
        )
    return {"model_id": model_id, "chats": chats, "names": names}


def listed(admin: Actor, model_id: str, **params) -> dict:
    return report(admin, f"models/{model_id}/chats", **params)


def test_model_chats_only_list_the_chats_inside_the_date_window(admin, drilldown):
    model_id, chats = drilldown["model_id"], drilldown["chats"]

    recent = listed(admin, model_id, **window(4))
    assert [chat["chat_id"] for chat in recent["chats"]] == [chats["Bob"], chats["Ann"]]
    assert recent["total"] == 2

    older = listed(
        admin, model_id, start_date=window(8)["start_date"], end_date=window(2)["start_date"]
    )
    assert {chat["chat_id"] for chat in older["chats"]} == {chats["Ann"], chats["Cy"]}
    assert older["total"] == 2
    assert listed(admin, model_id)["total"] == 3


def test_model_chats_are_ordered_by_updated_at_and_by_user_name(admin, drilldown):
    model_id, chats, names = drilldown["model_id"], drilldown["chats"], drilldown["names"]

    def order(**params) -> list[str]:
        return [chat["chat_id"] for chat in listed(admin, model_id, **params)["chats"]]

    newest_first = [chats["Bob"], chats["Ann"], chats["Cy"]]
    assert order() == newest_first
    assert order(order_by="updated_at", direction="desc") == newest_first
    assert order(order_by="updated_at", direction="asc") == newest_first[::-1]
    assert order(order_by="user_name", direction="asc") == [chats["Ann"], chats["Bob"], chats["Cy"]]
    assert order(order_by="user_name", direction="desc") == [
        chats["Cy"],
        chats["Bob"],
        chats["Ann"],
    ]
    by_name = listed(admin, model_id, order_by="user_name", direction="asc")["chats"]
    assert [chat["user_name"] for chat in by_name] == [names["Ann"], names["Bob"], names["Cy"]]


def test_model_chats_page_with_skip_and_limit_and_report_the_total(admin, drilldown):
    model_id, chats = drilldown["model_id"], drilldown["chats"]

    first = listed(admin, model_id, skip=0, limit=2)
    second = listed(admin, model_id, skip=2, limit=2)

    assert [chat["chat_id"] for chat in first["chats"]] == [chats["Bob"], chats["Ann"]]
    assert [chat["chat_id"] for chat in second["chats"]] == [chats["Cy"]]
    assert first["total"] == second["total"] == 3


# ---------------------------------------------------------------- model overview


def rate(client, model_id: str, rating: int) -> None:
    response = client.post(
        "/api/v1/evaluations/feedback",
        json={"type": "rating", "data": {"rating": rating, "model_id": model_id}},
    )
    assert response.status_code == 200, response.text


def test_overview_counts_the_tags_of_the_chats_that_used_the_model(admin, accounts, make_preset):
    owner = accounts()
    aster, birch = make_preset(), make_preset()
    common, rare = f"common{uuid.uuid4().hex[:6]}", f"rare{uuid.uuid4().hex[:6]}"
    first = seed_reply(owner, aster, "first")
    second = seed_reply(owner, aster, "second")
    elsewhere = seed_reply(owner, birch, "elsewhere")
    with owner.client() as client:
        for chat_id, name in (
            (first, common),
            (first, rare),
            (second, common),
            (elsewhere, common),
        ):
            tagged = client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": name})
            assert tagged.status_code == 200, tagged.text

    tags = {
        entry["tag"]: entry["count"] for entry in report(admin, f"models/{aster}/overview")["tags"]
    }

    assert tags == {common: 2, rare: 1}
    other = report(admin, f"models/{birch}/overview")["tags"]
    assert other == [{"tag": common, "count": 1}]


@pytest.mark.parametrize("days", [30, 0])
def test_overview_history_counts_won_and_lost_ratings_per_day(admin, accounts, make_preset, days):
    owner = accounts()
    rated, untouched = make_preset(), make_preset()
    with owner.client() as client:
        rate(client, rated, 1)
        rate(client, rated, 1)
        rate(client, rated, -1)
        rate(client, untouched, -1)

    history = report(admin, f"models/{rated}/overview", days=days)["history"]

    today = {"date": day_of(0), "won": 2, "lost": 1}
    assert today in history
    assert sum(entry["won"] for entry in history) == 2
    assert sum(entry["lost"] for entry in history) == 1
    if days == 0:
        assert history == [today]
    else:
        assert len(history) in (30, 31)
        assert history[0]["won"] == history[0]["lost"] == 0


# ---------------------------------------------------------------- refusal


@pytest.mark.parametrize(
    "path",
    [
        "summary",
        "models",
        "users",
        "messages",
        "daily",
        "tokens",
        "models/some-model/chats",
        "models/some-model/overview",
    ],
)
def test_a_plain_user_is_refused_every_analytics_route(accounts, path):
    user = accounts()
    with user.client() as client:
        response = client.get(f"/api/v1/analytics/{path}", params={"model_id": "some-model"})
    assert response.status_code in (401, 403), f"{path} answered {response.status_code}"
