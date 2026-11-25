"""Journey: the admin's analytics read the same after the upgrade that moves messages into rows.

The migration that keeps chat messages only as rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) rewrites every chat of an older install at the first start on dev.
Admin analytics count each assistant reply with its model, its owner and the tokens it used. What
an admin expects: after the upgrade the dashboard shows every reply of every chat under its model
and its owner, with the tokens its chat shows for it, and the totals, the per model and per user
lists and the token chart add up to what the older version showed. Each data set under
`upgrade_data/` is taken to the revision before the migration with the manual `alembic` command
and its replies get token usage stored as the server recorded it, in the chat and on the reply's
row. The checkout then starts on it, and the expected dashboard is computed from the chats as
their owners saw them. On the v0.11.4 SQLite set with the same usage, v0.11.4 itself shows these
numbers.

Discriminates: passes on dev f80a4f782 on all six data sets; with the migration dropping the
usage of replies that already had a row in a copy of it, both tests fail on every set (tokens 0).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import pytest
import sqlalchemy

from harness.prepared_data import ReleaseData, manual_alembic, run_sql
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

REVISION = "5e7a2c9d4b13"
BEFORE = f"{REVISION}-1"


@dataclass
class Reply:
    chat_id: str
    owner_id: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: int


def _alembic(release: ReleaseData, *arguments: str) -> None:
    finished = manual_alembic(release.data_dir, *arguments, database_url=release.database_url)
    assert finished.returncode == 0, (
        f"alembic {' '.join(arguments)} failed:\n{finished.stderr[-3000:]}"
    )


def _query(database_url: str, statement: str) -> list:
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.connect() as connection:
            return connection.execute(sqlalchemy.text(statement)).all()
    finally:
        engine.dispose()


def _stored_chats(database_url: str) -> dict[str, tuple[str, dict]]:
    chats = {}
    for chat_id, user_id, chat in _query(database_url, "SELECT id, user_id, chat FROM chat"):
        chats[chat_id] = (user_id, json.loads(chat) if isinstance(chat, str) else chat)
    return chats


def _history(chat: dict | None) -> dict:
    messages = ((chat or {}).get("history") or {}).get("messages")
    return messages if isinstance(messages, dict) else {}


def _replies(chats: dict[str, tuple[str, dict]]) -> list[Reply]:
    """Every assistant reply of every chat, with the tokens its chat shows for it."""
    replies = []
    for chat_id, (owner_id, chat) in chats.items():
        for message in _history(chat).values():
            if message.get("role") == "assistant":
                usage = message.get("usage") or {}
                replies.append(
                    Reply(
                        chat_id,
                        owner_id,
                        message["model"],
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        int(message["timestamp"]),
                    )
                )
    return replies


def _record_usage(release: ReleaseData) -> list[Reply]:
    """Token usage on every reply, in the chat and on its row, as the server saved it."""
    as_json = "CAST(:value AS json)" if release.manifest["engine"] == "postgres" else ":value"
    row_ids = {row_id for (row_id,) in _query(release.database_url, "SELECT id FROM chat_message")}
    counter = 0
    for chat_id, (_, chat) in sorted(_stored_chats(release.database_url).items()):
        messages = _history(chat)
        replies = sorted(
            (message_id, message)
            for message_id, message in messages.items()
            if message.get("role") == "assistant"
        )
        if not replies:
            continue
        for message_id, message in replies:
            counter += 1
            message["usage"] = {
                "prompt_tokens": 100 + 7 * counter,
                "completion_tokens": 20 + 3 * counter,
                "total_tokens": 120 + 10 * counter,
            }
            row_id = f"{chat_id}-{message_id}"
            assert row_id in row_ids, f"the data set's reply {row_id} has no row"
            run_sql(
                release.database_url,
                f"UPDATE chat_message SET usage = {as_json} WHERE id = :id",
                {"id": row_id, "value": json.dumps(message["usage"])},
            )
        run_sql(
            release.database_url,
            f"UPDATE chat SET chat = {as_json} WHERE id = :id",
            {"id": chat_id, "value": json.dumps(chat)},
        )
    assert counter >= 4, f"the data set holds too few replies: {counter}"
    return _replies(_stored_chats(release.database_url))


@dataclass
class Dashboard:
    upgraded: Upgraded
    replies: list[Reply]

    def read(self, path: str, **params) -> dict:
        response = self.upgraded.get("admin", f"/api/v1/analytics/{path}", params=params)
        assert response.status_code == 200, f"the dashboard's {path} failed: {response.text}"
        return response.json()


@pytest.fixture(scope="module", params=data_set_params())
def dashboard(request, tmp_path_factory) -> Iterator[Dashboard]:
    name = request.param
    replies: list[Reply] = []

    def prepare(release: ReleaseData) -> None:
        _alembic(release, "upgrade", BEFORE)
        replies.extend(_record_usage(release))

    with upgraded_release(name, tmp_path_factory.mktemp(name), prepare=prepare) as upgraded:
        yield Dashboard(upgraded, replies)


def _per_model(replies: list[Reply]) -> dict[str, dict]:
    models = defaultdict(
        lambda: {"count": 0, "input": 0, "output": 0, "users": set(), "chats": set()}
    )
    for reply in replies:
        entry = models[reply.model]
        entry["count"] += 1
        entry["input"] += reply.input_tokens
        entry["output"] += reply.output_tokens
        entry["users"].add(reply.owner_id)
        entry["chats"].add(reply.chat_id)
    return models


def _per_user(replies: list[Reply]) -> dict[str, tuple[int, int, int]]:
    users = defaultdict(lambda: [0, 0, 0])
    for reply in replies:
        users[reply.owner_id][0] += 1
        users[reply.owner_id][1] += reply.input_tokens
        users[reply.owner_id][2] += reply.output_tokens
    return {user_id: tuple(totals) for user_id, totals in users.items()}


def test_each_model_keeps_its_replies_and_tokens(dashboard):
    expected = _per_model(dashboard.replies)
    assert expected, "the data set holds no replies"
    tokens = {
        entry["model_id"]: (entry["message_count"], entry["input_tokens"], entry["output_tokens"])
        for entry in dashboard.read("tokens")["models"]
    }
    models = {
        entry["model_id"]: (entry["count"], entry["unique_users"], entry["unique_chats"])
        for entry in dashboard.read("models")["models"]
    }
    assert tokens == {
        model: (entry["count"], entry["input"], entry["output"])
        for model, entry in expected.items()
    }, f"replies and input and output tokens per model read {tokens}"
    assert models == {
        model: (entry["count"], len(entry["users"]), len(entry["chats"]))
        for model, entry in expected.items()
    }, f"replies, users and chats per model read {models}"

    first = min(reply.timestamp for reply in dashboard.replies) - 86400
    chart = dashboard.read(
        "daily", metric="tokens", start_date=first, end_date=int(time.time()) + 86400
    )
    charted = defaultdict(int)
    for bucket in chart["data"]:
        for model, value in bucket["models"].items():
            charted[model] += value
    assert dict(charted) == {
        model: entry["input"] + entry["output"] for model, entry in expected.items()
    }, f"the token chart adds up to {dict(charted)}"


def test_the_totals_and_each_user_keep_their_replies_and_tokens(dashboard):
    replies = dashboard.replies
    summary = dashboard.read("summary")
    shown = (
        summary["total_messages"],
        summary["total_chats"],
        summary["total_models"],
        summary["total_users"],
    )
    assert shown == (
        len(replies),
        len({reply.chat_id for reply in replies}),
        len({reply.model for reply in replies}),
        len({reply.owner_id for reply in replies}),
    ), f"replies, chats, models and users in the summary read {shown}"

    tokens = dashboard.read("tokens")
    assert (tokens["total_input_tokens"], tokens["total_output_tokens"]) == (
        sum(reply.input_tokens for reply in replies),
        sum(reply.output_tokens for reply in replies),
    ), f"total tokens read {tokens['total_input_tokens']} in, {tokens['total_output_tokens']} out"

    users = dashboard.read("users", metric="tokens")
    per_user = {
        entry["user_id"]: (entry["count"], entry["input_tokens"], entry["output_tokens"])
        for entry in users["users"]
    }
    assert per_user == _per_user(replies), f"replies and tokens per user read {per_user}"
    assert users["total"] == len(per_user)
