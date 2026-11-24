"""Journey: chats an older version left in an odd shape keep every message through the upgrade.

The migration that moves chat messages into rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) runs at the first start on dev (open-webui/open-webui#34985). Older
installs hold chats saved as plain text instead of a record, and chats whose saved message list is
null or a list, while their messages are in the message table. What a user expects: the server
starts, every one of these chats opens with all its messages, and stepping back down to the
revision before the migration gives the old version every chat with all its messages again. Each
data set under `upgrade_data/` is taken to the revision before the migration with the manual
`alembic` command, three chats with a conversation are rewritten into those shapes, and the
checkout then starts on it.

Discriminates: on dev 9d19883f1 the start fails on the plain text chat on every data set (36
errors); with that chat left out, the null and list shaped chats lose messages on all six data
sets (24 failures). With fixes for both on top of it, all 36 pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import pytest
import sqlalchemy

from harness.prepared_data import ReleaseData, manual_alembic, run_sql
from harness.upgraded_release import data_set_params, upgraded_release

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

REVISION = "5e7a2c9d4b13"
BEFORE = f"{REVISION}-1"
SHAPES = ("plain text", "null message list", "message list as a list")


def _alembic(release: ReleaseData, *arguments: str) -> None:
    finished = manual_alembic(release.data_dir, *arguments, database_url=release.database_url)
    assert finished.returncode == 0, (
        f"alembic {' '.join(arguments)} failed:\n{finished.stderr[-3000:]}"
    )


def _stored_chats(database_url: str) -> dict[str, dict]:
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(sqlalchemy.text("SELECT id, user_id, chat FROM chat")).all()
    finally:
        engine.dispose()
    chats = {}
    for chat_id, user_id, chat in rows:
        chat = json.loads(chat) if isinstance(chat, str) else chat
        chats[chat_id] = {"user_id": user_id, "chat": chat}
    return chats


def _shown(messages: dict) -> dict[str, tuple]:
    return {
        message_id: (message.get("role"), message.get("content"), message.get("parentId"))
        for message_id, message in messages.items()
    }


@dataclass
class Steps:
    before: dict[str, dict]  # shape -> chat id and the messages it held
    opened: dict[str, dict]  # shape -> the messages its owner sees after the upgrade
    stepped_down: dict[str, dict]  # shape -> the messages the old version's record holds


def _rewrite(release: ReleaseData, before: dict[str, dict]) -> None:
    """Three chats with a conversation rewritten into the shapes older versions left behind."""
    before["release"] = release
    _alembic(release, "upgrade", BEFORE)
    accounts = {account["id"]: who for who, account in release.manifest["accounts"].items()}
    owned = [
        (chat_id, accounts[stored["user_id"]], stored["chat"])
        for chat_id, stored in sorted(_stored_chats(release.database_url).items())
        if stored["user_id"] in accounts
        and len(((stored["chat"] or {}).get("history") or {}).get("messages") or {}) >= 2
    ]
    assert len(owned) >= len(SHAPES), f"the data set holds too few conversations: {len(owned)}"
    as_json = "CAST(:chat AS json)" if release.manifest["engine"] == "postgres" else ":chat"
    for shape, (chat_id, owner, chat) in zip(SHAPES, owned):
        before["shapes"][shape] = {
            "id": chat_id,
            "owner": owner,
            "messages": _shown(chat["history"]["messages"]),
        }
        messages = chat["history"]["messages"]
        if shape == "plain text":
            stored = json.dumps(json.dumps(chat))
        else:
            odd = {
                "null message list": None,
                "message list as a list": list(messages.values()),
            }[shape]
            stored = json.dumps({**chat, "history": {**chat["history"], "messages": odd}})
        run_sql(
            release.database_url,
            f"UPDATE chat SET chat = {as_json} WHERE id = :id",
            {"id": chat_id, "chat": stored},
        )


@pytest.fixture(scope="module", params=data_set_params())
def steps(request, tmp_path_factory) -> Iterator[Steps]:
    name = request.param
    before: dict = {"shapes": {}}
    with upgraded_release(
        name,
        tmp_path_factory.mktemp(name),
        prepare=lambda release: _rewrite(release, before),
    ) as release:
        opened = {}
        for shape, old in before["shapes"].items():
            response = release.get(old["owner"], f"/api/v1/chats/{old['id']}")
            opened[shape] = (
                _shown(response.json()["chat"]["history"]["messages"])
                if response.status_code == 200
                else {"status": response.status_code}
            )
        _alembic(before["release"], "downgrade", BEFORE)
        stored = _stored_chats(before["release"].database_url)
        stepped_down = {}
        for shape, old in before["shapes"].items():
            chat = stored[old["id"]]["chat"]
            history = chat.get("history") if isinstance(chat, dict) else None
            messages = (history or {}).get("messages")
            stepped_down[shape] = _shown(messages) if isinstance(messages, dict) else messages
        yield Steps(before["shapes"], opened, stepped_down)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_chat_opens_with_every_message_after_the_upgrade(steps, shape):
    assert steps.opened[shape] == steps.before[shape]["messages"], (
        f"the chat stored as {shape} opens without its messages: {steps.opened[shape]}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_old_version_gets_every_message_back_after_a_downgrade(steps, shape):
    assert steps.stepped_down[shape] == steps.before[shape]["messages"], (
        f"the chat stored as {shape} lost messages in the downgrade: {steps.stepped_down[shape]}"
    )
