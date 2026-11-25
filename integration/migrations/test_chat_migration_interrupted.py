"""Journey: an upgrade killed while it moves chat messages into rows finishes on the next start.

The migration that moves chat messages into rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) works through every chat of an install, which takes a while on a big
one, and a container can be killed in the middle (a crash, an out-of-memory kill, a deploy that
gives up waiting). What a user expects: the next start finishes the upgrade, and every chat opens
with exactly the messages it had, none lost and none twice, and can be continued. Each data set
under `upgrade_data/` is taken to the revision before the migration with the manual `alembic`
command and gets a few thousand more conversations, copies of one of alice's with ids of their
own. The checkout starts on it and is killed once the first chats are moved, then starts again.

Discriminates: passes on dev f80a4f782 for all eight data sets; with the migration adding its
`chat_message.data` column again without checking for it in a copy of it, the restart fails to
start (`duplicate column name: data`) and every test errors (checked on v0.11.4-sqlite).
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

import pytest
import sqlalchemy

from harness import upstream as reply
from harness.chat import ask
from harness.prepared_data import ReleaseData, killed_when, manual_alembic
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

BEFORE = "5e7a2c9d4b13-1"
COPIES = 3000
STILL_IN_THE_BLOB = sqlalchemy.text(
    "SELECT id FROM chat WHERE CAST(chat AS TEXT) LIKE '%\"history\"%'"
)


def _shown(messages: dict) -> dict[str, tuple]:
    return {
        message_id: (
            message.get("role"),
            message.get("content"),
            message.get("parentId"),
            message.get("childrenIds", []),
        )
        for message_id, message in messages.items()
    }


def _blob(chat) -> dict:
    chat = json.loads(chat) if isinstance(chat, str) else chat
    return chat if isinstance(chat, dict) else {}


def _blob_messages(chat: dict) -> dict:
    messages = (chat.get("history") or {}).get("messages")
    return messages if isinstance(messages, dict) else {}


@dataclass
class Killed:
    before: dict[str, dict] = field(default_factory=dict)  # chat id -> owner and its messages
    copies: set[str] = field(default_factory=set)
    moved_at_kill: set[str] = field(default_factory=set)  # chats moved before the kill
    release: ReleaseData | None = None


def _copy_with_own_ids(chat_row: dict, message_rows: list[dict], number: int) -> tuple:
    """The chat and its message rows under a new chat id and new message ids."""
    chat_id = str(uuid.uuid4())
    source_id = chat_row["id"]
    renamed = {row["id"].removeprefix(f"{source_id}-"): str(uuid.uuid4()) for row in message_rows}

    def renamed_ids(text: str) -> str:
        for old, new in renamed.items():
            text = text.replace(old, new)
        return text

    blob = json.loads(renamed_ids(json.dumps(_blob(chat_row["chat"]))))
    blob["title"] = f"{chat_row['title']} {number}"
    chat = {**chat_row, "id": chat_id, "title": blob["title"], "chat": blob, "share_id": None}
    rows = [
        {
            **row,
            "id": f"{chat_id}-{renamed[row['id'].removeprefix(f'{source_id}-')]}",
            "chat_id": chat_id,
            "parent_id": renamed.get(row["parent_id"], row["parent_id"]),
        }
        for row in message_rows
    ]
    return chat, rows


def _prepare(release: ReleaseData, killed: Killed) -> None:
    finished = manual_alembic(
        release.data_dir, "upgrade", BEFORE, database_url=release.database_url
    )
    assert finished.returncode == 0, f"alembic upgrade {BEFORE} failed:\n{finished.stderr[-3000:]}"
    killed.release = release
    accounts = {account["id"]: who for who, account in release.manifest["accounts"].items()}
    alice_id = release.manifest["accounts"]["alice"]["id"]
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.begin() as connection:
            chats = sqlalchemy.Table("chat", sqlalchemy.MetaData(), autoload_with=connection)
            message_table = sqlalchemy.Table(
                "chat_message", sqlalchemy.MetaData(), autoload_with=connection
            )
            stored = [dict(row) for row in connection.execute(sqlalchemy.select(chats)).mappings()]
            for row in stored:
                messages = _blob_messages(_blob(row["chat"]))
                if row["user_id"] in accounts and messages:
                    killed.before[row["id"]] = {
                        "owner": accounts[row["user_id"]],
                        "messages": _shown(messages),
                    }
            source = max(
                (row for row in stored if row["user_id"] == alice_id),
                key=lambda row: len(_blob_messages(_blob(row["chat"]))),
            )
            source_rows = [
                dict(row)
                for row in connection.execute(
                    sqlalchemy.select(message_table).where(message_table.c.chat_id == source["id"])
                ).mappings()
            ]
            assert len(source_rows) >= 2, f"alice's longest chat has {len(source_rows)} messages"
            new_chats, new_rows = [], []
            for number in range(COPIES):
                chat, rows = _copy_with_own_ids(source, source_rows, number)
                new_chats.append(chat)
                new_rows.extend(rows)
                killed.before[chat["id"]] = {
                    "owner": "alice",
                    "messages": _shown(_blob_messages(chat["chat"])),
                }
                killed.copies.add(chat["id"])
            connection.execute(chats.insert(), new_chats)
            connection.execute(message_table.insert(), new_rows)
    finally:
        engine.dispose()

    def still_in_blob() -> set[str]:
        engine = sqlalchemy.create_engine(release.database_url)
        try:
            with engine.connect() as connection:
                return set(connection.execute(STILL_IN_THE_BLOB).scalars())
        finally:
            engine.dispose()

    waiting = still_in_blob() & set(killed.before)
    assert waiting == set(killed.before), "chats lost their messages before the upgrade started"
    killed_when(
        release.data_dir,
        lambda: len(still_in_blob() & waiting) < len(waiting),
        settings=release.settings,
    )
    killed.moved_at_kill = waiting - still_in_blob()
    assert 0 < len(killed.moved_at_kill) < len(waiting) // 2, (
        f"the kill came after {len(killed.moved_at_kill)} of {len(waiting)} chats were moved; "
        "it has to land early in the migration"
    )


@dataclass
class Restarted:
    upgraded: Upgraded
    killed: Killed
    opened: dict[str, dict]  # chat id -> the messages its owner sees after the restart


def _open(upgraded: Upgraded, chat_id: str, owner: str) -> dict:
    response = upgraded.get(owner, f"/api/v1/chats/{chat_id}")
    if response.status_code != 200:
        return {"status": response.status_code}
    return _shown(response.json()["chat"]["history"]["messages"])


@pytest.fixture(scope="module", params=data_set_params())
def restarted(request, tmp_path_factory) -> Iterator[Restarted]:
    name = request.param
    killed = Killed()
    with upgraded_release(
        name,
        tmp_path_factory.mktemp(name),
        {"BYPASS_MODEL_ACCESS_CONTROL": "true"},  # the scripted model is shared with no one
        prepare=lambda release: _prepare(release, killed),
    ) as upgraded:
        for owner in {chat["owner"] for chat in killed.before.values()}:
            upgraded.client(owner).close()  # signs in once, before the threads share the tokens
        with ThreadPoolExecutor(max_workers=8) as pool:
            opened = dict(
                zip(
                    killed.before,
                    pool.map(
                        lambda chat_id: _open(upgraded, chat_id, killed.before[chat_id]["owner"]),
                        killed.before,
                    ),
                )
            )
        yield Restarted(upgraded, killed, opened)


def test_every_chat_opens_with_every_message_after_the_restart(restarted):
    before = restarted.killed.before
    changed = {
        chat_id: restarted.opened[chat_id]
        for chat_id in before
        if restarted.opened[chat_id] != before[chat_id]["messages"]
    }
    moved_early = changed.keys() & restarted.killed.moved_at_kill
    assert not changed, (
        f"{len(changed)} of {len(before)} chats open with other messages after the restart "
        f"({len(moved_early)} of them moved before the kill); "
        f"for example {next(iter(changed))}: {next(iter(changed.values()))}"
    )


def test_no_message_is_stored_twice(restarted):
    engine = sqlalchemy.create_engine(restarted.killed.release.database_url)
    try:
        with engine.connect() as connection:
            counts = dict(
                connection.execute(
                    sqlalchemy.text("SELECT chat_id, COUNT(*) FROM chat_message GROUP BY chat_id")
                ).all()
            )
    finally:
        engine.dispose()
    before = restarted.killed.before
    wrong = {
        chat_id: (counts.get(chat_id, 0), len(chat["messages"]))
        for chat_id, chat in before.items()
        if counts.get(chat_id, 0) != len(chat["messages"])
    }
    assert not wrong, (
        f"{len(wrong)} chats hold a different number of stored messages than they had "
        f"(stored, had): {dict(list(wrong.items())[:5])}"
    )


@pytest.mark.parametrize("when", ["moved before the kill", "moved after the restart"])
def test_a_chat_continues_where_it_left_off(restarted, when):
    killed, upgraded = restarted.killed, restarted.upgraded
    moved_early = when == "moved before the kill"
    chat_id = sorted(
        chat_id for chat_id in killed.copies if (chat_id in killed.moved_at_kill) == moved_early
    )[0]
    with upgraded.client("alice") as client:
        current_id = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["currentId"]
        prompt = f"And after that? {uuid.uuid4().hex[:6]}"
        upgraded.provider.queue(reply.text("Then we rest.", match=reply.answering(prompt)))
        turn, answer = ask(client, prompt, chat_id=chat_id, parent_id=current_id)
        assert answer["content"] == "Then we rest.", answer
        messages = _shown(
            client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        )

    earlier = killed.before[chat_id]["messages"]
    kept = {message_id: messages[message_id] for message_id in earlier if message_id in messages}
    assert kept.keys() == earlier.keys(), f"the continued chat lost {earlier.keys() - kept.keys()}"
    assert len(messages) == len(earlier) + 2, f"the continued chat holds {len(messages)} messages"
    assert messages[turn.user_message_id][2] == current_id, (
        "the new message does not follow the one the chat was on"
    )
    assert turn.user_message_id in messages[current_id][3]
