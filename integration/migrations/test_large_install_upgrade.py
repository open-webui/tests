"""Journey: a big install upgrades to message rows in bounded time and memory and keeps every chat.

The migration that moves chat messages into rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) runs at the first start after the upgrade and reads every chat of
the install. On a real install that means thousands of chats and some very long ones. What an
admin expects: the first start finishes within the few minutes the manual migration guide says
to wait before assuming a migration is stuck, fits the memory a small server or a container
limit gives Open WebUI, and afterwards every chat opens with every message as it was saved, is
found by search and can be continued.

The v0.13.4 data sets (the newest release before the change, SQLite and Postgres) are taken to
the revision before the migration with the manual `alembic` command. The 130 MB stored chat
(`harness/fat_chats.py`) is added as that version saved it, its whole history in the chat record
and a row per message, with its two sub-agent chats, and so are 2,000 ordinary chats spread over
the accounts. The checkout then starts on it, which runs the migration, and starts once more
without it, as the comparison for time and memory.

The bounds: the first start may take at most `STARTUP_LIMIT` (3 minutes), the low end of the 3
to 5 minutes the guide asks admins to wait before assuming a migration hangs; a slower start
makes them restart a working upgrade. Its process may use at most `EXTRA_MEMORY_LIMIT` (1 GiB)
more at its peak than the start after it: a normal start takes about 0.9 GiB, and the
performance guide covers containers limited to 2 GB, where a bigger upgrade is killed midway.

Discriminates: passes on dev f80a4f782 on SQLite and Postgres (the first start took 12 to 14 s
against 9 s for the start after it, and peaked 70 to 180 MiB above it). Mutations in a copy of
the migration: saving each row without its `data` fails both opening tests and the continued
chat on both databases (side-by-side replies lose their models); a 0.1 s pause per chat fails
the start time (218 s); holding 16 copies of each chat record fails the memory test (1793 MiB
more).
"""

from __future__ import annotations

import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest
import sqlalchemy

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.fat_chats import FatChat, import_fat_chat
from harness.prepared_data import ReleaseData, RunningBackend, manual_alembic, serving
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

BEFORE = "5e7a2c9d4b13-1"
DATA_SETS = [param for param in data_set_params() if param.id.startswith("v0.13.4-")]
FAT = "fat-130mb"
FAT_OWNER = "alice"
ORDINARY_CHATS = 2000
STARTUP_LIMIT = 180.0  # seconds
EXTRA_MEMORY_LIMIT = 1024**3  # bytes
# the release ran with it, so every account could chat with the provider's model
RELEASE_ENVIRONMENT = {"BYPASS_MODEL_ACCESS_CONTROL": "true"}
BOOT_SETTINGS = {"WEBUI_AUTH": "true", "RAG_EMBEDDING_ENGINE": "openai", **RELEASE_ENVIRONMENT}


@dataclass
class Start:
    seconds: float
    peak_bytes: int | None  # None where the process size cannot be read


@dataclass
class Before:
    fat: FatChat | None = None
    # chat id -> owner and the messages the old version saved
    chats: dict[str, dict] = field(default_factory=dict)
    words: dict[str, str] = field(default_factory=dict)  # ordinary chat id -> its unique word
    upgrade: Start | None = None


@dataclass
class Upgrade:
    before: Before
    restart: Start
    opened: dict[str, dict | str]  # chat id -> the messages its owner sees, or why it did not open
    found: dict[str, list[str]]  # search text -> the chats the owner finds
    continued: dict | str  # the big chat after one more message, or why it failed
    turn_ids: tuple[str, str, str]  # the message it was on, the new question and its answer


def _peak_bytes(process_id: int) -> int | None:
    status = Path(f"/proc/{process_id}/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _measured(backend: RunningBackend) -> Start:
    return Start(backend.seconds_to_start, _peak_bytes(backend.process_id))


def _account(release: ReleaseData, who: str) -> Actor:
    account = release.manifest["accounts"][who]
    return Actor(
        id=account["id"],
        name=account["name"],
        email=account["email"],
        password=account["password"],
        role=account["role"],
        token="",
        base_url="",
    )


def _ordinary_chat(number: int, owner_id: str, sent: int) -> tuple[dict, list[dict], str]:
    """A chat of two turns, its first answer regenerated once, as v0.13.4 saved it."""
    word = f"wombat{uuid.uuid4().hex[:10]}"
    names = ("u1", "a1", "a1b", "u2", "a2")
    parents = {"u1": None, "a1": "u1", "a1b": "u1", "u2": "a1b", "a2": "u2"}
    ids = {name: str(uuid.uuid4()) for name in names}
    paragraph = " ".join(["The trip plan covers trains, rooms and the weather."] * 6)
    messages = {}
    for offset, name in enumerate(names):
        message = {
            "id": ids[name],
            "parentId": ids[parents[name]] if parents[name] else None,
            "childrenIds": [ids[child] for child in names if parents[child] == name],
            "role": "user" if name.startswith("u") else "assistant",
            "content": f"{name} of chat {number}, about the {word}. {paragraph}",
            "timestamp": sent + offset,
        }
        if name.startswith("u"):
            message["models"] = [MOCK_MODEL_ID]
        else:
            message |= {
                "model": MOCK_MODEL_ID,
                "modelName": "Mock Model",
                "modelIdx": 0,
                "done": True,
                "usage": {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100},
            }
        messages[ids[name]] = message
    chat_id = str(uuid.uuid4())
    title = f"Trip plan {number}"
    chat = {
        "id": chat_id,
        "user_id": owner_id,
        "title": title,
        "chat": {
            "title": title,
            "models": [MOCK_MODEL_ID],
            "history": {"messages": messages, "currentId": ids["a2"]},
        },
        "created_at": sent,
        "updated_at": sent + len(names),
        "archived": False,
        "pinned": False,
        "meta": {},
        "current_message_id": ids["a2"],
    }
    rows = [
        {
            "id": f"{chat_id}-{message['id']}",
            "chat_id": chat_id,
            "user_id": owner_id,
            "role": message["role"],
            "parent_id": message["parentId"],
            "content": message["content"],
            "model_id": message.get("model"),
            "done": True,
            "usage": message.get("usage"),
            "created_at": message["timestamp"],
            "updated_at": message["timestamp"],
        }
        for message in messages.values()
    ]
    return chat, rows, word


def _add_ordinary_chats(release: ReleaseData, before: Before) -> None:
    owners = [who for who in sorted(release.manifest["accounts"])]
    chats, rows = [], []
    for number in range(ORDINARY_CHATS):
        owner = owners[number % len(owners)]
        owner_id = release.manifest["accounts"][owner]["id"]
        chat, chat_rows, word = _ordinary_chat(number, owner_id, 1_700_000_000 + number * 10)
        chats.append(chat)
        rows.extend(chat_rows)
        before.chats[chat["id"]] = {"owner": owner, "messages": chat["chat"]["history"]["messages"]}
        before.words[chat["id"]] = word
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.begin() as connection:
            chat_table = sqlalchemy.Table("chat", sqlalchemy.MetaData(), autoload_with=connection)
            message_table = sqlalchemy.Table(
                "chat_message", sqlalchemy.MetaData(), autoload_with=connection
            )
            connection.execute(chat_table.insert(), chats)
            connection.execute(message_table.insert(), rows)
    finally:
        engine.dispose()


def _saved_messages(release: ReleaseData, chat_ids: list[str]) -> dict[str, dict]:
    """The messages each chat's record holds, as the old version saved them."""
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.text("SELECT id, chat FROM chat WHERE id IN :ids").bindparams(
                    sqlalchemy.bindparam("ids", expanding=True)
                ),
                {"ids": chat_ids},
            ).all()
    finally:
        engine.dispose()
    saved = {}
    for chat_id, chat in rows:
        chat = json.loads(chat) if isinstance(chat, str) else chat
        saved[chat_id] = chat["history"]["messages"]
    return saved


def _prepare(release: ReleaseData, before: Before) -> None:
    finished = manual_alembic(
        release.data_dir, "upgrade", BEFORE, database_url=release.database_url
    )
    assert finished.returncode == 0, f"alembic upgrade {BEFORE} failed:\n{finished.stderr[-3000:]}"
    before.fat = import_fat_chat(release, _account(release, FAT_OWNER), FAT, before_rows=True)
    fat_ids = [before.fat.chat_id, *before.fat.subagent_chat_ids]
    for chat_id, messages in _saved_messages(release, fat_ids).items():
        before.chats[chat_id] = {"owner": FAT_OWNER, "messages": messages}
    _add_ordinary_chats(release, before)
    settings = {**BOOT_SETTINGS, **release.settings}
    with serving(release.data_dir, settings, timeout=STARTUP_LIMIT * 2) as upgrading:
        before.upgrade = _measured(upgrading)


def _differences(saved: dict, opened: dict) -> list[str]:
    """What the owner sees differently from what was saved, one line per message."""
    found = [f"{message_id} is missing" for message_id in saved.keys() - opened.keys()]
    found += [f"{message_id} was never saved" for message_id in opened.keys() - saved.keys()]
    for message_id in saved.keys() & opened.keys():
        changed = sorted(
            key for key, value in saved[message_id].items() if opened[message_id].get(key) != value
        )
        if changed:
            found.append(f"{message_id} reads differently in {', '.join(changed)}")
    return found


def _open(upgraded: Upgraded, chat_id: str, owner: str, saved: dict) -> dict | str:
    response = upgraded.get(owner, f"/api/v1/chats/{chat_id}")
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    history = response.json()["chat"]["history"]
    return {
        "differences": _differences(saved, history["messages"])[:5],
        "current": history.get("currentId"),
    }


def _search(upgraded: Upgraded, owner: str, text: str) -> list[str]:
    response = upgraded.get(owner, "/api/v1/chats/search", params={"text": text})
    assert response.status_code == 200, f"searching for {text!r} failed: {response.text}"
    return [chat["id"] for chat in response.json()]


def _continue(upgraded: Upgraded, fat: FatChat, saved: dict) -> tuple[dict | str, tuple]:
    """The big chat after one more message on the branch it was left on."""
    question = f"Where do we stand now? {uuid.uuid4().hex[:6]}"
    upgraded.provider.queue(reply.text("Everything is on track.", match=reply.answering(question)))
    with upgraded.client(FAT_OWNER) as client:
        current_id = client.get(f"/api/v1/chats/{fat.chat_id}").json()["chat"]["history"][
            "currentId"
        ]
        turn, answer = ask(client, question, chat_id=fat.chat_id, parent_id=current_id)
        ids = (current_id, turn.user_message_id, turn.assistant_message_id)
        if answer.get("content") != "Everything is on track.":
            return f"the reply reads {answer.get('content')!r}", ids
        response = client.get(f"/api/v1/chats/{fat.chat_id}")
    if response.status_code != 200:
        return f"HTTP {response.status_code} on reopening: {response.text[:300]}", ids
    messages = response.json()["chat"]["history"]["messages"]
    new = {message_id: messages.get(message_id, {}) for message_id in ids[1:]}
    earlier = {key: value for key, value in messages.items() if key not in new}
    # the message it was on gains the new question as a child, and nothing else changes
    expected = {
        **saved,
        current_id: {
            **saved[current_id],
            "childrenIds": [*saved[current_id].get("childrenIds", []), ids[1]],
        },
    }
    return {
        "differences": _differences(expected, earlier)[:5],
        "question": new[ids[1]],
        "answer": new[ids[2]],
        "children_of_current": messages.get(current_id, {}).get("childrenIds", []),
    }, ids


@pytest.fixture(scope="module", params=DATA_SETS)
def upgrade(request, tmp_path_factory) -> Iterator[Upgrade]:
    name = request.param
    before = Before()
    with upgraded_release(
        name,
        tmp_path_factory.mktemp(name),
        settings=RELEASE_ENVIRONMENT,
        prepare=lambda release: _prepare(release, before),
    ) as upgraded:
        restart = _measured(upgraded.backend)
        for owner in {chat["owner"] for chat in before.chats.values()}:
            upgraded.client(owner).close()  # signs in once, before the threads share the tokens
        with ThreadPoolExecutor(max_workers=8) as pool:
            opened = dict(
                zip(
                    before.chats,
                    pool.map(
                        lambda chat_id: _open(
                            upgraded,
                            chat_id,
                            before.chats[chat_id]["owner"],
                            before.chats[chat_id]["messages"],
                        ),
                        before.chats,
                    ),
                )
            )
        ordinary_id = next(iter(before.words))
        ordinary_owner = before.chats[ordinary_id]["owner"]
        found = {
            before.fat.manifest["search_term"]: _search(
                upgraded, FAT_OWNER, before.fat.manifest["search_term"]
            ),
            before.words[ordinary_id]: _search(upgraded, ordinary_owner, before.words[ordinary_id]),
        }
        continued, turn_ids = _continue(
            upgraded, before.fat, before.chats[before.fat.chat_id]["messages"]
        )
        yield Upgrade(before, restart, opened, found, continued, turn_ids)


def test_the_first_start_after_the_upgrade_finishes_within_minutes(upgrade):
    upgrading = upgrade.before.upgrade
    assert upgrading.seconds <= STARTUP_LIMIT, (
        f"the first start after the upgrade took {upgrading.seconds:.0f}s with the 130 MB chat "
        f"and {ORDINARY_CHATS} more; the start after it took {upgrade.restart.seconds:.0f}s"
    )


def test_the_upgrade_fits_in_the_memory_of_a_small_server(upgrade):
    upgrading, restart = upgrade.before.upgrade, upgrade.restart
    if upgrading.peak_bytes is None or restart.peak_bytes is None:
        pytest.skip(f"the process size is read from /proc, which {sys.platform} lacks")
    extra = upgrading.peak_bytes - restart.peak_bytes
    assert extra <= EXTRA_MEMORY_LIMIT, (
        f"the first start after the upgrade peaked at {upgrading.peak_bytes / 2**20:.0f} MiB, "
        f"{extra / 2**20:.0f} MiB more than the start after it "
        f"({restart.peak_bytes / 2**20:.0f} MiB)"
    )


def test_the_big_chat_opens_with_every_message_as_saved(upgrade):
    fat = upgrade.before.fat
    for chat_id in [fat.chat_id, *fat.subagent_chat_ids]:
        opened = upgrade.opened[chat_id]
        assert isinstance(opened, dict), f"the chat {chat_id} does not open: {opened}"
        assert opened["differences"] == [], (
            f"the chat {chat_id} opens with other messages than were saved: {opened['differences']}"
        )
    assert upgrade.opened[fat.chat_id]["current"] == fat.manifest["last_reply"], (
        f"the big chat opens on {upgrade.opened[fat.chat_id]['current']}, not on its last reply"
    )


def test_every_ordinary_chat_opens_with_every_message_as_saved(upgrade):
    words = upgrade.before.words
    assert len(words) == ORDINARY_CHATS
    broken = {
        chat_id: upgrade.opened[chat_id]
        for chat_id in words
        if not isinstance(upgrade.opened[chat_id], dict) or upgrade.opened[chat_id]["differences"]
    }
    assert not broken, (
        f"{len(broken)} of {ORDINARY_CHATS} chats do not open as saved, for example "
        f"{next(iter(broken))}: {next(iter(broken.values()))}"
    )


def test_search_finds_the_big_chat_and_an_ordinary_one(upgrade):
    fat = upgrade.before.fat
    term = fat.manifest["search_term"]
    assert upgrade.found[term] == [fat.chat_id], (
        f"searching for a word of one early message of the big chat found {upgrade.found[term]}"
    )
    ordinary_id, word = next(iter(upgrade.before.words.items()))
    assert upgrade.found[word] == [ordinary_id], (
        f"searching for the word only one ordinary chat holds found {upgrade.found[word]}"
    )


def test_the_big_chat_takes_one_more_message_and_keeps_the_rest(upgrade):
    continued = upgrade.continued
    assert isinstance(continued, dict), f"the big chat cannot be continued: {continued}"
    current_id, question_id, answer_id = upgrade.turn_ids
    assert continued["differences"] == [], (
        f"one more message changed earlier messages of the big chat: {continued['differences']}"
    )
    assert continued["question"].get("parentId") == current_id, (
        f"the new question does not follow the message the chat was on: {continued['question']}"
    )
    assert question_id in continued["children_of_current"], (
        "the message the chat was on does not lead to the new question"
    )
    assert continued["answer"].get("parentId") == question_id, (
        f"the answer does not follow the new question: {continued['answer']}"
    )
