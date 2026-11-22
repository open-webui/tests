"""Stored fat chats, so a test can meet a long conversation instead of only ever a short one.

Each data set under `data/fat_chats/` is one conversation of hundreds of messages, in the shapes
Open WebUI stores: edits, regenerations and side-by-side replies of several models, native tool
calls with large results, reasoning, citations, attached files, code and tables, a stretch
answered by a terminal's coding agent with its signed sessions, and the sub-agent chats it
delegated to. `FAT_CHATS` names them, smallest first; `scripts/generate_fat_chats.py` made them.

`import_fat_chat(instance, owner, name)` writes one into the instance's own database, SQLite or
Postgres, the way the app stores a chat: the chat record with its whole history and one
`chat_message` row per message. Every import gets new chat ids, so a data set can be imported
any number of times; `owner` owns it and is its author, and `terminal_id` and `cwd` point the
agent's sessions at a real terminal (they are signed again for the new ids either way). It
returns a `FatChat` with the new ids and the data set's manifest: message counts and the ids of
messages worth a test's attention (an edited turn, a regenerated one, the largest reply, the
agent's latest reply, the term only one early message holds). `delete_fat_chat` removes it with
its sub-agent chats through the API, as the sidebar's Delete does. `fat_chat` in
`harness/fixtures.py` does both around a test.
"""

from __future__ import annotations

import functools
import json
import lzma
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import jwt

from harness.actors import Actor
from harness.inflight import wait_until_no_reply_runs
from harness.instance import WEBUI_SECRET_KEY, LaunchedInstance

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fat_chats"
FORMAT = 1
FAT_CHATS = ("fat-1mb", "fat-10mb", "fat-50mb", "fat-130mb")

# Stand-ins in the stored data, replaced on import
OWNER_ID = "f17e0000-0000-4000-8000-00000000a11c"
OWNER_NAME = "Fixture Owner"
TERMINAL_ID = "fixture-terminal"
TERMINAL_CWD = "/home/user/workspace"

# chat_message columns whose name differs from the message's key
COLUMN_KEYS = {
    "parent_id": "parentId",
    "model_id": "model",
    "status_history": "statusHistory",
    "context_summary": "contextSummary",
}
MESSAGE_COLUMNS = (
    "role",
    "parent_id",
    "content",
    "output",
    "model_id",
    "files",
    "sources",
    "embeds",
    "meta",
    "status_history",
    "error",
    "usage",
    "context_summary",
)


@dataclass
class FatChat:
    name: str
    chat_id: str
    subagent_chat_ids: list[str]
    manifest: dict
    seconds: float  # how long the import took


@functools.lru_cache(maxsize=2)
def _stored(name: str) -> str:
    if name not in FAT_CHATS:
        raise ValueError(f"no fat chat named {name!r}; there are {', '.join(FAT_CHATS)}")
    return lzma.decompress((DATA_DIR / f"{name}.json.xz").read_bytes()).decode()


def stored_manifest(name: str) -> dict:
    """The data set's manifest as stored, with the stand-in ids."""
    text = _stored(name)
    manifest, _ = json.JSONDecoder().raw_decode(text, text.index('"manifest":') + 11)
    return manifest


def import_fat_chat(
    instance: LaunchedInstance,
    owner: Actor,
    name: str,
    terminal_id: str | None = None,
    cwd: str | None = None,
) -> FatChat:
    started = time.monotonic()
    manifest = stored_manifest(name)
    replacements = {
        old: str(uuid.uuid4()) for old in [manifest["main_chat_id"], *manifest["subagent_chat_ids"]]
    }
    replacements.update({OWNER_ID: owner.id, OWNER_NAME: owner.name})
    replacements.update(
        {TERMINAL_ID: terminal_id or TERMINAL_ID, TERMINAL_CWD: cwd or TERMINAL_CWD}
    )

    def replaced(text: str) -> str:
        for old, new in replacements.items():
            text = text.replace(old, json.dumps(new)[1:-1])
        return text

    data = json.loads(replaced(_stored(name)))
    now = int(time.time())
    chat_rows, message_rows = [], []
    for chat in data["chats"]:
        messages = chat["chat"]["history"]["messages"]
        for message in messages.values():
            _sign_agent_session(message, replaced)
        chat_rows.append(
            {
                **chat,
                "user_id": owner.id,
                "updated_at": now,
                "last_read_at": now,
                "archived": False,
                "pinned": False,
                "variables": {},
            }
        )
        message_rows += [
            _message_row(chat["id"], owner.id, message, now) for message in messages.values()
        ]
    _insert(instance, chat_rows, message_rows)
    manifest = data["manifest"]
    return FatChat(
        name=name,
        chat_id=manifest["main_chat_id"],
        subagent_chat_ids=manifest["subagent_chat_ids"],
        manifest=manifest,
        seconds=time.monotonic() - started,
    )


def delete_fat_chat(owner: Actor, fat: FatChat) -> None:
    with owner.client() as client:
        # A reply marked done can still be saving; deleting then cancels it mid-write
        wait_until_no_reply_runs(client, fat.chat_id, timeout=60)
        deleted = client.delete(f"/api/v1/chats/{fat.chat_id}")
    assert deleted.status_code == 200 and deleted.json() is True, (
        f"deleting fat chat {fat.name} failed: {deleted.text}"
    )


def _sign_agent_session(message: dict, replaced) -> None:
    """Sign the agent's session again, for the chat, owner and terminal it now belongs to."""
    meta = message.get("meta") or {}
    if "agent" not in meta:
        return
    session = jwt.decode(meta["agent"], options={"verify_signature": False})
    session = json.loads(replaced(json.dumps(session)))
    meta["agent"] = jwt.encode(session, WEBUI_SECRET_KEY, algorithm="HS256")


def _message_row(chat_id: str, owner_id: str, message: dict, now: int) -> dict:
    """The `chat_message` row the app writes for a message of the chat's history."""
    row = {column: message.get(COLUMN_KEYS.get(column, column)) for column in MESSAGE_COLUMNS}
    return {
        **row,
        "id": f"{chat_id}-{message['id']}",
        "chat_id": chat_id,
        "user_id": message.get("user_id") or owner_id,
        "done": message.get("done", True),
        "usage": _row_usage(message.get("usage")),
        "created_at": message["timestamp"],
        "updated_at": now,
    }


def _row_usage(usage: dict | None) -> dict | None:
    """The usage as the app keeps it in the row, with the total it counts up when missing."""
    if not usage:
        return None
    if "total_tokens" in usage or not {"input_tokens", "output_tokens"} <= set(usage):
        return usage
    return {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]}


def _insert(instance: LaunchedInstance, chat_rows: list[dict], message_rows: list[dict]) -> None:
    import sqlalchemy

    options = {"timeout": 30} if instance.database_url.startswith("sqlite") else {}
    engine = sqlalchemy.create_engine(instance.database_url, connect_args=options)
    try:
        tables = sqlalchemy.MetaData()
        tables.reflect(engine, only=["chat", "chat_message"])
        with engine.begin() as connection:
            connection.execute(tables.tables["chat"].insert(), chat_rows)
            connection.execute(tables.tables["chat_message"].insert(), message_rows)
    finally:
        engine.dispose()
