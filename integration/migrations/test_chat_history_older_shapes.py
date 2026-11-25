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

Eight more chats are added there in shapes older versions left behind: a broken history next to
a plain list whose first entry has no id (#35004), an answer whose parent is gone, no current
message, NaN in a reply's usage (SQLite only), a plain list with no history, a reply without
content, content as a list of parts and timestamps in milliseconds. Each must open on the branch it
was left on with every message at its time, and take one more message on that branch without
losing any. A plain list repeating a message id is not covered: no version or importer wrote one,
and v0.13.4 already showed such a chat collapsed.

Discriminates: on dev 9d19883f1 the start fails on the plain text chat on every data set (36
errors); with that chat left out, the null and list shaped chats lose messages on all six data
sets (24 failures). With fixes for both on top of it, all 36 pass. Legacy shapes: all pass on dev
f80a4f782; a backend copy without the #35012 fix fails the list whose first entry has no id.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import Iterator

import httpx
import pytest
import sqlalchemy

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.prepared_data import ReleaseData, manual_alembic, run_sql
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from harness.upstream import MOCK_MODEL_ID

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
    legacy: dict[str, "Legacy"]  # legacy shape -> the chat as the old version saved it
    reopened: dict[str, tuple]  # legacy shape -> the chat opened, then after one more message


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


# Chats an older version saved in a broken or very old shape, added before the upgrade.
LEGACY_SHAPES = (
    "first message without an id",
    "orphaned parent",
    "missing current message",
    "NaN in usage",
    "messages without a history",
    "content as None",
    "content as a list",
    "millisecond timestamps",
)
LEGACY_OWNER = "alice"
# the releases ran with it, so every account could chat with the provider's model
RELEASE_ENVIRONMENT = {"BYPASS_MODEL_ACCESS_CONTROL": "true"}
FOLLOW_UP = ("Anything else for the trip?", "Bring a power adapter.")
TEXTS = {
    "u1": "What should I pack for Vienna?",
    "a1": "Pack an umbrella.",
    "a1b": "Pack sunglasses, it is sunny.",
    "u2": "And for the evening?",
    "a2": "A light jacket.",
}
PARENTS = {"u1": None, "a1": "u1", "a1b": "u1", "u2": "a1b", "a2": "u2"}
SHOWN = ("u1", "a1b", "u2", "a2")  # the branch the chat was left on


@dataclass
class Legacy:
    chat_id: str
    blob: dict  # the chat record as the old version saved it
    rows: list[dict]  # the messages the old version also saved as rows
    every_message: list[tuple[str, str]]  # role and text of each message, sorted
    thread: list[tuple[str, str]]  # what the chat shows, first to last
    answers_to_first: list[str]  # the answers the first question offers, in order
    sent_at: dict[str, int]  # text -> second the message was sent


def _role(name: str) -> str:
    return "user" if name.startswith("u") else "assistant"


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _conversation(ids: dict[str, str], sent: int) -> dict[str, dict]:
    """Two turns, the first answer regenerated once, left on the second answer."""
    messages = {}
    for offset, name in enumerate(PARENTS):
        message = {
            "id": ids[name],
            "parentId": ids[PARENTS[name]] if PARENTS[name] else None,
            "childrenIds": [ids[child] for child, parent in PARENTS.items() if parent == name],
            "role": _role(name),
            "content": TEXTS[name],
            "timestamp": sent + offset,
        }
        if _role(name) == "assistant":
            message |= {"model": MOCK_MODEL_ID, "done": True}
        else:
            message["models"] = [MOCK_MODEL_ID]
        messages[ids[name]] = message
    return messages


def _legacy_chat(shape: str, chat_id: str, sent: int) -> Legacy:
    ids = {name: str(uuid.uuid4()) for name in PARENTS}
    messages = _conversation(ids, sent)
    history = {"messages": messages, "currentId": ids["a2"]}
    blob = {"title": f"Kept as {shape}", "models": [MOCK_MODEL_ID], "history": history}
    thread = [(_role(name), TEXTS[name]) for name in SHOWN]
    answers_to_first = [TEXTS["a1"], TEXTS["a1b"]]
    unlinked = [{**messages[ids[name]]} for name in SHOWN]
    for message in unlinked:
        del message["parentId"], message["childrenIds"]
    if shape == "orphaned parent":
        del messages[ids["a1b"]]
        messages[ids["u1"]]["childrenIds"] = [ids["a1"]]
        thread = [(_role(name), TEXTS[name]) for name in ("u2", "a2")]
        answers_to_first = [TEXTS["a2"]]
    elif shape == "missing current message":
        del history["currentId"]
    elif shape == "NaN in usage":
        messages[ids["a2"]]["usage"] = {
            "prompt_tokens": math.nan,
            "completion_tokens": 4,
            "total_tokens": math.nan,
        }
    elif shape == "content as None":
        messages[ids["a1"]]["content"] = None
        answers_to_first = ["", TEXTS["a1b"]]
    elif shape == "content as a list":
        messages[ids["u2"]]["content"] = [{"type": "text", "text": TEXTS["u2"]}]
    elif shape == "millisecond timestamps":
        for message in messages.values():
            message["timestamp"] *= 1000
    elif shape == "first message without an id":
        # #35004: a broken history next to an older plain list, the list's first entry without an id
        del unlinked[0]["id"]
        blob["history"] = {"messages": None, "currentId": ids["a2"]}
        blob["messages"] = unlinked
        answers_to_first = [TEXTS["a1b"]]
    elif shape == "messages without a history":
        del blob["history"]
        blob["messages"] = unlinked
        answers_to_first = [TEXTS["a1b"]]
    if shape == "first message without an id":
        rows = [messages[ids[name]] for name in SHOWN]
    elif "history" in blob:
        rows = list(messages.values())
    else:
        rows = []
    listed = blob["messages"] if "messages" in blob else list(messages.values())
    every_message = sorted((message["role"], _text(message["content"])) for message in listed)
    sent_at = {_text(message["content"]): _as_seconds(message["timestamp"]) for message in listed}
    return Legacy(chat_id, blob, rows, every_message, thread, answers_to_first, sent_at)


def _as_seconds(timestamp) -> float:
    return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp


def _add_legacy_chats(release: ReleaseData, before: dict[str, dict]) -> None:
    """One chat per legacy shape, saved with its rows as the old version left it."""
    postgres = release.manifest["engine"] == "postgres"
    as_json = "CAST(:{} AS json)" if postgres else ":{}"
    owner_id = release.manifest["accounts"][LEGACY_OWNER]["id"]
    sent = 1_700_000_000
    for shape in LEGACY_SHAPES:
        if postgres and shape == "NaN in usage":
            continue
        legacy = _legacy_chat(shape, str(uuid.uuid4()), sent)
        before["legacy"][shape] = legacy
        run_sql(
            release.database_url,
            "INSERT INTO chat (id, user_id, title, chat, created_at, updated_at, archived, pinned,"
            f" meta) VALUES (:id, :user_id, :title, {as_json.format('chat')}, :sent, :sent,"
            f" false, false, {as_json.format('meta')})",
            {
                "id": legacy.chat_id,
                "user_id": owner_id,
                "title": legacy.blob["title"],
                "chat": json.dumps(legacy.blob),
                "sent": sent,
                "meta": "{}",
            },
        )
        for message in legacy.rows:
            run_sql(
                release.database_url,
                "INSERT INTO chat_message (id, chat_id, user_id, role, parent_id, content,"
                " model_id, done, usage, created_at, updated_at) VALUES (:id, :chat_id, :user_id,"
                f" :role, :parent_id, {as_json.format('content')}, :model_id, :done,"
                f" {as_json.format('usage')}, :created_at, :created_at)",
                {
                    "id": f"{legacy.chat_id}-{message['id']}",
                    "chat_id": legacy.chat_id,
                    "user_id": owner_id,
                    "role": message["role"],
                    "parent_id": message.get("parentId"),
                    "content": json.dumps(message["content"]),
                    "model_id": message.get("model"),
                    "done": message.get("done", True),
                    "usage": json.dumps(message["usage"]) if "usage" in message else None,
                    "created_at": int(_as_seconds(message["timestamp"])),
                },
            )
        sent += 100


def _refuse_constant(name: str):
    raise ValueError(f"{name} is not JSON a browser reads")


def _open(client: httpx.Client, chat_id: str) -> dict | str:
    """The chat's history as the web client reads it, or why it does not open."""
    response = client.get(f"/api/v1/chats/{chat_id}")
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    try:
        return json.loads(response.text, parse_constant=_refuse_constant)["chat"]["history"]
    except ValueError as error:
        return f"the answer is not valid JSON: {error}"


def _thread(history: dict) -> list[dict]:
    """The messages the chat shows, first to last: from the current one up its parents."""
    messages = history["messages"]
    shown, seen_ids, message_id = [], set(), history.get("currentId")
    while message_id in messages and message_id not in seen_ids:
        seen_ids.add(message_id)
        shown.append(messages[message_id])
        message_id = messages[message_id].get("parentId")
    return shown[::-1]


def _read(messages: list[dict]) -> list[tuple[str, str]]:
    return [(message.get("role"), _text(message.get("content"))) for message in messages]


def _open_and_continue(release: Upgraded, chat_id: str) -> tuple[dict | str, dict | str]:
    """The chat as its owner opens it, then as it reads after one more message on its branch."""
    with release.client(LEGACY_OWNER) as client:
        opened = _open(client, chat_id)
        if isinstance(opened, str):
            return opened, opened
        question, answer = FOLLOW_UP
        release.provider.queue(reply.text(answer, match=reply.answering(question)))
        earlier = [{"role": role, "content": text} for role, text in _read(_thread(opened))]
        try:
            parent_id = opened.get("currentId")
            turn = send_message(
                client, question, chat_id=chat_id, parent_id=parent_id, history=earlier
            )
            wait_for_reply(client, turn)
        except (AssertionError, httpx.HTTPError) as error:
            return opened, f"sending one more message failed: {error}"
        return opened, _open(client, chat_id)


@pytest.fixture(scope="module", params=data_set_params())
def steps(request, tmp_path_factory) -> Iterator[Steps]:
    name = request.param
    before: dict = {"shapes": {}, "legacy": {}}

    def prepare(release: ReleaseData) -> None:
        _rewrite(release, before)
        _add_legacy_chats(release, before)

    with upgraded_release(
        name, tmp_path_factory.mktemp(name), settings=RELEASE_ENVIRONMENT, prepare=prepare
    ) as release:
        opened = {}
        for shape, old in before["shapes"].items():
            response = release.get(old["owner"], f"/api/v1/chats/{old['id']}")
            opened[shape] = (
                _shown(response.json()["chat"]["history"]["messages"])
                if response.status_code == 200
                else {"status": response.status_code}
            )
        reopened = {
            shape: _open_and_continue(release, legacy.chat_id)
            for shape, legacy in before["legacy"].items()
        }
        _alembic(before["release"], "downgrade", BEFORE)
        stored = _stored_chats(before["release"].database_url)
        stepped_down = {}
        for shape, old in before["shapes"].items():
            chat = stored[old["id"]]["chat"]
            history = chat.get("history") if isinstance(chat, dict) else None
            messages = (history or {}).get("messages")
            stepped_down[shape] = _shown(messages) if isinstance(messages, dict) else messages
        yield Steps(before["shapes"], opened, stepped_down, before["legacy"], reopened)


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


def _legacy(steps: Steps, shape: str) -> tuple[Legacy, tuple]:
    if shape not in steps.legacy:
        pytest.skip("Postgres refuses NaN in a json value, so no version could store it there")
    return steps.legacy[shape], steps.reopened[shape]


@pytest.mark.parametrize("shape", LEGACY_SHAPES)
def test_a_legacy_chat_opens_on_its_branch_with_every_message(steps, shape):
    legacy, (opened, _) = _legacy(steps, shape)
    assert isinstance(opened, dict), f"the chat with {shape} does not open: {opened}"
    messages = list(opened["messages"].values())
    assert sorted(_read(messages)) == legacy.every_message, (
        f"the chat with {shape} opens with other messages: {sorted(_read(messages))}"
    )
    thread = _thread(opened)
    assert _read(thread) == legacy.thread, (
        f"the chat with {shape} opens on another branch: {_read(thread)}"
    )
    first = thread[0]
    answers = [
        _text(opened["messages"][child_id]["content"]) for child_id in first.get("childrenIds", [])
    ]
    assert answers == legacy.answers_to_first, (
        f"the first question of the chat with {shape} offers the answers {answers}"
    )
    sent_at = {
        _text(message["content"]): _as_seconds(message.get("timestamp") or 0)
        for message in messages
    }
    assert sent_at == legacy.sent_at, f"the chat with {shape} moved its messages in time: {sent_at}"


@pytest.mark.parametrize("shape", LEGACY_SHAPES)
def test_a_legacy_chat_takes_one_more_message_on_its_branch(steps, shape):
    legacy, (opened, continued) = _legacy(steps, shape)
    assert isinstance(continued, dict), f"the chat with {shape} cannot be continued: {continued}"
    expected = [*legacy.thread, ("user", FOLLOW_UP[0]), ("assistant", FOLLOW_UP[1])]
    assert _read(_thread(continued)) == expected, (
        f"the chat with {shape} reads {_read(_thread(continued))} after one more message"
    )
    changed = {
        message_id: continued["messages"].get(message_id, {}).get("content")
        for message_id, message in opened["messages"].items()
        if _read([continued["messages"].get(message_id, {})]) != _read([message])
    }
    assert not changed, (
        f"the next send lost or changed messages of the chat with {shape}: {changed}"
    )
