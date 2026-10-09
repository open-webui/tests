"""Journey: a new install's database goes through its whole life on SQLite and on Postgres.

Every start runs `alembic upgrade head`, and since v0.11.3 a failure there stops the boot, so a
migration chain that breaks on an empty database is a dead install. Regressions this guards:

* #29280 (8c0c7b3b6, v0.11.3): `migrations/env.py` imports the calendar model, whose import
  chain reached back into a still-loading config module; the upgrade raised before a single
  table existed.
* 38d63c18f30f (Postgres, #24560): recreated the user primary key on a fresh database; DDL is
  transactional there, so the whole chain rolled back and the boot died on `relation "config"
  does not exist`.
* b10670c03dd5 (SQLite): dropped the index backing a UNIQUE constraint, which SQLite refuses,
  so the chain stopped partway and later tables were missing.

The server starts on an empty database, the first account becomes the admin and saves
something in every feature area, the server restarts on that database (a second upgrade, as
every container restart runs) and it all reads back, while accounts are still added, renamed
and deleted. The skill it saved starts its history with one version, which the restart keeps.
The operator's manual commands from the migration guide then run on an empty database of their
own: upgrade to head, the key tables it lists, a second upgrade, a step back and forward again,
and `alembic downgrade base` unwinding every table. Postgres runs on the
embedded server (`pgserver`) in either database mode of the suite.

Twin of unit/migrations/test_lifecycle.py and of unit/deps/test_alembic.py.
Discriminates: passes on dev ef67cc3fa on both engines; a module-scope
`from open_webui.config import ENABLE_SIGNUP` in a copy's `models/calendar.py` (the #29280
cycle) fails every test on both; a `downgrade()` of `56359461a091` (calendar tables) that no
longer drops `calendar_event_attendee` fails the downgrade test on both; a column added to
the note model with no migration behind it fails the install on SQLite (saving a note answers
400); a batch alteration of `4ace53fd72c8` told not to recreate its table (`recreate='never'`)
fails the install and the manual commands on SQLite, as does an `env.py` that never calls
`run_migrations()`. A copy whose skill save writes no history entry (9bbb95048) fails the skill
test on both.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import sqlalchemy

from harness import backends
from harness.calendar_api import HOUR_NS, create_event, default_calendar_id
from harness.prepared_data import RunningBackend, manual_alembic, serving

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

PASSWORD = "lifecycle-admin-password-123"
# Any one of these missing means the chain stopped before the migration creating it.
CRITICAL_TABLES = {
    "auth",
    "calendar",
    "chat",
    "config",
    "function",
    "group",
    "knowledge",
    "knowledge_directory",
    "knowledge_file",
    "memory",
    "note",
    "oauth_session",
    "prompt",
    "skill",
    "skill_history",
    "tag",
    "tool",
    "user",
}
# Well below the ~60 tables of dev, so new tables never make it churn.
MIN_TABLE_COUNT = 30


@dataclass
class Database:
    engine: str
    url: str | None  # None on SQLite: the server and alembic find `webui.db` in the data dir
    data_dir: Path

    @property
    def settings(self) -> dict[str, str]:
        database = {"DATABASE_URL": self.url} if self.url else {}
        return {"ENABLE_CHANNELS": "true", **database}

    def alembic(self, *arguments: str) -> str:
        finished = manual_alembic(self.data_dir, *arguments, database_url=self.url)
        assert finished.returncode == 0, (
            f"alembic {' '.join(arguments)} failed on {self.engine}:\n{finished.stderr[-3000:]}"
        )
        return finished.stdout

    def tables(self) -> set[str]:
        url = self.url or f"sqlite:///{self.data_dir / 'webui.db'}"
        engine = sqlalchemy.create_engine(url)
        try:
            return set(sqlalchemy.inspect(engine).get_table_names())
        finally:
            engine.dispose()


@contextlib.contextmanager
def _empty_database(engine: str, data_dir: Path) -> Iterator[Database]:
    data_dir.mkdir(parents=True)
    if engine == "sqlite":
        yield Database(engine, None, data_dir)
        return
    with backends.postgres_database() as url:
        yield Database(engine, url, data_dir)


ENGINES = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param("postgres", id="postgres", marks=pytest.mark.requires_postgres),
]


@dataclass
class Install:
    database: Database
    admin: dict
    made: dict[str, str] = field(default_factory=dict)


def _sign_in(server: RunningBackend, email: str, password: str) -> str:
    with server.client() as client:
        signed_in = client.post("/api/v1/auths/signin", json={"email": email, "password": password})
    assert signed_in.status_code == 200, f"HTTP {signed_in.status_code}: {signed_in.text}"
    return signed_in.json()["token"]


def _created(response: httpx.Response, what: str) -> dict:
    assert response.status_code == 200, f"saving {what} failed: {response.text}"
    return response.json()


def _save_one_of_everything(client: httpx.Client) -> dict[str, str]:
    """One record in each feature area, the way its page saves it; returns their ids."""
    tag = uuid.uuid4().hex[:8]
    chat = _created(
        client.post("/api/v1/chats/new", json={"chat": {"title": f"Trip {tag}", "messages": []}}),
        "a chat",
    )
    _created(client.post(f"/api/v1/chats/{chat['id']}/tags", json={"name": "travel"}), "a tag")
    folder = _created(client.post("/api/v1/folders/", json={"name": f"Plans {tag}"}), "a folder")
    moved = client.post(f"/api/v1/chats/{chat['id']}/folder", json={"folder_id": folder["id"]})
    _created(moved, "the chat's folder")
    note = client.post(
        "/api/v1/notes/create",
        json={"title": f"Packing {tag}", "data": {"content": {"md": "passport"}}},
    )
    memory = client.post("/api/v1/memories/add", json={"content": f"likes trains {tag}"})
    prompt = client.post(
        "/api/v1/prompts/create",
        json={"command": f"trip-{tag}", "name": "Trip", "content": "Plan a trip"},
    )
    knowledge = client.post(
        "/api/v1/knowledge/create", json={"name": f"Guides {tag}", "description": ""}
    )
    group = client.post(
        "/api/v1/groups/create", json={"name": f"Travellers {tag}", "description": ""}
    )
    channel = client.post("/api/v1/channels/create", json={"name": f"trips-{tag}"})
    skill = client.post(
        "/api/v1/skills/create",
        json={"id": f"packing-{tag}", "name": f"Packing {tag}", "content": "Pack light."},
    )
    start = time.time_ns()
    event = create_event(
        client, default_calendar_id(client), start_at=start, end_at=start + HOUR_NS
    )
    return {
        "chat": chat["id"],
        "folder": folder["id"],
        "note": _created(note, "a note")["id"],
        "memory": _created(memory, "a memory")["id"],
        "prompt": _created(prompt, "a prompt")["id"],
        "knowledge": _created(knowledge, "a knowledge base")["id"],
        "group": _created(group, "a group")["id"],
        "channel": _created(channel, "a channel")["id"],
        "event": _created(event, "a calendar event")["id"],
        "skill": _created(skill, "a skill")["id"],
    }


@pytest.fixture(scope="module", params=ENGINES)
def installed(request, tmp_path_factory) -> Iterator[Install]:
    """A new install that started once, took its first admin and saved one of everything."""
    root = tmp_path_factory.mktemp(f"install-{request.param}")
    with _empty_database(request.param, root / "data") as database:
        with serving(database.data_dir, database.settings) as server, server.client() as client:
            signed_up = client.post(
                "/api/v1/auths/signup",
                json={"name": "First", "email": "first@example.com", "password": PASSWORD},
            )
            assert signed_up.status_code == 200, f"the first sign-up failed: {signed_up.text}"
            admin = signed_up.json()
            client.headers["Authorization"] = f"Bearer {admin['token']}"
            made = _save_one_of_everything(client)
        yield Install(database, admin, made)


@pytest.fixture(scope="module")
def restarted(installed) -> Iterator[httpx.Client]:
    """The admin's client on the same install after a restart."""
    database = installed.database
    with serving(database.data_dir, database.settings) as server:
        token = _sign_in(server, installed.admin["email"], PASSWORD)
        with server.client(token) as client:
            yield client


def test_the_first_account_on_a_new_install_is_the_admin(installed):
    assert installed.admin["role"] == "admin", (
        f"the first account on a new {installed.database.engine} install got "
        f"{installed.admin['role']}"
    )


def test_a_restart_keeps_what_every_feature_saved(installed, restarted):
    made = installed.made
    reads = {
        "chat": f"/api/v1/chats/{made['chat']}",
        "note": f"/api/v1/notes/{made['note']}",
        "prompt": f"/api/v1/prompts/id/{made['prompt']}",
        "knowledge": f"/api/v1/knowledge/{made['knowledge']}",
        "group": f"/api/v1/groups/id/{made['group']}",
        "channel": f"/api/v1/channels/{made['channel']}",
        "event": f"/api/v1/calendars/events/{made['event']}",
    }
    statuses = {what: restarted.get(path).status_code for what, path in reads.items()}
    missing = {what: status for what, status in statuses.items() if status != 200}
    assert not missing, f"records gone after a restart on {installed.database.engine}: {missing}"

    chat = restarted.get(f"/api/v1/chats/{made['chat']}").json()
    assert chat["folder_id"] == made["folder"]
    tags = restarted.get(f"/api/v1/chats/{made['chat']}/tags").json()
    assert "travel" in {tag["name"] for tag in tags}
    memories = restarted.get("/api/v1/memories/").json()
    assert made["memory"] in {memory["id"] for memory in memories}


def test_the_first_skill_save_starts_its_history_and_a_restart_keeps_it(installed, restarted):
    skill_id = installed.made["skill"]
    skill = restarted.get(f"/api/v1/skills/id/{skill_id}").json()
    history = restarted.get(f"/api/v1/skills/id/{skill_id}/history").json()
    assert [(entry["id"], entry["parent_id"]) for entry in history] == [
        (skill["version_id"], None)
    ], f"the new skill's history on {installed.database.engine}: {history}"
    assert history[0]["user_id"] == installed.admin["id"]
    files = restarted.get(f"/api/v1/skills/id/{skill_id}/files").json()
    assert [file["path"] for file in files["files"]] == ["SKILL.md"]


def test_accounts_are_added_renamed_and_deleted_after_a_restart(restarted):
    email = f"life-{uuid.uuid4().hex[:8]}@example.com"
    added = restarted.post(
        "/api/v1/auths/add",
        json={"name": "Original", "email": email, "password": "life-password-1", "role": "user"},
    )
    assert added.status_code == 200, added.text
    account_id = added.json()["id"]

    renamed = restarted.post(
        f"/api/v1/users/{account_id}/update",
        json={"name": "Renamed", "email": email, "role": "user", "profile_image_url": ""},
    )
    assert renamed.status_code == 200, renamed.text
    assert restarted.get(f"/api/v1/users/{account_id}").json()["name"] == "Renamed"

    deleted = restarted.delete(f"/api/v1/users/{account_id}")
    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    assert restarted.get(f"/api/v1/users/{account_id}").status_code in (400, 404)


@pytest.fixture(params=ENGINES)
def empty_database(request, tmp_path) -> Iterator[Database]:
    with _empty_database(request.param, tmp_path / "data") as database:
        yield database


def test_the_manual_migration_commands_run_clean(empty_database):
    database = empty_database
    database.alembic("upgrade", "head")
    head = database.alembic("heads").split()[0]
    assert head in database.alembic("current"), "the database is not at head after the upgrade"

    tables = database.tables()
    assert not CRITICAL_TABLES - tables, f"missing after upgrade: {CRITICAL_TABLES - tables}"
    assert len(tables) >= MIN_TABLE_COUNT, f"only {len(tables)} tables; the chain stopped early"

    database.alembic("upgrade", "head")
    database.alembic("downgrade", "-1")
    assert head not in database.alembic("current"), "downgrade -1 did not step back"
    database.alembic("upgrade", "head")
    assert head in database.alembic("current")


def test_downgrade_to_base_unwinds_every_table(empty_database):
    database = empty_database
    database.alembic("upgrade", "head")

    database.alembic("downgrade", "base")

    leftover = database.tables() - {"alembic_version"}
    assert not leftover, f"a migration's downgrade() leaves tables behind: {sorted(leftover)}"
