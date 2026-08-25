"""Regressions for three v0.11.1 repairs of data persisted in the wrong shape.

1. `user.oauth` double-encoded (PR #28107, commit bd8378f643, issue #28101).
   The 0.6.41 -> 0.9.6 user-table migration `b10670c03dd5` wrote the oauth
   mapping as `json.dumps({provider: {...}})` into a JSON column, so the column
   held a JSON *string* instead of an object. Every provider+sub lookup missed
   and those accounts could not sign in. The fix stores the object directly and
   adds migration `6d09d1bf1f23`, which rewrites already-double-encoded rows.

2. Default model settings (commit 5c05608e3a).
   `Config.repair_flattened_dict_configs()` became `Config.repair_config_rows()`
   and gained a pass over `ui.default_models` / `ui.default_pinned_models`:
   instances whose rows were stored as a list instead of the comma-separated
   string the app reads had their defaults silently ignored.

3. Connection tags saved as plain strings (commit 8be4c5fa6a, issue #28749).
   `main.get_models` did `[tag.get('name') for tag in meta['tags']]` inside a
   bare try/except, so a connection whose tags were stored as strings raised,
   the except blanked `model['tags']`, and the connection editor broke. The fix
   guards `info`/`meta` as dicts and runs the new `normalize_model_tags`.

Discriminates: passes on v0.11.1, fails on v0.11.0 (the oauth column stays a
JSON string, list-shaped default-model rows stay lists, and string tags come
back blanked or raise out of `get_models`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

# Head of the migration chain on v0.11.0, and the parent of the repair chain on
# v0.11.1. Seeding here reaches the same schema on both refs; upgrading to head
# afterwards is a no-op on v0.11.0 and runs 6d09d1bf1f23 on v0.11.1.
PRE_REPAIR_REVISION = "f0bd01a18a3d"

# The user-table migration that writes user.oauth, and the revision before it.
USER_TABLE_REVISION = "b10670c03dd5"
USER_TABLE_DOWN_REVISION = "2f1211949ecc"

_OAUTH_OBJECT = {"google": {"sub": "108154321"}}


def _run_python(
    backend: Path, db_url: str, data_dir: Path, body: str, *, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a Python snippet with `backend/` on sys.path and the DB env set."""
    preamble = textwrap.dedent(
        f"""
        import json, os, sys
        os.environ["DATABASE_URL"] = {db_url!r}
        os.environ["DATA_DIR"] = {str(data_dir)!r}
        os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key-for-unit-tests")
        sys.path.insert(0, {str(backend)!r})
        """
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


_ALEMBIC_PREAMBLE = """
import sqlalchemy as sa
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine
from open_webui.env import OPEN_WEBUI_DIR

cfg = AlembicConfig(OPEN_WEBUI_DIR / "alembic.ini")
cfg.set_main_option("script_location", str(OPEN_WEBUI_DIR / "migrations"))
"""

# Seed four oauth shapes through a JSON-typed column, then upgrade to head and
# read them back the same way. Only the double-encoded row may change.
_REPAIR_BODY = _ALEMBIC_PREAMBLE + f"""
command.upgrade(cfg, {PRE_REPAIR_REVISION!r})

user = sa.table(
    "user",
    sa.column("id", sa.Text),
    sa.column("name", sa.Text),
    sa.column("email", sa.Text),
    sa.column("oauth", sa.JSON),
)
seeds = {{
    "u_double": json.dumps({_OAUTH_OBJECT!r}),
    "u_object": {_OAUTH_OBJECT!r},
    "u_null": None,
    "u_plain": "not-json-at-all",
    "u_scalar": "12345",
}}

engine = create_engine(os.environ["DATABASE_URL"])
with engine.begin() as c:
    for uid, value in seeds.items():
        c.execute(sa.insert(user).values(id=uid, name=uid, email=uid + "@example.com", oauth=value))
engine.dispose()

command.upgrade(cfg, "head")

engine = create_engine(os.environ["DATABASE_URL"])
out = {{}}
with engine.connect() as c:
    for uid in seeds:
        v = c.execute(sa.select(user.c.oauth).where(user.c.id == uid)).scalar()
        out[uid] = {{"type": type(v).__name__, "value": v}}
engine.dispose()

print("RESULT:" + json.dumps(out))
"""

# Legacy `oauth_sub` rows converted by b10670c03dd5 itself.
_OAUTH_SUB_BODY = _ALEMBIC_PREAMBLE + f"""
command.upgrade(cfg, {USER_TABLE_DOWN_REVISION!r})

from sqlalchemy import text

engine = create_engine(os.environ["DATABASE_URL"])
with engine.begin() as c:
    c.execute(
        text("INSERT INTO user (id, name, email, oauth_sub) VALUES (:i, :n, :m, :s)"),
        {{"i": "u_sub", "n": "Dana", "m": "dana@example.com", "s": "google@108154321"}},
    )
engine.dispose()

command.upgrade(cfg, {USER_TABLE_REVISION!r})

user = sa.table("user", sa.column("id", sa.Text), sa.column("oauth", sa.JSON))
engine = create_engine(os.environ["DATABASE_URL"])
with engine.connect() as c:
    v = c.execute(sa.select(user.c.oauth).where(user.c.id == "u_sub")).scalar()
engine.dispose()

print("RESULT:" + json.dumps({{"type": type(v).__name__, "value": v}}))
"""


def _migrate_and_read(backend: Path, tmp_path: Path, body: str, what: str) -> dict:
    db_path = (tmp_path / "webui.db").resolve()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    result = _run_python(backend, f"sqlite:///{db_path.as_posix()}", data_dir, body)

    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("RESULT:")), None)
    if result.returncode != 0 or line is None:
        pytest.fail(
            f"{what} crashed or produced no result.\n"
            f"returncode={result.returncode}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{result.stdout[-1500:]}"
        )
    return json.loads(line.removeprefix("RESULT:"))


# --------------------------------------------------------------------------
# 21. Signing in after a long-delayed upgrade (user.oauth double-encoded).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repaired_oauth_rows(open_webui_backend: Path, tmp_path_factory) -> dict:
    """One migration run shared by the oauth-repair assertions."""
    tmp_path = tmp_path_factory.mktemp("oauth-repair")
    return _migrate_and_read(
        open_webui_backend, tmp_path, _REPAIR_BODY, "upgrade to head with seeded user.oauth rows"
    )


def test_double_encoded_oauth_is_repaired_to_an_object(repaired_oauth_rows: dict) -> None:
    """The whole point: a JSON-string oauth value becomes a real object again."""
    row = repaired_oauth_rows["u_double"]
    assert row["type"] == "dict", (
        f"user.oauth is still {row['type']} after upgrading to head; provider+sub "
        f"lookups keep missing and the account cannot sign in (#28101). "
        f"value={row['value']!r}"
    )
    assert row["value"] == _OAUTH_OBJECT, row["value"]


def test_already_correct_oauth_object_is_left_alone(repaired_oauth_rows: dict) -> None:
    row = repaired_oauth_rows["u_object"]
    assert row["type"] == "dict" and row["value"] == _OAUTH_OBJECT, row


def test_null_oauth_stays_null(repaired_oauth_rows: dict) -> None:
    assert repaired_oauth_rows["u_null"]["value"] is None, repaired_oauth_rows["u_null"]


@pytest.mark.parametrize(
    ("uid", "expected"), [("u_plain", "not-json-at-all"), ("u_scalar", "12345")]
)
def test_non_object_oauth_strings_are_left_alone(
    repaired_oauth_rows: dict, uid: str, expected: str
) -> None:
    """Only strings that decode to an object are rewritten."""
    row = repaired_oauth_rows[uid]
    assert row["type"] == "str" and row["value"] == expected, row


def test_legacy_oauth_sub_migrates_to_an_object(open_webui_backend: Path, tmp_path: Path) -> None:
    """b10670c03dd5 itself must write an object, not a serialized string."""
    row = _migrate_and_read(
        open_webui_backend, tmp_path, _OAUTH_SUB_BODY, "oauth_sub -> oauth conversion"
    )
    assert row["type"] == "dict", (
        f"the oauth_sub conversion wrote {row['type']} into the JSON column, so new "
        f"upgrades keep producing unusable oauth rows. value={row['value']!r}"
    )
    assert row["value"] == _OAUTH_OBJECT, row["value"]


# --------------------------------------------------------------------------
# 85. Repairing default model settings.
# --------------------------------------------------------------------------


async def _run_config_repair(config_model_module, monkeypatch, tmp_path: Path, seed: dict) -> dict:
    """Run the startup config repair against a throwaway sqlite config table."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    Config = config_model_module.Config
    db_path = (tmp_path / "config.db").resolve()
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Config.__table__.create)
        async with session_maker() as db:
            for key, value in seed.items():
                db.add(Config(key=key, value=value, updated_at=0))
            await db.commit()

        @asynccontextmanager
        async def _scratch_db():
            async with session_maker() as db:
                yield db

        monkeypatch.setattr(config_model_module, "get_async_db", _scratch_db)
        monkeypatch.setattr(Config, "PERSISTENT_ENABLED", True)

        # v0.11.0 only has the old name; run whichever the checkout provides so
        # the assertion is about behaviour, not about the rename.
        repair = getattr(Config, "repair_config_rows", None) or Config.repair_flattened_dict_configs
        await repair()

        async with session_maker() as db:
            rows = (await db.execute(config_model_module.select(Config))).scalars().all()
            return {row.key: row.value for row in rows}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["ui.default_models", "ui.default_pinned_models"])
async def test_list_shaped_default_model_rows_are_flattened(
    config_model_module, monkeypatch, tmp_path: Path, key: str
) -> None:
    """A list-shaped row is rewritten to the comma-separated string the app reads."""
    stored = await _run_config_repair(
        config_model_module, monkeypatch, tmp_path, {key: ["gpt-4o", "  llama3  ", ""]}
    )
    assert stored[key] == "gpt-4o,llama3", (
        f"{key} was left as {stored[key]!r}; the instance's default models stay ignored"
    )


@pytest.mark.asyncio
async def test_string_shaped_default_model_row_is_untouched(
    config_model_module, monkeypatch, tmp_path: Path
) -> None:
    stored = await _run_config_repair(
        config_model_module, monkeypatch, tmp_path, {"ui.default_models": "gpt-4o,llama3"}
    )
    assert stored["ui.default_models"] == "gpt-4o,llama3", stored


@pytest.mark.asyncio
async def test_flattened_dict_config_rows_still_reassemble(
    config_model_module, monkeypatch, tmp_path: Path
) -> None:
    """The pre-existing repair pass survived the rename."""
    stored = await _run_config_repair(
        config_model_module,
        monkeypatch,
        tmp_path,
        {"user.permissions.chat.controls": False, "user.permissions.workspace.models": True},
    )
    assert stored["user.permissions"] == {
        "chat": {"controls": False},
        "workspace": {"models": True},
    }, stored
    assert "user.permissions.chat.controls" not in stored, stored


# --------------------------------------------------------------------------
# 195. Connections whose tags were saved as plain text.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def main_module(owui_module):
    return owui_module("open_webui.main")


async def _call_get_models(main_module, monkeypatch, models: list[dict]) -> list[dict]:
    """Drive the real `main.get_models`, mocking only the model source and ACL."""

    async def _all_models(request, refresh=False, user=None):
        return models

    async def _filtered(models, user):
        return models

    async def _config_get(key, *args, **kwargs):
        return None

    monkeypatch.setattr(main_module, "get_all_models", _all_models)
    monkeypatch.setattr(main_module, "get_filtered_models", _filtered)
    monkeypatch.setattr(main_module.Config, "get", _config_get)

    response = await main_module.get_models(request=None, refresh=False, user=object())
    return response["data"]


@pytest.mark.asyncio
async def test_string_tags_from_a_connection_survive_get_models(
    main_module, monkeypatch
) -> None:
    """Tags stored as plain strings must come back as tag objects, not blanked."""
    models = [
        {"id": "conn-model", "name": "Conn Model", "info": {"meta": {"tags": ["alpha", "beta"]}}}
    ]
    data = await _call_get_models(main_module, monkeypatch, models)

    assert data[0]["tags"] == [{"name": "alpha"}, {"name": "beta"}], (
        f"string tags were dropped instead of normalized: {data[0]['tags']!r} (#28749)"
    )


@pytest.mark.asyncio
async def test_string_tags_are_normalized_in_place_for_the_connection_editor(
    main_module, monkeypatch
) -> None:
    """`info.meta.tags` is what the connection editor panel reads back."""
    models = [
        {"id": "conn-model", "name": "Conn Model", "info": {"meta": {"tags": ["alpha", "beta"]}}}
    ]
    data = await _call_get_models(main_module, monkeypatch, models)

    assert data[0]["info"]["meta"]["tags"] == [{"name": "alpha"}, {"name": "beta"}], data[0]["info"]


@pytest.mark.asyncio
async def test_null_info_does_not_break_get_models(main_module, monkeypatch) -> None:
    """A model with `info: None` must not take the whole listing down."""
    models = [{"id": "bare-model", "name": "Bare", "info": None}]
    data = await _call_get_models(main_module, monkeypatch, models)

    assert data[0]["tags"] == [], data[0]


@pytest.mark.asyncio
async def test_dict_tags_still_work(main_module, monkeypatch) -> None:
    models = [
        {
            "id": "conn-model",
            "name": "Conn Model",
            "info": {"meta": {"tags": [{"name": "alpha"}]}},
            "tags": [{"name": "beta"}],
        }
    ]
    data = await _call_get_models(main_module, monkeypatch, models)

    assert {tag["name"] for tag in data[0]["tags"]} == {"alpha", "beta"}, data[0]["tags"]


@pytest.mark.asyncio
async def test_profile_image_url_is_still_stripped(main_module, monkeypatch) -> None:
    models = [
        {
            "id": "conn-model",
            "name": "Conn Model",
            "info": {"meta": {"profile_image_url": "data:image/png;base64,AAA", "tags": []}},
        }
    ]
    data = await _call_get_models(main_module, monkeypatch, models)

    assert "profile_image_url" not in data[0]["info"]["meta"], data[0]["info"]["meta"]
