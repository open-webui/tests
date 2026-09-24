"""Guard: the migration chain installs, re-runs and unwinds cleanly on SQLite and on Postgres.

Every startup runs `alembic upgrade head`, and since v0.11.3 a failure there stops the boot, so
a chain that breaks on a fresh database is a dead install. Regressions pinned here:

* #29280 (8c0c7b3b6, v0.11.3): `migrations/env.py` imports `Calendar`, whose import chain
  reached back into a still-loading `open_webui.config`; `upgrade head` raised before a single
  table existed.
* 38d63c18f30f (Postgres, #24560): recreated the user primary key on a fresh database; DDL is
  transactional, so the whole chain rolled back and startup crashed on `relation "config" does
  not exist`.
* b10670c03dd5 (SQLite): dropped the index backing a UNIQUE constraint, which SQLite refuses,
  so the chain stopped partway and later tables were missing.

One run per engine, in a fresh interpreter: upgrade a new database to head, list its tables,
upgrade again (a restart), import the config module, round-trip a user through the model layer
and unwind to base. Each test reads one step of that run. A fresh SQLite install at boot is also
what every integration instance does; Postgres is only covered here.

Discriminates: passes on dev bbfa876af on both engines; a module-scope
`from open_webui.config import ...` in a copy's `models/calendar.py` (the #29280 cycle) fails
the upgrade on both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import postgres_at, sqlite_at

pytestmark = pytest.mark.regression

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
    "tag",
    "tool",
    "user",
}
# Well below the ~60 tables of dev, so new tables never make it churn.
MIN_TABLE_COUNT = 30

LIFECYCLE = """
import asyncio, importlib, traceback
from sqlalchemy import create_engine, inspect

report = {}


def step(name, action):
    try:
        report[name] = {'value': action()}
    except Exception:
        report[name] = {'error': traceback.format_exc()[-3000:]}
    return 'error' not in report[name]


def table_names():
    engine = create_engine(os.environ['DATABASE_URL'])
    try:
        return sorted(inspect(engine).get_table_names())
    finally:
        engine.dispose()


async def user_round_trip():
    from open_webui.models.users import Users

    user_id = 'lifecycle-user'
    await Users.insert_new_user(id=user_id, name='Original', email='life@example.com', role='user')
    stored = await Users.get_user_by_id(id=user_id)
    await Users.update_user_by_id(id=user_id, updated={'name': 'Renamed'})
    renamed = await Users.get_user_by_id(id=user_id)
    await Users.delete_user_by_id(id=user_id)
    gone = await Users.get_user_by_id(id=user_id) is None
    return [stored.email, renamed.name, gone]


if step('upgrade', lambda: command.upgrade(cfg, 'head')):
    step('tables', table_names)
    step('upgrade again', lambda: command.upgrade(cfg, 'head'))
    step('config import', lambda: importlib.import_module('open_webui.config').__name__)
    step('user round trip', lambda: asyncio.run(user_round_trip()))
    step('downgrade', lambda: command.downgrade(cfg, 'base'))
    step('tables after downgrade', table_names)
print('RESULT:' + json.dumps(report))
"""


@pytest.fixture(scope="module", params=["sqlite", "postgres"])
def lifecycle(request, open_webui_backend: Path, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp(request.param)
    if request.param == "sqlite":
        return sqlite_at(open_webui_backend, root).run(LIFECYCLE, what="the SQLite lifecycle")
    with postgres_at(open_webui_backend, root) as database:
        return database.run(LIFECYCLE, what="the Postgres lifecycle")


def _outcome(lifecycle: dict, step: str):
    if step not in lifecycle:
        pytest.fail(f"the run never reached {step!r}: the upgrade failed first")
    if "error" in lifecycle[step]:
        pytest.fail(f"{step} raised:\n{lifecycle[step]['error']}")
    return lifecycle[step]["value"]


def test_upgrade_head_runs_clean_on_a_fresh_database(lifecycle):
    _outcome(lifecycle, "upgrade")


def test_the_chain_creates_every_critical_table(lifecycle):
    tables = set(_outcome(lifecycle, "tables"))

    assert not CRITICAL_TABLES - tables, f"missing after upgrade: {CRITICAL_TABLES - tables}"
    assert len(tables) >= MIN_TABLE_COUNT, f"only {len(tables)} tables; the chain stopped early"


def test_upgrading_an_upgraded_database_again_is_a_no_op(lifecycle):
    """Container restarts and redeploys run it on every boot."""
    _outcome(lifecycle, "upgrade again")


def test_the_config_module_loads_on_the_migrated_database(lifecycle):
    assert _outcome(lifecycle, "config import") == "open_webui.config"


def test_a_user_round_trips_through_the_model_layer(lifecycle):
    """A usable schema: types, defaults and JSON columns included."""
    assert _outcome(lifecycle, "user round trip") == ["life@example.com", "Renamed", True]


def test_downgrade_to_base_unwinds_every_table(lifecycle):
    _outcome(lifecycle, "downgrade")
    leftover = set(_outcome(lifecycle, "tables after downgrade")) - {"alembic_version"}

    assert not leftover, f"a migration's downgrade() leaves tables behind: {sorted(leftover)}"
