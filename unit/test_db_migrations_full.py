"""Full migration + install tests against fresh SQLite and Postgres.

Stronger than test_fresh_db_migrations.py — instead of just checking
that `alembic upgrade head` returns 0 and that one table exists, this
suite parametrises across both backends and exercises:

  test_upgrade_head_runs_clean         alembic chain completes
  test_critical_tables_exist           every must-have table is present
  test_minimum_table_count             chain didn't stop early
  test_open_webui_config_imports       full import path the user actually hits
  test_user_crud_round_trip            schema is usable, not just present
  test_upgrade_is_idempotent           re-running upgrade is a no-op
  test_downgrade_to_base_clean         schema unwinds without errors

Each test gets its own fresh database, automatically created and
cleaned up by the `fresh_db` fixture:

  - SQLite — temp file under pytest's tmp_path (auto-deleted).
  - Postgres — embedded `pgserver`, instance under tmp_path, server
    stopped and pgdata removed on fixture teardown. Test is skipped if
    `pgserver` isn't installed (`pip install pgserver`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest


# Tables that every successful upgrade must produce. Catches partial
# migration chains: any one of these missing means the chain stopped
# before reaching the migration that creates it.
_CRITICAL_TABLES: frozenset[str] = frozenset({
    'auth', 'chat', 'config', 'function', 'group', 'knowledge',
    'knowledge_directory', 'knowledge_file', 'memory', 'note',
    'oauth_session', 'prompt', 'tag', 'tool', 'user',
})

# Sanity lower-bound on total tables. Each new migration usually adds
# 1+ tables — currently dev is at ~40. Set well below that so the
# threshold isn't a magnet for churn.
_MIN_TABLE_COUNT = 30


# -----------------------------------------------------------------------------
# Subprocess helper
# -----------------------------------------------------------------------------


def _run_python(
    backend: Path, db_url: str, data_dir: Path, body: str, *, timeout: int = 180
) -> subprocess.CompletedProcess:
    """Run a Python snippet with `backend/` on sys.path and DB env set.

    Each test runs in its own subprocess so sys.modules state doesn't
    leak between tests (open_webui caches a lot at import time).
    """
    preamble = textwrap.dedent(
        f"""
        import json, os, sys
        os.environ['DATABASE_URL'] = {db_url!r}
        os.environ['DATA_DIR'] = {str(data_dir)!r}
        sys.path.insert(0, {str(backend)!r})
        """
    )
    return subprocess.run(
        [sys.executable, '-c', preamble + body],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    )


def _assert_ok(result: subprocess.CompletedProcess, sentinel: str, label: str) -> None:
    if result.returncode != 0 or sentinel not in result.stdout:
        pytest.fail(
            f'{label} failed.\n'
            f'--- stderr (tail) ---\n{result.stderr[-3000:]}\n'
            f'--- stdout (tail) ---\n{result.stdout[-1500:]}'
        )


# -----------------------------------------------------------------------------
# Fresh-DB fixtures
# -----------------------------------------------------------------------------


@contextmanager
def _sqlite_fresh(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """SQLite at tmp_path/webui.db. tmp_path is auto-cleaned by pytest."""
    db_path = (tmp_path / 'webui.db').resolve()
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    yield (f'sqlite:///{db_path.as_posix()}', data_dir)


@contextmanager
def _postgres_fresh(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Embedded Postgres via pgserver. Stops + removes pgdata on exit."""
    pgserver = pytest.importorskip(
        'pgserver', reason='pgserver not installed (pip install pgserver)'
    )
    pg_dir = tmp_path / 'pgdata'
    pg_dir.mkdir(exist_ok=True)
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    server = pgserver.get_server(str(pg_dir), cleanup_mode=None)
    try:
        url = server.get_uri().replace('postgresql://', 'postgresql+psycopg2://', 1)
        yield (url, data_dir)
    finally:
        try:
            server.cleanup()
        except Exception:
            pass


@pytest.fixture(params=['sqlite', 'postgres'])
def fresh_db(request, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Yield (DATABASE_URL, DATA_DIR) for a fresh DB; cleans up after.

    Parametrised on backend so every test runs against both engines.
    """
    factory = _sqlite_fresh if request.param == 'sqlite' else _postgres_fresh
    with factory(tmp_path) as resource:
        yield resource


# -----------------------------------------------------------------------------
# Body chunks (executed inside the subprocess)
# -----------------------------------------------------------------------------


_BODY_UPGRADE = """
from alembic import command
from alembic.config import Config as AlembicConfig
from open_webui.env import OPEN_WEBUI_DIR

cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
command.upgrade(cfg, 'head')
"""


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.regression
def test_upgrade_head_runs_clean(open_webui_backend: Path, fresh_db) -> None:
    """alembic upgrade head completes without raising on a fresh DB."""
    db_url, data_dir = fresh_db
    result = _run_python(
        open_webui_backend, db_url, data_dir, _BODY_UPGRADE + "\nprint('OK')\n"
    )
    _assert_ok(result, 'OK', 'alembic upgrade head')


@pytest.mark.regression
def test_critical_tables_exist(open_webui_backend: Path, fresh_db) -> None:
    """Every must-have table is present after upgrade head."""
    db_url, data_dir = fresh_db
    body = _BODY_UPGRADE + textwrap.dedent("""
        from sqlalchemy import create_engine, inspect
        engine = create_engine(os.environ['DATABASE_URL'])
        tables = sorted(inspect(engine).get_table_names())
        engine.dispose()
        print('TABLES:', json.dumps(tables))
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'TABLES:', 'collecting table list')

    line = next(
        (l for l in result.stdout.splitlines() if l.startswith('TABLES:')), None
    )
    assert line is not None
    tables = set(json.loads(line.removeprefix('TABLES:').strip()))

    missing = _CRITICAL_TABLES - tables
    assert not missing, (
        f'Critical tables missing after upgrade head: {sorted(missing)}. '
        f'Present: {sorted(tables)}'
    )


@pytest.mark.regression
def test_minimum_table_count(open_webui_backend: Path, fresh_db) -> None:
    """Total table count is at the lower bound — catches "chain stopped
    early" cases that still happen to include the critical subset."""
    db_url, data_dir = fresh_db
    body = _BODY_UPGRADE + textwrap.dedent("""
        from sqlalchemy import create_engine, inspect
        engine = create_engine(os.environ['DATABASE_URL'])
        tables = sorted(inspect(engine).get_table_names())
        engine.dispose()
        print('COUNT:', len(tables))
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'COUNT:', 'collecting table count')

    line = next(
        (l for l in result.stdout.splitlines() if l.startswith('COUNT:')), None
    )
    assert line is not None
    count = int(line.removeprefix('COUNT:').strip())
    assert count >= _MIN_TABLE_COUNT, (
        f'Only {count} tables after upgrade head; expected >= {_MIN_TABLE_COUNT}. '
        f'Migration chain probably stopped partway.'
    )


@pytest.mark.regression
def test_open_webui_config_imports_cleanly(
    open_webui_backend: Path, fresh_db
) -> None:
    """Full user-facing import path: importing `open_webui.config`
    must not raise, and CONFIG_DATA must populate. This is the exact
    path that crashed in urbenlegend's open-webui#24560 report."""
    db_url, data_dir = fresh_db
    body = textwrap.dedent("""
        import open_webui.config as c
        assert isinstance(c.CONFIG_DATA, dict), repr(c.CONFIG_DATA)
        # STATE.load() ran without raising — that's the regression check.
        print('CONFIG_OK:', json.dumps(c.CONFIG_DATA))
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'CONFIG_OK', 'open_webui.config import')


@pytest.mark.regression
def test_user_crud_round_trip(open_webui_backend: Path, fresh_db) -> None:
    """Schema is usable, not just present: create / read / update /
    delete a user via the SQLAlchemy User model. Catches schema bugs
    that pass alembic but break ORM use (wrong column type, missing
    nullable, broken JSON cast, etc.)."""
    db_url, data_dir = fresh_db
    body = _BODY_UPGRADE + textwrap.dedent("""
        import time
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from open_webui.models.users import User

        engine = create_engine(os.environ['DATABASE_URL'])
        Session = sessionmaker(bind=engine)
        s = Session()

        now = int(time.time())
        uid = 'crud-test-user'
        s.add(User(
            id=uid, name='Original', email='crud@example.com', role='user',
            last_active_at=now, created_at=now, updated_at=now,
        ))
        s.commit()

        fetched = s.query(User).filter_by(id=uid).one()
        assert fetched.email == 'crud@example.com', fetched.email
        assert fetched.name == 'Original', fetched.name

        fetched.name = 'Renamed'
        s.commit()

        again = s.query(User).filter_by(id=uid).one()
        assert again.name == 'Renamed', again.name

        s.delete(again)
        s.commit()

        assert s.query(User).filter_by(id=uid).first() is None
        s.close()
        engine.dispose()
        print('CRUD_OK')
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'CRUD_OK', 'User CRUD round-trip')


@pytest.mark.regression
def test_upgrade_is_idempotent(open_webui_backend: Path, fresh_db) -> None:
    """`alembic upgrade head` is safe to re-run — the second invocation
    should be a no-op, not an error. Container restarts and helm
    redeploys rely on this."""
    db_url, data_dir = fresh_db
    body = textwrap.dedent("""
        from alembic import command
        from alembic.config import Config as AlembicConfig
        from open_webui.env import OPEN_WEBUI_DIR

        cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
        cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))

        command.upgrade(cfg, 'head')
        command.upgrade(cfg, 'head')   # idempotent re-run
        print('IDEMPOTENT_OK')
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'IDEMPOTENT_OK', 're-running upgrade head')


@pytest.mark.regression
def test_downgrade_to_base_clean(open_webui_backend: Path, fresh_db) -> None:
    """`alembic downgrade base` after `upgrade head` must unwind the
    schema completely. Any leftover (other than alembic_version) means
    a migration has a broken downgrade()."""
    db_url, data_dir = fresh_db
    body = textwrap.dedent("""
        from alembic import command
        from alembic.config import Config as AlembicConfig
        from open_webui.env import OPEN_WEBUI_DIR
        from sqlalchemy import create_engine, inspect

        cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
        cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))

        command.upgrade(cfg, 'head')
        command.downgrade(cfg, 'base')

        engine = create_engine(os.environ['DATABASE_URL'])
        remaining = sorted(inspect(engine).get_table_names())
        engine.dispose()

        # alembic_version is alembic's own bookkeeping table; everything
        # else should be gone after downgrade to base.
        leftover = [t for t in remaining if t != 'alembic_version']
        print('LEFTOVER:', json.dumps(leftover))
    """)
    result = _run_python(open_webui_backend, db_url, data_dir, body)
    _assert_ok(result, 'LEFTOVER:', 'downgrade to base')

    line = next(
        (l for l in result.stdout.splitlines() if l.startswith('LEFTOVER:')), None
    )
    assert line is not None
    leftover = json.loads(line.removeprefix('LEFTOVER:').strip())
    assert not leftover, (
        f'Tables left after downgrade base: {leftover}. '
        f"At least one migration's downgrade() doesn't fully reverse upgrade()."
    )
