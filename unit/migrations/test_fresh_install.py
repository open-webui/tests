"""Regression tests for fresh-database install of open-webui.

After commit 2c2d06c31 ("refac: deprecate peewee migration layer")
removed the peewee bootstrap, two alembic migrations that previously
ran against peewee-shaped tables started breaking *fresh* installs
because the new init migration (7e5b5dc7342b) leaves the schema in
a different shape:

  Postgres — 38d63c18f30f (Add oauth_session table)
    Gated its user-PK fixup on `not id_column.get('unique', False)`.
    On Postgres, PK columns aren't separately marked unique, so the
    gate is True on fresh DBs; the block then unconditionally calls
    `create_primary_key('pk_user_id', ['id'])` on a table that already
    has a PK on `id`. Postgres aborts with InvalidTableDefinition;
    DDL is transactional, so the whole upgrade chain rolls back and
    the config table never persists. On startup, STATE.load() crashes
    with `psycopg2.errors.UndefinedTable: relation "config" does not
    exist`. (Reported by urbenlegend, last reply on the #24560 thread.)

  SQLite — b10670c03dd5 (Update user table)
    `_drop_sqlite_indexes_for_column` issued DROP INDEX on every index
    referencing the target column, including the auto-created indexes
    that back UNIQUE constraints (sqlite_autoindex_*). SQLite refuses
    with "index associated with UNIQUE or PRIMARY KEY constraint
    cannot be dropped". The migration chain stops partway, later
    tables are missing.

Both reproduce on a *fresh* DB with no prior peewee migration history.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run_alembic_upgrade_head(
    backend: Path, database_url: str, data_dir: Path
) -> tuple[int, str, str]:
    """Run alembic upgrade head as a subprocess with the given DB URL.

    Uses a subprocess so each invocation gets a fresh sys.modules — the
    open_webui.config module caches state in module globals after first
    import. Bypasses open_webui.config's run_migrations() wrapper
    (which silently swallows migration errors with `log.exception(...)`),
    so the alembic exception propagates as a non-zero exit code.
    """
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ['DATABASE_URL'] = {database_url!r}
        os.environ['DATA_DIR'] = {str(data_dir)!r}
        sys.path.insert(0, {str(backend)!r})

        from alembic import command
        from alembic.config import Config as AlembicConfig
        from open_webui.env import OPEN_WEBUI_DIR

        cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
        cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
        command.upgrade(cfg, 'head')

        # Sanity check: the config table — the one that surfaced the
        # original symptom — must exist after upgrade.
        from sqlalchemy import create_engine, inspect
        engine = create_engine({database_url!r})
        tables = set(inspect(engine).get_table_names())
        engine.dispose()
        assert 'config' in tables, f'config table missing after upgrade; got: {{sorted(tables)}}'

        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.regression
def test_alembic_upgrade_head_succeeds_on_fresh_sqlite(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Regression: alembic upgrade head must succeed on a fresh SQLite DB.

    Catches the b10670c03dd5 SQLite-only failure ("index associated
    with UNIQUE or PRIMARY KEY constraint cannot be dropped") which
    stops the migration chain partway through every fresh install.
    """
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rc, stdout, stderr = _run_alembic_upgrade_head(open_webui_backend, db_url, data_dir)

    if rc != 0:
        pytest.fail(
            f"alembic upgrade head failed on fresh SQLite. "
            f"This breaks the default `docker run` path and every fresh "
            f"sqlite install.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout


@pytest.mark.regression
def test_alembic_upgrade_head_succeeds_on_fresh_postgres(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Regression: alembic upgrade head must succeed on a fresh Postgres DB.

    Catches the 38d63c18f30f Postgres-only failure ("multiple primary
    keys for table 'user' are not allowed") that rolls the whole
    transaction back, leaving the database with no tables — config
    included. The user-visible symptom on startup is

        psycopg2.errors.UndefinedTable: relation "config" does not exist

    Uses pgserver (embedded Postgres) so the test runs anywhere with
    the `pgserver` PyPI package installed. Skips if pgserver isn't
    available.
    """
    pgserver = pytest.importorskip(
        "pgserver", reason="pgserver not installed (pip install pgserver)"
    )

    pg_dir = tmp_path / "pgdata"
    pg_dir.mkdir()

    server = pgserver.get_server(str(pg_dir), cleanup_mode=None)
    try:
        # pgserver gives us a postgresql:// URI; switch to the
        # postgresql+psycopg2:// form sqlalchemy expects.
        db_url = server.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        rc, stdout, stderr = _run_alembic_upgrade_head(open_webui_backend, db_url, data_dir)

        if rc != 0:
            pytest.fail(
                f"alembic upgrade head failed on fresh Postgres. "
                f"This is the urbenlegend report on open-webui#24560 — "
                f"every fresh Postgres install crashes on startup with "
                f"'relation \"config\" does not exist' because the alembic "
                f"transaction is rolled back.\n"
                f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
                f"--- stdout (tail) ---\n{stdout[-1000:]}"
            )

        assert "OK" in stdout, stdout
    finally:
        try:
            server.cleanup()
        except Exception:
            pass
