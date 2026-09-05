"""Guard: the ORM models and the migration chain must describe one schema.

Every table and column the models declare has to be created by a migration.
Nothing enforces that: SQLAlchemy builds its metadata from the model classes
and alembic builds the database from the revision files, and the two only meet
at runtime. Adding a column to a model and forgetting the migration produces a
green test suite, a green startup, and then `OperationalError: no such column`
on every query that touches the table — for everyone who upgrades, on a
database that already exists.

Only the direction that breaks is asserted. Columns and tables the database
has but no model declares are legacy leftovers (`document`, `chatidtag`,
`config_old`, alembic's own bookkeeping) and are harmless.

Runs `upgrade head` against a fresh SQLite file and imports every module under
models/ in one subprocess, because both cache module state on first import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BODY = """
import importlib, json, pathlib
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect
from open_webui.env import OPEN_WEBUI_DIR

cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
command.upgrade(cfg, 'head')

from open_webui.internal.db import Base

for path in sorted((OPEN_WEBUI_DIR / 'models').glob('*.py')):
    if path.stem != '__init__':
        importlib.import_module('open_webui.models.' + path.stem)

inspector = inspect(create_engine(DATABASE_URL))
live = {name: {c['name'] for c in inspector.get_columns(name)}
        for name in inspector.get_table_names()}
declared = {table.name: {column.name for column in table.columns}
            for table in Base.metadata.sorted_tables}

print('---RESULT---')
print(json.dumps({
    'declared_tables': sorted(declared),
    'missing_tables': sorted(set(declared) - set(live)),
    'missing_columns': {name: sorted(declared[name] - live[name])
                        for name in declared
                        if name in live and declared[name] - live[name]},
}))
"""


@pytest.fixture(scope="module")
def schema_comparison(open_webui_backend: Path, tmp_path_factory) -> dict:
    """Migrate a fresh database, then read the models' own metadata beside it."""
    scratch = tmp_path_factory.mktemp("schema-drift")
    data_dir = scratch / "data"
    data_dir.mkdir()
    database_url = f"sqlite:///{(data_dir / 'webui.db').as_posix()}"

    script = (
        textwrap.dedent(
            f"""
        import os, sys
        os.environ['WEBUI_SECRET_KEY'] = 'test-secret-key'
        os.environ['DATABASE_URL'] = {database_url!r}
        os.environ['DATA_DIR'] = {str(data_dir)!r}
        os.environ['STATIC_DIR'] = {str(scratch / "static")!r}
        os.environ['ENABLE_DB_MIGRATIONS'] = 'false'
        sys.path.insert(0, {str(open_webui_backend)!r})
        DATABASE_URL = {database_url!r}
        """
        )
        + BODY
    )

    (scratch / "static").mkdir()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode != 0 or "---RESULT---" not in result.stdout:
        pytest.fail(
            "could not build the schema comparison.\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{result.stdout[-1000:]}"
        )
    return json.loads(result.stdout.split("---RESULT---", 1)[1].strip().splitlines()[-1])


def test_the_models_declare_a_schema_at_all(schema_comparison: dict) -> None:
    """A guard on the guard: an import that quietly loaded nothing would make
    the two tests below pass against an empty comparison."""
    assert len(schema_comparison["declared_tables"]) > 20, schema_comparison["declared_tables"]


def test_every_model_table_is_created_by_a_migration(schema_comparison: dict) -> None:
    """A model with no migration behind it means every query against it fails
    on a database that was not created from scratch today."""
    assert not schema_comparison["missing_tables"], (
        f"tables declared by a model but never created by a migration: "
        f"{schema_comparison['missing_tables']}"
    )


def test_every_model_column_is_created_by_a_migration(schema_comparison: dict) -> None:
    """The common half of the same accident: the column is added to the model,
    the migration is forgotten, and every existing install breaks on upgrade."""
    assert not schema_comparison["missing_columns"], (
        f"columns declared by a model but missing from the migrated schema: "
        f"{schema_comparison['missing_columns']}"
    )
