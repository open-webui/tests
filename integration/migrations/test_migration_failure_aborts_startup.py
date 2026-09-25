"""Regression: a failed Alembic upgrade was swallowed at startup (commit 8c0c7b3b6, issue
#29280, shipped in v0.11.3).

`run_migrations()` wrapped the upgrade in a `try/except Exception` that only logged, so a broken
migration left the instance running on a half-migrated schema until the first query met a
missing table or column. The fix re-raises, so the boot stops at the migration error itself.
Here the instance boots on a copy of a migrated database whose recorded revision
(`alembic_version`) names one no migration file has, which Alembic refuses to upgrade from. The
unaltered copy booting is the control: nothing else about the copy stops the boot.

Twin of unit/migrations/test_migration_failure_aborts_startup.py.
Discriminates: passes on dev bbfa876af, fails with the `raise` in `run_migrations` removed in a
copy of it (the instance answers /health on the unupgradable database).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from harness import backends
from harness.prepared_data import boot_until_settled, snapshot_database

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.skipif(
        backends.DATABASE == "postgres", reason="copies the SQLite file, none on Postgres"
    ),
]

UNKNOWN_REVISION = "deadbeef"


def test_the_unaltered_database_boots(instance, tmp_path):
    """Control: the copy itself is a database the checkout starts on."""
    snapshot_database(instance, tmp_path)

    outcome = boot_until_settled(tmp_path)

    assert outcome.healthy, f"the copied database does not boot:\n{outcome.log[-3000:]}"


def test_a_failing_migration_stops_the_boot(instance, tmp_path):
    database = snapshot_database(instance, tmp_path)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE alembic_version SET version_num = ?", (UNKNOWN_REVISION,))

    outcome = boot_until_settled(tmp_path)

    assert not outcome.healthy, (
        "the instance started on a database its migrations could not upgrade, so its first "
        "query meets a half-migrated schema (#29280)"
    )
    assert outcome.exit_code, f"the boot neither started nor failed:\n{outcome.log[-3000:]}"
    assert UNKNOWN_REVISION in outcome.log, "the boot stopped without naming the failed upgrade"
