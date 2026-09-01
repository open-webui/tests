"""Regression: alembic's migration environment could not import cleanly
(commit 8c0c7b3b6, issue #29280, shipped in v0.11.3).

`models/calendar.py` imported `rrule_interval_seconds` from `utils.automations`
at module scope. That module chains into `events.py`, then
`retrieval/web/utils.py`, which imports back from `open_webui.config` for a
name defined further down that same file. `migrations/env.py` imports
`Calendar` before `config` has finished loading (importing `config` runs
migrations as a side effect), so the chain closed a real cycle: Python
returned the partially-initialized `calendar` module and the `Calendar` name
was not there yet.

`test_migration_failure_aborts_startup.py` covers the other half of this same
commit (a failed upgrade must abort startup instead of being logged and
swallowed) against a mocked `alembic.command.upgrade`, so it cannot observe
this import cycle at all. This test runs the real upgrade instead, which is
where the cycle actually lived.

Discriminates: passes on v0.11.3, fails on v0.11.2 (`alembic upgrade head`
raises before a single table is created).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from .conftest import AlembicUpgradeRunner


@pytest.mark.regression
def test_alembic_env_imports_the_calendar_model_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_alembic_upgrade_head: AlembicUpgradeRunner
) -> None:
    """Regression: migrations/env.py's import of Calendar must not cycle back into config."""
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rc, stdout, stderr = run_alembic_upgrade_head(open_webui_backend, db_url, data_dir)

    if rc != 0:
        pytest.fail(
            f"alembic upgrade head failed while loading migrations/env.py. "
            f"This breaks every fresh install and every upgrade.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "calendar" in tables, f"calendar table missing after upgrade; got: {sorted(tables)}"
