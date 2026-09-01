"""Guard: open_webui.config must import standalone without a circular import.

config.py is imported by more of the backend than any other in-package module
(60+ call sites) and runs migrations as a side effect of being imported
(`if ENABLE_DB_MIGRATIONS: run_migrations()` at module scope). A module
anywhere in the migration import graph that imports back from config before
config has finished loading breaks every fresh install and every upgrade —
exactly what happened in open-webui/open-webui#29280 (commit 8c0c7b3b6,
v0.11.3), where models/calendar.py's import chain closed that cycle. This
guards config.py's own standalone import directly, independent of which
module might reach back into it next.

run_migrations()'s own except block swallows exactly this kind of failure, so
a plain `import` can succeed even when the cycle fired and migrations never
ran. Checking that the config table actually exists afterward is what makes
this test meaningful; checking only the import's exit code is not enough.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from .conftest import ImportRunner


@pytest.mark.regression
def test_config_imports_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_fresh_import: ImportRunner
) -> None:
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"

    rc, stdout, stderr = run_fresh_import(open_webui_backend, db_url, "open_webui.config", tmp_path)

    if rc != 0:
        pytest.fail(
            f"import open_webui.config failed. This breaks every fresh install "
            f"and every upgrade.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "config" in tables, f"config table missing after import; got: {sorted(tables)}"
