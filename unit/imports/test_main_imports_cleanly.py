"""Guard: open_webui.main must import standalone without a circular import.

main.py is the FastAPI entrypoint: importing it constructs the app and wires
up essentially every router, model and util in the backend (45+ top-level
open_webui.* imports, cascading further from there). It is the single
broadest cyclic-import canary available — a cycle anywhere reachable from app
boot surfaces here, not just the narrower migration-import graph that
open-webui/open-webui#29280 broke (commit 8c0c7b3b6, v0.11.3).

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
def test_main_imports_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_fresh_import: ImportRunner
) -> None:
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"

    rc, stdout, stderr = run_fresh_import(open_webui_backend, db_url, "open_webui.main", tmp_path)

    if rc != 0:
        pytest.fail(
            f"import open_webui.main failed. This breaks app startup outright.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "config" in tables, f"config table missing after import; got: {sorted(tables)}"
