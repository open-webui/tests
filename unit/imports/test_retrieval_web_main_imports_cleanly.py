"""Guard: open_webui.retrieval.web.main must import standalone without a circular import.

retrieval/web/main.py wires up every web-search provider and is imported
across the retrieval stack (34+ call sites). Its sibling in the same package,
retrieval/web/utils.py, was the module that actually closed the cycle in
open-webui/open-webui#29280 (commit 8c0c7b3b6, v0.11.3) by importing back from
a still-loading open_webui.config. This guards the other heavily-imported
file in that same package against the same failure mode.

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
def test_retrieval_web_main_imports_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_fresh_import: ImportRunner
) -> None:
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"

    rc, stdout, stderr = run_fresh_import(
        open_webui_backend, db_url, "open_webui.retrieval.web.main", tmp_path
    )

    if rc != 0:
        pytest.fail(
            f"import open_webui.retrieval.web.main failed. This breaks every "
            f"web-search provider.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "config" in tables, f"config table missing after import; got: {sorted(tables)}"
