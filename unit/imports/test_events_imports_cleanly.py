"""Guard: open_webui.events must import standalone without a circular import.

events.py is the pub/sub hub between automations, chat, and retrieval
(35+ call sites), and it was one link in the actual chain open-webui/open-webui
#29280 (commit 8c0c7b3b6, v0.11.3) broke: models/calendar.py -> utils/automations.py
-> events.py -> retrieval/web/utils.py -> back into a still-loading config.py.
That fix only moved one import in calendar.py; nothing guards events.py
against gaining a new bad import of its own from one of its many other
importers, which is what this test does.

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
def test_events_imports_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_fresh_import: ImportRunner
) -> None:
    db_path = (tmp_path / "webui.db").resolve()
    db_url = f"sqlite:///{db_path.as_posix()}"

    rc, stdout, stderr = run_fresh_import(open_webui_backend, db_url, "open_webui.events", tmp_path)

    if rc != 0:
        pytest.fail(
            f"import open_webui.events failed. This breaks automations, chat "
            f"notifications, and anything else routed through the event bus.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )

    assert "OK" in stdout, stdout

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "config" in tables, f"config table missing after import; got: {sorted(tables)}"
