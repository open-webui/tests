"""Guard: open_webui.config must import standalone without a circular import.

config.py is imported by more of the backend than any other in-package module
(60+ call sites) and runs migrations as a side effect of being imported
(`if ENABLE_DB_MIGRATIONS: run_migrations()` at module scope). A module
anywhere in the migration import graph that imports back from config before
config has finished loading breaks every fresh install and every upgrade,
exactly what happened in open-webui/open-webui#29280 (commit 8c0c7b3b6,
v0.11.3), where models/calendar.py's import chain closed that cycle. This
guards config.py's own standalone import directly, independent of which
module might reach back into it next.

Discriminates: passes on dev bbfa876af, fails once a copy's `models/calendar.py` imports
from `open_webui.config` at module scope (the #29280 cycle).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


def test_config_imports_without_a_cycle(cold_import) -> None:
    tables = cold_import("open_webui.config")

    assert "config" in tables, f"the migrations never ran; tables: {sorted(tables)}"
