"""Guard: open_webui.retrieval.web.main must import standalone without a circular import.

retrieval/web/main.py wires up every web-search provider and is imported
across the retrieval stack (34+ call sites). Its sibling in the same package,
retrieval/web/utils.py, was the module that actually closed the cycle in
open-webui/open-webui#29280 (commit 8c0c7b3b6, v0.11.3) by importing back from
a still-loading open_webui.config. This guards the other heavily-imported
file in that same package against the same failure mode.

Discriminates: passes on dev bbfa876af, fails once a copy's `models/calendar.py` imports
from `open_webui.config` at module scope (the #29280 cycle).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


def test_retrieval_web_main_imports_without_a_cycle(cold_import) -> None:
    tables = cold_import("open_webui.retrieval.web.main")

    assert "config" in tables, f"the migrations never ran; tables: {sorted(tables)}"
