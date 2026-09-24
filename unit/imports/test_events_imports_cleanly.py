"""Guard: open_webui.events must import standalone without a circular import.

events.py is the pub/sub hub between automations, chat, and retrieval
(35+ call sites), and it was one link in the actual chain open-webui/open-webui
#29280 (commit 8c0c7b3b6, v0.11.3) broke: models/calendar.py -> utils/automations.py
-> events.py -> retrieval/web/utils.py -> back into a still-loading config.py.
That fix only moved one import in calendar.py; nothing guards events.py
against gaining a new bad import of its own from one of its many other
importers, which is what this test does.

Discriminates: passes on dev bbfa876af, fails once a copy's `models/calendar.py` imports
from `open_webui.config` at module scope (the #29280 cycle).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


def test_events_imports_without_a_cycle(cold_import) -> None:
    tables = cold_import("open_webui.events")

    assert "config" in tables, f"the migrations never ran; tables: {sorted(tables)}"
