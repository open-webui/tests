"""Guard: open_webui.utils.auth must import standalone without a circular import.

utils/auth.py backs the `get_verified_user`/`get_admin_user` dependency used
by nearly every router (39+ call sites) and, like models/calendar.py in
open-webui/open-webui#29280 (commit 8c0c7b3b6, v0.11.3), its own import chain
reaches back into open_webui.config, which runs migrations as a side effect
of being imported. It sits outside the migrations/env.py import graph that
bug lived in (that pulls in models/auths.py, a different module despite the
similar name), so this covers the same bug class on a path of its own.

Discriminates: passes on dev bbfa876af, fails once a copy's `models/calendar.py` imports
from `open_webui.config` at module scope (the #29280 cycle).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


def test_auth_imports_without_a_cycle(cold_import) -> None:
    tables = cold_import("open_webui.utils.auth")

    assert "config" in tables, f"the migrations never ran; tables: {sorted(tables)}"
