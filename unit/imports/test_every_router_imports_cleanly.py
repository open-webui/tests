"""Guard: every router module must import on its own, from a cold interpreter.

Booting the app imports main.py, which pulls in every router in one fixed order. That hides a
cycle that only fires when a router is imported first: whichever module the cycle runs through
is already finished in sys.modules by the time the later router asks for it, so the
partial-import error never happens. Alembic's env.py, the `open-webui` CLI and any script that
imports a single router all hit the uncovered order.

One fresh interpreter per router, run side by side, since each import drags in the whole
retrieval and provider tree. Marked slow; deselect with -m 'not slow'.

Discriminates: passes on dev bbfa876af, fails for every router once a copy's
`models/calendar.py` imports from `open_webui.config` at module scope (the #29280 cycle).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .conftest import import_cold

ROUTERS_DIR = Path("open_webui") / "routers"


@pytest.mark.slow
def test_every_router_imports_without_a_cycle(open_webui_backend: Path, tmp_path: Path) -> None:
    routers = sorted(
        path.stem
        for path in (open_webui_backend / ROUTERS_DIR).glob("*.py")
        if path.stem != "__init__"
    )
    assert routers, f"no router modules under {open_webui_backend / ROUTERS_DIR}"

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        imports = pool.map(
            lambda router: import_cold(
                open_webui_backend, tmp_path / router, f"open_webui.routers.{router}"
            ),
            routers,
        )
        failed = [imported for imported in imports if imported.failure]

    assert not failed, "routers that fail a cold import:\n" + "\n".join(
        f"=== {imported.module}\n{imported.failure}" for imported in failed
    )
