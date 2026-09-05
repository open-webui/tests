"""Guard: every router module must import on its own, from a cold interpreter.

test_main_imports_cleanly.py imports main.py, which pulls in every router in
one fixed order. That hides a cycle that only fires when a router is imported
first: whichever module the cycle runs through is already finished in
sys.modules by the time the later router asks for it, so the partial-import
error never happens. Alembic's env.py, the `open-webui` CLI and any script
that imports a single router all hit the uncovered order.

One subprocess per router, so each import starts from an empty sys.modules —
the same reason the other guards in this directory use subprocesses.

Marked slow: importing a router drags in the whole retrieval and provider
dependency tree, around fifteen seconds each. Deselect with -m 'not slow'.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import ImportRunner

ROUTERS_DIR = Path("open_webui") / "routers"


def _router_modules() -> list[str]:
    """Collected at import time so each router is its own test id.

    Resolves the checkout the same way unit/conftest.py does; an empty list
    here just means the parametrized test never runs, and the source-missing
    skip comes from the open_webui_backend fixture as usual.
    """
    import os

    source = os.getenv("OPEN_WEBUI_SOURCE_DIR")
    candidates = [Path(source)] if source else []
    candidates += [parent / "open-webui" / "backend" for parent in Path(__file__).resolve().parents]
    for candidate in candidates:
        directory = candidate / ROUTERS_DIR
        if directory.is_dir():
            return sorted(p.stem for p in directory.glob("*.py") if p.stem != "__init__")
    return []


ROUTERS = _router_modules()


@pytest.mark.slow
@pytest.mark.parametrize("router", ROUTERS or ["<no checkout>"])
def test_router_imports_without_a_cycle(
    open_webui_backend: Path, tmp_path: Path, run_fresh_import: ImportRunner, router: str
) -> None:
    if not ROUTERS:
        pytest.skip("no open-webui checkout resolved at collection time")

    db_url = f"sqlite:///{(tmp_path / 'webui.db').resolve().as_posix()}"
    module = f"open_webui.routers.{router}"
    rc, stdout, stderr = run_fresh_import(open_webui_backend, db_url, module, tmp_path)

    if rc != 0:
        pytest.fail(
            f"import {module} failed from a cold interpreter.\n"
            f"--- stderr (tail) ---\n{stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{stdout[-1000:]}"
        )
    assert "OK" in stdout, stdout
