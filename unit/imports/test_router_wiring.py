"""Guard: every router module must be mounted, exactly once, on its own prefix.

A router is only reachable because main.py imports it and hands it to
app.include_router. Nothing connects the two automatically, so a new router
that never gets its include_router line is a file full of endpoints that all
answer 404, and a copy-pasted line that mounts two routers on one prefix
shadows whichever FastAPI matches second — both fail as a missing feature at
runtime rather than as an error at startup.

A source audit of main.py; nothing is imported.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROUTERS_DIR = Path("open_webui") / "routers"
INCLUDE = re.compile(r"include_router\(\s*(\w+)\.router\s*,\s*prefix=['\"]([^'\"]+)['\"]")


@pytest.fixture(scope="module")
def main_source(open_webui_backend: Path) -> str:
    path = open_webui_backend / "open_webui" / "main.py"
    if not path.is_file():
        pytest.skip(f"no main.py at {path}")
    # A commented-out mount is exactly the accident this file is looking for,
    # so comment lines must not count as wiring.
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def router_modules(open_webui_backend: Path) -> list[str]:
    directory = open_webui_backend / ROUTERS_DIR
    if not directory.is_dir():
        pytest.skip(f"no routers directory at {directory}")
    modules = sorted(p.stem for p in directory.glob("*.py") if p.stem != "__init__")
    assert modules, f"no router modules under {directory}"
    return modules


@pytest.fixture(scope="module")
def mounted(main_source: str) -> list[tuple[str, str]]:
    found = INCLUDE.findall(main_source)
    assert found, "main.py contains no include_router calls"
    return found


def test_every_router_module_is_mounted(
    router_modules: list[str], mounted: list[tuple[str, str]]
) -> None:
    """An unmounted router is a whole feature returning 404 with no error
    anywhere in the logs."""
    mounted_names = {name for name, _ in mounted}
    unmounted = [module for module in router_modules if module not in mounted_names]
    assert not unmounted, f"router modules never passed to include_router: {unmounted}"


def test_no_router_is_mounted_twice(mounted: list[tuple[str, str]]) -> None:
    names = [name for name, _ in mounted]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"routers mounted more than once: {duplicates}"


def test_every_prefix_is_unique(mounted: list[tuple[str, str]]) -> None:
    """Two routers on one prefix means the second one's endpoints are
    unreachable wherever the paths collide."""
    prefixes = [prefix for _, prefix in mounted]
    collisions = {
        prefix: sorted(name for name, other in mounted if other == prefix)
        for prefix in prefixes
        if prefixes.count(prefix) > 1
    }
    assert not collisions, f"routers sharing a prefix: {collisions}"


def test_main_mounts_nothing_that_is_not_a_router_module(
    router_modules: list[str], mounted: list[tuple[str, str]]
) -> None:
    """A mount left behind after a router was renamed or deleted breaks the
    import of main.py outright."""
    unknown = sorted({name for name, _ in mounted if name not in router_modules})
    assert not unknown, f"include_router names with no module under routers/: {unknown}"
