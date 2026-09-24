"""Guard: every router module must be mounted, exactly once, on its own prefix.

A router is only reachable because main.py imports it and hands it to `app.include_router`.
Nothing connects the two automatically, so a new router that never gets its `include_router`
line is a file full of endpoints that all answer 404, and a copy-pasted line that mounts two
routers on one prefix shadows whichever FastAPI matches second. Both fail as a missing feature
at runtime, never as an error at startup.

An `ast` audit of main.py; nothing is imported. The name a mount uses is resolved through
main.py's own imports, so aliases and formatting do not matter, and a commented-out mount does
not count.

Discriminates: passes on dev bbfa876af; in a copy of it, deleting one `include_router` line
fails the first test and mounting a second router on an existing prefix fails the third.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROUTERS_PACKAGE = "open_webui.routers"


def _router_modules(backend: Path) -> set[str]:
    directory = backend / "open_webui" / "routers"
    modules = {path.stem for path in directory.glob("*.py") if path.stem != "__init__"}
    if not modules:
        pytest.fail(f"no router modules under {directory}; retarget ROUTERS_PACKAGE")
    return modules


def _imported_routers(tree: ast.Module) -> dict[str, str]:
    """Local name -> router module, for every `from open_webui.routers import x as y`."""
    return {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == ROUTERS_PACKAGE
        for alias in node.names
    }


def _prefix(mount: ast.Call) -> str:
    prefix = next((keyword.value for keyword in mount.keywords if keyword.arg == "prefix"), None)
    if prefix is None:
        return ""
    return prefix.value if isinstance(prefix, ast.Constant) else ast.unparse(prefix)


def _mounts(backend: Path) -> list[tuple[str, str]]:
    """(mounted name, prefix) for every `include_router(<name>.router, prefix=...)`."""
    tree = ast.parse((backend / "open_webui" / "main.py").read_text(encoding="utf-8"))
    local_names = _imported_routers(tree)
    mounts = []
    for node in ast.walk(tree):
        is_mount = isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "include_router"
        router = node.args[0] if is_mount and node.args else None
        if not (isinstance(router, ast.Attribute) and isinstance(router.value, ast.Name)):
            continue
        mounts.append((local_names.get(router.value.id, router.value.id), _prefix(node)))
    if not mounts:
        pytest.fail("main.py contains no include_router calls; the audit needs retargeting")
    return mounts


@pytest.fixture(scope="module")
def mounted(open_webui_backend: Path) -> list[tuple[str, str]]:
    return _mounts(open_webui_backend)


def test_every_router_module_is_mounted(open_webui_backend: Path, mounted) -> None:
    """An unmounted router is a whole feature returning 404 with no error anywhere."""
    unmounted = _router_modules(open_webui_backend) - {name for name, _ in mounted}
    assert not unmounted, f"router modules never passed to include_router: {sorted(unmounted)}"


def test_no_router_is_mounted_twice(mounted) -> None:
    names = [name for name, _ in mounted]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"routers mounted more than once: {duplicates}"


def test_every_prefix_is_unique(mounted) -> None:
    """Two routers on one prefix leave the second one's colliding endpoints unreachable."""
    prefixes = [prefix for _, prefix in mounted]
    collisions = {
        prefix: sorted(name for name, other in mounted if other == prefix)
        for prefix in prefixes
        if prefixes.count(prefix) > 1
    }
    assert not collisions, f"routers sharing a prefix: {collisions}"


def test_main_mounts_nothing_that_is_not_a_router_module(open_webui_backend: Path, mounted):
    """A mount left behind after a router was renamed or deleted breaks main.py's import."""
    unknown = {name for name, _ in mounted} - _router_modules(open_webui_backend)
    assert not unknown, f"include_router names with no module under routers/: {sorted(unknown)}"
