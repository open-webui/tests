"""Guard: the declared route table must not contain shadowed or malformed paths.

FastAPI matches routes in declaration order and accepts a second registration
of a path it already has, so a copy-pasted decorator makes the later endpoint
unreachable without a warning at startup or an error in the logs. The same
goes for a path that forgets its leading slash: it still registers, it just
glues onto the router prefix as a different URL than the one written.

Read with ast, so this runs over all 31 routers in well under a second and
needs none of the backend's dependencies.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

ROUTERS_DIR = Path("open_webui") / "routers"
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
PATH_PARAMETER = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


def _routes(source: str) -> list[tuple[str, str, str]]:
    """Every (method, path, handler) a module declares with @router.<method>."""
    declared = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in HTTP_METHODS:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            declared.append((decorator.func.attr, decorator.args[0].value, node.name))
    return declared


@pytest.fixture(scope="module")
def route_table(open_webui_backend: Path) -> dict[str, list[tuple[str, str, str]]]:
    directory = open_webui_backend / ROUTERS_DIR
    if not directory.is_dir():
        pytest.skip(f"no routers directory at {directory}")
    table = {
        path.name: _routes(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.py"))
        if path.stem != "__init__"
    }
    assert any(routes for routes in table.values()), f"no routes found under {directory}"
    return table


def test_the_route_table_is_not_empty(route_table: dict) -> None:
    """A guard on the guard: a decorator style this parser cannot read would
    silently empty the table and pass everything below."""
    total = sum(len(routes) for routes in route_table.values())
    assert total > 100, f"only {total} routes parsed across {len(route_table)} routers"


def test_no_route_is_declared_twice_in_one_router(route_table: dict) -> None:
    """The second registration of a method and path never runs."""
    shadowed = {}
    for module, routes in route_table.items():
        counts = Counter((method, path) for method, path, _ in routes)
        collisions = {key for key, count in counts.items() if count > 1}
        if collisions:
            shadowed[module] = sorted(
                (method, path, handler)
                for method, path, handler in routes
                if (method, path) in collisions
            )
    assert not shadowed, f"routes registered more than once: {shadowed}"


def test_every_route_path_starts_with_a_slash(route_table: dict) -> None:
    """Without it the path is concatenated straight onto the router prefix and
    the endpoint answers on a URL nobody meant to publish."""
    malformed = [
        (module, method, path)
        for module, routes in route_table.items()
        for method, path, _ in routes
        if not path.startswith("/")
    ]
    assert not malformed, f"route paths with no leading slash: {malformed}"


def test_no_path_repeats_a_parameter_name(route_table: dict) -> None:
    """FastAPI cannot bind two path segments to one argument; the second value
    silently wins."""
    repeated = []
    for module, routes in route_table.items():
        for method, path, handler in routes:
            names = [name.strip() for name in PATH_PARAMETER.findall(path)]
            if len(names) != len(set(names)):
                repeated.append((module, method, path, handler))
    assert not repeated, f"route paths using one parameter name twice: {repeated}"
