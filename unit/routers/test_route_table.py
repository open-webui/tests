"""Guard: the declared route table must not contain shadowed or malformed paths.

FastAPI matches routes in declaration order and accepts a second registration of a method and
path it already has, so a copy-pasted decorator makes the later endpoint unreachable without a
warning at startup. A path without its leading slash still registers, glued onto the router
prefix as a URL nobody wrote, and a path naming one parameter twice binds only one of them.

The live OpenAPI schema cannot see the main case: `get_openapi` merges a repeated method and
path into one operation. So this reads every router and `main.py` with `ast`, which also needs
none of the backend's dependencies.

Discriminates: passes on dev bbfa876af; a second `@router.get` for a path its module already
declares, a path without a leading slash or a path repeating a parameter name each fail their
test.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
PATH_PARAMETER = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


def declared_path(decorator: ast.expr) -> str | None:
    """The path of `@<router>.<method>(path, ...)` or `(path=...)`, else None."""
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    if decorator.func.attr not in HTTP_METHODS:
        return None
    arguments = [*decorator.args[:1], *(k.value for k in decorator.keywords if k.arg == "path")]
    constants = [argument.value for argument in arguments if isinstance(argument, ast.Constant)]
    return constants[0] if constants else None


def routes_of(source: str) -> list[tuple[str, str, str]]:
    """Every (method, path, handler) a module declares."""
    return [
        (decorator.func.attr, path, node.name)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if (path := declared_path(decorator)) is not None
    ]


@pytest.fixture(scope="module")
def route_table(open_webui_backend: Path) -> dict[str, list[tuple[str, str, str]]]:
    package = open_webui_backend / "open_webui"
    modules = [*sorted((package / "routers").glob("*.py")), package / "main.py"]
    missing = [module for module in modules if not module.is_file()]
    assert len(modules) > 2 and not missing, f"retarget this guard: no routers under {package}"
    return {module.name: routes_of(module.read_text(encoding="utf-8")) for module in modules}


def test_the_route_table_is_not_empty(route_table):
    """A decorator style this parser cannot read would empty the table and pass everything."""
    total = sum(len(routes) for routes in route_table.values())
    assert total > 300, f"only {total} routes parsed across {len(route_table)} modules"
    assert route_table["main.py"], "no routes parsed from main.py"


def test_no_route_is_declared_twice_in_one_module(route_table):
    shadowed = {}
    for module, routes in route_table.items():
        counts = Counter((method, path) for method, path, _ in routes)
        repeated = sorted(route for route in routes if counts[route[:2]] > 1)
        if repeated:
            shadowed[module] = repeated
    assert not shadowed, f"routes registered more than once: {shadowed}"


def test_every_route_path_starts_with_a_slash(route_table):
    malformed = [
        (module, method, path)
        for module, routes in route_table.items()
        for method, path, _ in routes
        if not path.startswith("/")
    ]
    assert not malformed, f"route paths with no leading slash: {malformed}"


def test_no_path_repeats_a_parameter_name(route_table):
    repeated = [
        (module, method, path)
        for module, routes in route_table.items()
        for method, path, _ in routes
        if len(names := PATH_PARAMETER.findall(path)) != len(set(names))
    ]
    assert not repeated, f"route paths using one parameter name twice: {repeated}"
