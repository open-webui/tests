"""Journey: every endpoint a router module declares is served, once, by that module's handler.

A router is only reachable because main.py imports it and mounts it with `include_router`.
Nothing connects the two automatically, so a router that never gets its mount line is a file of
endpoints that all answer 404, a router mounted a second time on another prefix serves its
endpoints twice, and two routers mounted on one prefix leave the second one's colliding
endpoints unreachable. None of that fails at startup.

Every `@router.<method>(path)` in every file under `routers/`, router packages such as `audio/`
included, is read from the source and looked up in the live `/openapi.json`. A module counts as
mounted under a prefix when every one of its endpoints is served at that prefix plus its path,
with the operation id FastAPI derives from the module's own handler; each module needs exactly
one such prefix. The instance runs with SCIM on, so the one router main.py mounts behind a
switch is served too.

Twin of unit/imports/test_router_wiring.py, which keeps the audit of main.py's mount lines for
what no request sees: a duplicate mount on the same prefix, and two routers sharing a prefix
before any of their paths collide.
Discriminates: in a backend copy, deleting the `include_router` line for `notifications` fails
`test_every_router_module_is_served`; mounting `utils` a second time under `/api/v1/utils2`
fails `test_no_router_module_is_served_twice`; mounting `automations` on the `/api/v1/notes`
prefix fails `test_every_router_module_is_served` for `notes` (`/create` and `/{id}` collide);
deleting the mount of the `audio` package fails it for `audio`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from harness.instance import resolve_backend
from harness.scim import SCIM_ENV

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PATH_CONVERTER = re.compile(r"\{([^}:]+):[^}]+\}")


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    return {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _methods_of(decorator: ast.Call, constants: dict[str, ast.expr]) -> list[str]:
    """`@router.get(...)` gives GET; `@router.api_route(..., methods=...)` its list."""
    if decorator.func.attr in HTTP_METHODS:
        return [decorator.func.attr.upper()]
    if decorator.func.attr != "api_route":
        return []
    methods = next(keyword.value for keyword in decorator.keywords if keyword.arg == "methods")
    if isinstance(methods, ast.Name):
        methods = constants[methods.id]
    # HEAD and OPTIONS reach the same handler and are left out of the schema
    return [
        method.upper() for method in ast.literal_eval(methods) if method.lower() in HTTP_METHODS
    ]


def _declared_endpoints(source: str) -> list[tuple[str, str, str]]:
    """(METHOD, path, handler name) for every route decorator in one router module."""
    tree = ast.parse(source)
    constants = _module_constants(tree)
    endpoints = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            path = decorator.args[0] if decorator.args else None
            if not isinstance(path, ast.Constant):
                continue
            for method in _methods_of(decorator, constants):
                endpoints.append((method, PATH_CONVERTER.sub(r"{\1}", path.value), node.name))
    return endpoints


def _module_name(routers: Path, path: Path) -> str:
    """`audio` for `audio/__init__.py`, `audio.realtime` for `audio/realtime.py`."""
    parts = path.relative_to(routers).with_suffix("").parts
    return ".".join(part for part in parts if part != "__init__")


@pytest.fixture(scope="module")
def declared() -> dict[str, list[tuple[str, str, str]]]:
    """`{router module: its endpoints}` for every file under `routers/` and its packages."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    routers = Path(backend) / "open_webui" / "routers"
    modules = {
        _module_name(routers, path): _declared_endpoints(path.read_text(encoding="utf-8"))
        for path in sorted([*routers.glob("*.py"), *routers.glob("*/*.py")])
        if path.parent != routers or path.stem != "__init__"
    }
    assert len(modules) > 20, f"retarget this sweep: only {len(modules)} router modules read"
    return modules


@pytest.fixture(scope="module")
def served(instance_with) -> dict[tuple[str, str], str]:
    """`{(METHOD, path): operation id}` from the live schema of an instance with SCIM on."""
    with instance_with(SCIM_ENV).client() as client:
        schema = client.get("/openapi.json")
    assert schema.status_code == 200, f"no /openapi.json to read: HTTP {schema.status_code}"
    return {
        (method.upper(), path): operation["operationId"]
        for path, operations in schema.json()["paths"].items()
        for method, operation in operations.items()
        if method in HTTP_METHODS
    }


def _operation_id(handler: str, served_path: str) -> str:
    """The prefix of the id FastAPI gives an operation, method suffix left off."""
    return re.sub(r"\W", "_", handler + served_path) + "_"


def _mount_prefixes(endpoints: list, served: dict[tuple[str, str], str]) -> set[str]:
    """Every prefix under which all of a module's endpoints are served by its own handlers."""
    candidates = {
        served_path[: len(served_path) - len(path)]
        for method, path, _ in endpoints
        for served_method, served_path in served
        if served_method == method and served_path.endswith(path)
    }
    return {
        prefix
        for prefix in candidates
        if all(
            served.get((method, prefix + path), "").startswith(
                _operation_id(handler, prefix + path)
            )
            for method, path, handler in endpoints
        )
    }


def test_every_router_module_is_served(declared, served):
    unserved = [
        module
        for module, endpoints in declared.items()
        if endpoints and not _mount_prefixes(endpoints, served)
    ]
    assert not unserved, f"router modules with no prefix serving all their endpoints: {unserved}"


def test_no_router_module_is_served_twice(declared, served):
    doubled = {
        module: sorted(prefixes)
        for module, endpoints in declared.items()
        if len(prefixes := _mount_prefixes(endpoints, served)) > 1
    }
    assert not doubled, f"router modules served under more than one prefix: {doubled}"


def test_the_sweep_reads_the_route_table(declared, served):
    """A parser that lost the decorators would pass both checks above."""
    endpoint_count = sum(len(endpoints) for endpoints in declared.values())
    assert endpoint_count > 400, endpoint_count
    assert len(served) > 400, len(served)
