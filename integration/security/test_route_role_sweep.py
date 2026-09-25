"""Journey: every route refuses the accounts below the role its source asks for.

Each route in the live `/openapi.json` gets its required role from the backend source: the
route's `Depends(...)` chain, read with `ast` and followed through dependency functions its
module defines. A plain user is then refused on every admin-only route, and a pending account
and a signed-out client on every route that wants a verified user. The request carries a body
that fails validation, so a route that skipped its check answers 422 or better, never 401.

A route the classifier cannot read, and a route with no auth dependency at all, fail unless
the short tables below name them with a reason, and must still refuse a signed-out client.

Discriminates: in a backend copy, removing `user=Depends(get_admin_user)` from
`/api/v1/configs/export` turns `test_every_route_has_a_known_role` (the route reads as public)
and `test_a_signed_out_client_is_refused_every_signed_in_route` (HTTP 200) red; swapping it for
a module-local dependency that checks nothing turns the same two red (no role the sweep knows);
letting `get_admin_user` pass a user and `get_verified_user` pass a pending account turns the
user and pending sweeps red.
"""

from __future__ import annotations

import ast
import re

import httpx
import pytest

from harness.instance import resolve_backend

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ROLE_OF_DEPENDENCY = {
    "get_admin_user": "admin",
    "get_verified_user": "verified",
    "get_current_user": "signed_in",
}
ROLE_STRENGTH = ["public", "signed_in", "verified", "admin"]
# dependencies that neither grant nor check a role
NEUTRAL_DEPENDENCIES = {"get_async_session", "get_session"}
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PATH_CONVERTER = re.compile(r"\{([^}:]+):[^}]+\}")
# a JSON string where handlers want an object, a form or nothing: validation fails if reached
INVALID_BODY = b'"not the body this route wants"'
REFUSED = {401, 403}

# Routes that need no account, by (method, path): why.
PUBLIC_ROUTES = {
    ("GET", "/health"): "the load balancer's probe",
    ("GET", "/health/db"): "the load balancer's probe",
    ("GET", "/ready"): "the load balancer's probe",
    ("GET", "/ollama/"): "a static status probe",
    ("GET", "/api/version"): "the sign-in page reads it",
    ("GET", "/api/changelog"): "the sign-in page reads it",
    ("GET", "/api/config"): "the sign-in page reads it; it adds more for a valid token",
    ("GET", "/manifest.json"): "the browser fetches it before sign-in",
    ("GET", "/opensearch.xml"): "the browser fetches it before sign-in",
    ("POST", "/api/v1/auths/signin"): "signs in",
    ("POST", "/api/v1/auths/signup"): "signs up, gated by the signup setting",
    ("POST", "/api/v1/auths/ldap"): "signs in",
    ("POST", "/api/v1/auths/signout"): "signs out whatever cookie it gets",
    ("POST", "/api/v1/auths/oauth/{provider}/token/exchange"): "signs in, off by default",
    ("GET", "/oauth/{provider}/login"): "starts an OAuth sign-in",
    ("GET", "/oauth/{provider}/login/callback"): "the provider redirects here",
    ("GET", "/oauth/{provider}/callback"): "the provider redirects here",
    ("GET", "/oauth/clients/{client_id}/callback"): "a tool server's provider redirects here",
    ("POST", "/oauth/backchannel-logout"): "the provider calls it with a signed token",
    ("POST", "/api/v1/channels/webhooks/{webhook_id}/{token}"): "the path carries the secret",
    ("GET", "/api/v1/retrieval/ef/{text}"): "declared only with ENV=dev, like /openapi.json",
}

# Routes whose role the dependency chain does not tell, by (method, path): how they check.
CHECKED_IN_HANDLER = {
    ("GET", "/api/v1/chats/share/{share_id}"): "a share open to anyone needs no account",
}

# Router modules the shared instance does not mount: why.
UNMOUNTED_ROUTERS = {
    "scim": "ENABLE_SCIM is off on the shared instance",
}


def _dependency_name(default: ast.expr) -> str | None:
    """`name` for a `Depends(name)` default, else None."""
    if not isinstance(default, ast.Call):
        return None
    callee = default.func
    callee_name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
    if callee_name != "Depends" or not default.args:
        return None
    target = default.args[0]
    return target.id if isinstance(target, ast.Name) else ast.unparse(target)


def _dependencies_of(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = function.args
    defaults = [*arguments.defaults, *(d for d in arguments.kw_defaults if d is not None)]
    return [name for default in defaults if (name := _dependency_name(default))]


def _role_of(function, module_functions: dict, seen: frozenset = frozenset()) -> str | None:
    """The strongest role the function's dependencies demand, None when one cannot be read."""
    roles = []
    unreadable = False
    for name in _dependencies_of(function):
        if name in ROLE_OF_DEPENDENCY:
            roles.append(ROLE_OF_DEPENDENCY[name])
        elif name in NEUTRAL_DEPENDENCIES:
            continue
        elif name in module_functions and name not in seen:
            role = _role_of(module_functions[name], module_functions, seen | {name})
            # a dependency of the module's own that demands no known role may check anything
            if role in (None, "public"):
                unreadable = True
            else:
                roles.append(role)
        else:
            unreadable = True
    if roles:
        return max(roles, key=ROLE_STRENGTH.index)
    return None if unreadable else "public"


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    return {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _methods_of(decorator: ast.Call, constants: dict[str, ast.expr]) -> list[str]:
    """`@<router>.get(...)` gives GET; `@<router>.api_route(..., methods=...)` its list."""
    if decorator.func.attr in HTTP_METHODS:
        return [decorator.func.attr.upper()]
    if decorator.func.attr != "api_route":
        return []
    methods = next(k.value for k in decorator.keywords if k.arg == "methods")
    if isinstance(methods, ast.Name):
        methods = constants[methods.id]
    # HEAD and OPTIONS reach the same handler and are left out of the schema
    listed = ast.literal_eval(methods)
    return [method.upper() for method in listed if method.lower() in HTTP_METHODS]


def _declared_routes(tree: ast.Module):
    """Every (METHOD, path, handler) a module declares with a router or app decorator."""
    constants = _module_constants(tree)
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
                yield method, path.value, node


def _router_prefixes(main_tree: ast.Module) -> dict[str, str]:
    """`{module: prefix}` from every `app.include_router(<module>.router, prefix=...)`."""
    prefixes = {}
    for node in ast.walk(main_tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "include_router":
            continue
        router = node.args[0]
        prefix = next(k.value.value for k in node.keywords if k.arg == "prefix")
        prefixes[router.value.id] = prefix
    return prefixes


@pytest.fixture(scope="module")
def source_routes() -> dict[tuple[str, str], tuple[str, str | None]]:
    """`{(METHOD, path): (router module, role)}` for every route the source declares."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    package = backend / "open_webui"
    main_tree = ast.parse((package / "main.py").read_text(encoding="utf-8"))
    modules = {"main": ("", main_tree)}
    for module, prefix in _router_prefixes(main_tree).items():
        source = (package / "routers" / f"{module}.py").read_text(encoding="utf-8")
        modules[module] = (prefix, ast.parse(source))

    routes = {}
    for module, (prefix, tree) in modules.items():
        module_functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for method, path, handler in _declared_routes(tree):
            full_path = PATH_CONVERTER.sub(r"{\1}", prefix + path)
            routes[(method, full_path)] = (module, _role_of(handler, module_functions))
    assert len(routes) > 300, f"retarget this sweep: only {len(routes)} routes parsed"
    return routes


@pytest.fixture(scope="module")
def live_routes(instance) -> dict[tuple[str, str], dict]:
    """`{(METHOD, path): operation}` from the instance's own OpenAPI schema."""
    with instance.client() as client:
        schema = client.get("/openapi.json")
    assert schema.status_code == 200, f"no /openapi.json to sweep: HTTP {schema.status_code}"
    return {
        (method.upper(), path): operation
        for path, operations in schema.json()["paths"].items()
        for method, operation in operations.items()
        if method in HTTP_METHODS
    }


def _placeholder_path(path: str, operation: dict) -> str:
    """The path with every parameter filled; the role check runs before any lookup."""
    for parameter in operation.get("parameters", []):
        if parameter["in"] != "path":
            continue
        is_integer = parameter.get("schema", {}).get("type") == "integer"
        path = path.replace(f"{{{parameter['name']}}}", "0" if is_integer else "sweep-missing-id")
    return path


def _expected_role(route: tuple[str, str], source_routes) -> str:
    """The route's role; one no table excuses must at least want an account."""
    _, role = source_routes.get(route, (None, None))
    if route in PUBLIC_ROUTES or route in CHECKED_IN_HANDLER:
        return "public"
    return "signed_in" if role in (None, "public") else role


def _not_refused(client: httpx.Client, live_routes: dict, roles: set[str], source_routes) -> list:
    """Every route demanding one of `roles` that `client` was not refused on, with its status."""
    reached = []
    for (method, path), operation in sorted(live_routes.items()):
        if _expected_role((method, path), source_routes) not in roles:
            continue
        response = client.request(
            method,
            _placeholder_path(path, operation),
            content=INVALID_BODY,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code not in REFUSED:
            reached.append((method, path, response.status_code))
    return reached


def test_every_route_has_a_known_role(source_routes, live_routes):
    unknown = []
    for route in sorted(live_routes):
        if route not in source_routes:
            unknown.append((*route, "not declared anywhere the sweep reads"))
            continue
        _, role = source_routes[route]
        if role is None and route not in CHECKED_IN_HANDLER:
            unknown.append((*route, "its dependencies name no role the sweep knows"))
        if role == "public" and route not in PUBLIC_ROUTES:
            unknown.append((*route, "it has no auth dependency"))
    assert not unknown, f"routes with no role the sweep can check: {unknown}"


def test_every_declared_route_is_served(source_routes, live_routes):
    """A route the schema lacks would drop out of the sweep unseen."""
    unserved = [
        route
        for route, (module, _) in sorted(source_routes.items())
        if module not in UNMOUNTED_ROUTERS and route not in live_routes
    ]
    assert not unserved, f"declared routes missing from /openapi.json: {unserved}"


def test_the_exception_tables_name_live_routes(source_routes, live_routes):
    stale = [route for route in [*PUBLIC_ROUTES, *CHECKED_IN_HANDLER] if route not in live_routes]
    declared_modules = {module for module, _ in source_routes.values()}
    stale += [module for module in UNMOUNTED_ROUTERS if module not in declared_modules]
    assert not stale, f"exception entries naming nothing the instance serves: {stale}"


def test_a_user_is_refused_every_admin_route(source_routes, live_routes, make_user):
    with make_user().client() as client:
        reached = _not_refused(client, live_routes, {"admin"}, source_routes)

    assert not reached, f"admin-only routes a plain user was not refused on: {reached}"


def test_a_pending_account_is_refused_every_verified_route(source_routes, live_routes, make_user):
    with make_user(role="pending").client() as client:
        reached = _not_refused(client, live_routes, {"verified", "admin"}, source_routes)

    assert not reached, f"routes a pending account was not refused on: {reached}"


def test_a_signed_out_client_is_refused_every_signed_in_route(source_routes, live_routes, instance):
    with httpx.Client(base_url=instance.base_url, timeout=60.0) as client:
        reached = _not_refused(
            client, live_routes, {"signed_in", "verified", "admin"}, source_routes
        )

    assert not reached, f"routes a signed-out client was not refused on: {reached}"


def test_the_sweep_covers_the_route_table(source_routes, live_routes):
    """A classifier that lost the auth dependencies would pass every sweep above."""
    served_roles = [source_routes[route][1] for route in live_routes if route in source_routes]
    assert served_roles.count("admin") > 100, served_roles.count("admin")
    assert served_roles.count("verified") > 200, served_roles.count("verified")
