"""Regression: the Ollama version route was readable without signing in.

open-webui 0.11.0 fix `41573d52f` (PR #27199): `get_ollama_versions` in
`routers/ollama.py` was the only Ollama route besides the static health check
that carried no authentication dependency. An anonymous caller could read the
configured backend's version string, and because `/api/version/{url_idx}`
indexes `ollama.base_urls` directly, they could also walk `url_idx` until the
lookup failed and learn how many backends were configured. The fix adds the same
`get_verified_user` dependency the sibling routes already carry.

Discriminates: passes on v0.11.0, fails on v0.10.2 (both version routes resolve
with no auth dependency at all).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.routing import APIRoute

pytestmark = pytest.mark.regression

AUTH_DEPENDENCIES = {"get_current_user", "get_verified_user", "get_admin_user"}
PUBLIC_ENDPOINTS = {"get_status"}  # static health check, returns no configuration


@pytest.fixture(scope="session")
def ollama_router(owui_module):
    return owui_module("open_webui.routers.ollama")


def _dependency_names(route: APIRoute) -> set[str]:
    names = set()
    pending = list(route.dependant.dependencies)
    while pending:
        dependency = pending.pop()
        # security schemes are class instances, so __name__ is not guaranteed
        name = getattr(dependency.call, "__name__", None)
        if name:
            names.add(name)
        pending.extend(dependency.dependencies)
    return names


def _routes(ollama_router, path: str | None = None) -> list[APIRoute]:
    routes = [r for r in ollama_router.router.routes if isinstance(r, APIRoute)]
    return [r for r in routes if path is None or r.path == path]


def _config_get(values: dict) -> AsyncMock:
    return AsyncMock(side_effect=lambda key, default=None: values.get(key, default))


# --- narrow: the version routes ---------------------------------------------


@pytest.mark.parametrize("path", ["/api/version", "/api/version/{url_idx}"])
def test_version_route_requires_an_authenticated_user(ollama_router, path):
    routes = _routes(ollama_router, path)
    assert routes, f"{path} is no longer registered, so this test proves nothing"
    for route in routes:
        assert _dependency_names(route) & AUTH_DEPENDENCIES, (
            f"{path} resolves without authentication, so anyone who can reach the server "
            "learns which Ollama version it talks to (#27199)"
        )


def test_backend_count_is_not_derivable_anonymously(ollama_router):
    """`url_idx` indexes `ollama.base_urls`, so an open route leaks the count."""
    route = _routes(ollama_router, "/api/version/{url_idx}")[0]
    assert "url_idx" in {p.name for p in route.dependant.path_params}
    assert _dependency_names(route) & AUTH_DEPENDENCIES, (
        "an anonymous caller could raise url_idx until the lookup failed and count the "
        "configured Ollama backends (#27199)"
    )


# --- broad: no Ollama route may be anonymous --------------------------------


def test_no_ollama_route_is_unauthenticated(ollama_router):
    unauthenticated = sorted(
        f"{sorted(route.methods)} {route.path}"
        for route in _routes(ollama_router)
        if route.endpoint.__name__ not in PUBLIC_ENDPOINTS
        and not _dependency_names(route) & AUTH_DEPENDENCIES
    )
    assert unauthenticated == [], (
        f"these Ollama routes serve anonymous callers: {unauthenticated}. Every route "
        "except the static health check must require a signed-in user (#27199)"
    )


# --- nearby: behaviour that must hold on both refs --------------------------


def test_health_check_stays_public(ollama_router):
    """It returns a constant, so gating it would break liveness probes."""
    for route in _routes(ollama_router, "/"):
        assert not _dependency_names(route) & AUTH_DEPENDENCIES


@pytest.mark.asyncio
async def test_authenticated_user_reads_the_lowest_version(ollama_router):
    config = _config_get(
        {
            "ollama.enable": True,
            "ollama.base_urls": ["http://a:11434", "http://b:11434"],
            "ollama.api_configs": {},
        }
    )
    versions = iter([{"version": "0.5.7"}, {"version": "0.4.9"}])
    with (
        patch.object(ollama_router.Config, "get", config),
        patch.object(
            ollama_router,
            "send_get_request",
            AsyncMock(side_effect=lambda *a, **kw: next(versions)),
        ),
    ):
        result = await ollama_router.get_ollama_versions(request=None)
    assert result == {"version": "0.4.9"}


@pytest.mark.asyncio
async def test_authenticated_user_reads_a_single_backend_version(ollama_router):
    config = _config_get({"ollama.enable": True, "ollama.base_urls": ["http://a:11434"]})
    with (
        patch.object(ollama_router.Config, "get", config),
        patch.object(ollama_router, "send_request", AsyncMock(return_value={"version": "0.5.7"})),
    ):
        result = await ollama_router.get_ollama_versions(request=None, url_idx=0)
    assert result == {"version": "0.5.7"}


@pytest.mark.asyncio
async def test_disabled_ollama_reports_no_version(ollama_router):
    with patch.object(ollama_router.Config, "get", _config_get({"ollama.enable": False})):
        assert await ollama_router.get_ollama_versions(request=None) == {"version": False}


@pytest.mark.asyncio
async def test_no_reachable_backend_raises(ollama_router):
    from fastapi import HTTPException

    config = _config_get(
        {"ollama.enable": True, "ollama.base_urls": ["http://a:11434"], "ollama.api_configs": {}}
    )
    with (
        patch.object(ollama_router.Config, "get", config),
        patch.object(ollama_router, "send_get_request", AsyncMock(return_value=None)),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await ollama_router.get_ollama_versions(request=None)
    assert excinfo.value.status_code == 500
