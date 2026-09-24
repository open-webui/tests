"""Regression: the Ollama version route answered callers who had not signed in.

open-webui 0.11.0, fix `41573d52f` (PR #27199): `get_ollama_versions` was the only Ollama route
besides the health check without an authentication dependency. An anonymous caller read the
backend's version, and by walking `/api/version/{url_idx}` until the lookup failed, learned how
many backends were configured. The fix adds `get_verified_user`. Here every `/ollama` route the
instance publishes in its OpenAPI schema is called without a token.

Twin of unit/security/test_ollama_version_requires_auth.py.

Discriminates: passes on bbfa876af, fails with the `get_verified_user` dependency removed (with the
admin-only `url_idx` check 4b1019009 later built on it): both anonymous version calls answer 200
and the sweep lists them; the rest passes on both.
"""

from __future__ import annotations

import re

import httpx
import pytest

from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")
PUBLIC_ROUTES = {("GET", "/ollama/"), ("HEAD", "/ollama/")}  # the static health check


def _anonymous(instance, method: str, path: str) -> httpx.Response:
    return httpx.request(method, f"{instance.base_url}{path}", timeout=30.0)


def _ollama_routes(instance) -> list[tuple[str, str]]:
    schema = httpx.get(f"{instance.base_url}/openapi.json", timeout=30.0)
    assert schema.status_code == 200, "the instance publishes no OpenAPI schema to sweep"
    return sorted(
        (method.upper(), path)
        for path, operations in schema.json()["paths"].items()
        if path.startswith("/ollama/")
        for method in operations
    )


@pytest.fixture
def ollama_backends(admin, preserve, listener):
    """`ollama_backends("a", "b")` points Ollama at listener paths; only a and b answer."""
    preserve(OLLAMA_CONFIG)
    listener.route("GET", "/a/api/version", json_answer({"version": "0.5.7"}))
    listener.route("GET", "/b/api/version", json_answer({"version": "0.4.9"}))

    def configure(*names: str) -> None:
        urls = [f"{listener.base_url}/{name}" for name in names]
        config = {"ENABLE_OLLAMA_API": True, "OLLAMA_BASE_URLS": urls, "OLLAMA_API_CONFIGS": {}}
        with admin.client() as client:
            assert client.post(OLLAMA_CONFIG[1], json=config).status_code == 200

    return configure


# narrow: the version routes


@pytest.mark.parametrize("path", ["/ollama/api/version", "/ollama/api/version/0"])
def test_version_route_refuses_an_anonymous_caller(instance, path):
    answer = _anonymous(instance, "GET", path)

    assert answer.status_code == 401, (
        f"GET {path} answered an anonymous caller with HTTP {answer.status_code}: {answer.text} "
        "(#27199)"
    )


# broad: no Ollama route but the health check serves anonymous callers


def test_no_ollama_route_answers_an_anonymous_caller(instance):
    routes = [route for route in _ollama_routes(instance) if route not in PUBLIC_ROUTES]
    assert len(routes) > 20, f"the schema lists too few Ollama routes to sweep: {routes}"

    served = []
    for method, path in routes:
        answer = _anonymous(instance, method, re.sub(r"\{[^}]+\}", "0", path))
        if answer.status_code != 401:
            served.append(f"{method} {path} -> {answer.status_code}")

    assert served == [], f"Ollama routes that serve anonymous callers (#27199): {served}"


# nearby: the health check stays public, signed-in callers still get versions


def test_health_check_stays_public(instance):
    answer = _anonymous(instance, "GET", "/ollama/")

    assert answer.status_code == 200
    assert answer.json() == {"status": True}


def test_signed_in_user_reads_the_lowest_backend_version(ollama_backends, user):
    ollama_backends("a", "b")
    with user.client() as client:
        answer = client.get("/ollama/api/version")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"version": "0.4.9"}


def test_admin_reads_one_backend_version_by_index(ollama_backends, admin):
    ollama_backends("a", "b")
    with admin.client() as client:
        answer = client.get("/ollama/api/version/0")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"version": "0.5.7"}


def test_no_answering_backend_is_a_server_error(ollama_backends, user):
    ollama_backends("silent")
    with user.client() as client:
        answer = client.get("/ollama/api/version")

    assert answer.status_code == 500, answer.text


def test_signed_in_user_without_ollama_gets_no_version(user):
    with user.client() as client:
        answer = client.get("/ollama/api/version")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"version": False}
