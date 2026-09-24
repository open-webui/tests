"""Regression: the per-connection model listings must check the caller's role.

open-webui 0.11.4 PR #29619 (`4b1019009`): the endpoints that list the models or version of a
single Ollama or OpenAI connection (`/openai/models/{idx}`, `/ollama/api/tags/{idx}`,
`/ollama/api/version/{idx}`, `/ollama/v1/models/{idx}`) must refuse every non-admin with a 401
before the connection is contacted, while the aggregate listings stay open to everyone. The
commit moved the check from a route dependency into each handler.

Twin of unit/security/test_connection_listing_roles.py.

Discriminates: passes on dev bbfa876af; with the role check removed from the handlers (and no
route dependency) a user gets HTTP 200 on all four and the Ollama listings are fetched from the
connection on their behalf. Reverting 4b1019009 alone keeps them green: its route-level
`get_admin_user` refused users over HTTP too.
"""

from __future__ import annotations

import pytest

from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")

SINGLE_CONNECTION_LISTINGS = {
    "/openai/models/0": None,
    "/ollama/api/tags/0": "/api/tags",
    "/ollama/api/version/0": "/api/version",
    "/ollama/v1/models/0": "/api/tags",
}


@pytest.fixture
def ollama_connection(admin, preserve, listener):
    """Ollama switched on with one connection, served by the listener."""
    listener.route(
        "GET",
        "/api/tags",
        json_answer({"models": [{"name": "private-model", "model": "private-model"}]}),
    )
    listener.route("GET", "/api/version", json_answer({"version": "0.9.1"}))
    preserve(OLLAMA_CONFIG)
    with admin.client() as client:
        current = client.get("/ollama/config").json()
        enabled = {
            **current,
            "ENABLE_OLLAMA_API": True,
            "OLLAMA_BASE_URLS": [listener.base_url],
            "OLLAMA_API_CONFIGS": {},
        }
        client.post("/ollama/config/update", json=enabled).raise_for_status()
    return listener


@pytest.mark.parametrize("path", SINGLE_CONNECTION_LISTINGS)
def test_a_user_is_refused_every_single_connection_listing(path, make_user, ollama_connection):
    connection_path = SINGLE_CONNECTION_LISTINGS[path]
    with make_user().client() as client:
        response = client.get(path)

    assert response.status_code == 401, (
        f"a non-admin got HTTP {response.status_code} from {path}, reading a single "
        f"connection's models or version (#29619): {response.text[:200]}"
    )
    if connection_path:
        assert ollama_connection.requests_to(connection_path) == [], (
            f"{path} contacted the connection on a non-admin's behalf (#29619)"
        )


@pytest.mark.parametrize("path", SINGLE_CONNECTION_LISTINGS)
def test_an_admin_reaches_every_single_connection_listing(path, admin, ollama_connection):
    with admin.client() as client:
        response = client.get(path)

    assert response.status_code == 200, f"{path}: HTTP {response.status_code} {response.text}"


@pytest.mark.parametrize("path", ["/openai/models", "/ollama/api/tags", "/ollama/v1/models"])
def test_the_aggregate_listings_stay_open_to_users(path, make_user, ollama_connection):
    with make_user().client() as client:
        response = client.get(path)

    assert response.status_code == 200, f"{path}: HTTP {response.status_code} {response.text}"
