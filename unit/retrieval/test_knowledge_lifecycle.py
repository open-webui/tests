"""Guard: a processing-status stream must not hold a request-scoped database session.

open-webui PR #28183 (ba0c4b393): `GET /api/v1/files/{id}/process/status` and
`GET /api/v1/knowledge/{id}/files/pending` took `Depends(get_async_session)`. FastAPI releases
such a session only when the response body is finished, and with `stream=true` these bodies
run for up to two hours, so every watcher pinned a pooled connection. The fix drops the
dependency; each query opens a short session of its own. No HTTP response shows a pinned pool
connection, so this reads each route's resolved dependency tree.

The rest of this file moved to integration/retrieval/test_knowledge_lifecycle.py.

Discriminates: passes on dev bbfa876af; declaring `db = Depends(get_async_session)` on either
stream handler again fails the stream test.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

STREAMING_ROUTES = [
    ("open_webui.routers.files", "/{id}/process/status"),
    ("open_webui.routers.knowledge", "/{id}/files/pending"),
]
SHORT_LIVED_ROUTE = ("open_webui.routers.files", "/{id}/data/content")


@pytest.fixture(scope="module")
def get_async_session(owui_module):
    return owui_module("open_webui.internal.db").get_async_session


@pytest.fixture(scope="module")
def dependencies_of(owui_module):
    """`dependencies_of(module, path)`: every callable the GET route at `path` depends on."""

    def walk(dependant):
        for dependency in dependant.dependencies:
            yield dependency.call
            yield from walk(dependency)

    def resolve(module_name: str, path: str) -> set:
        router = owui_module(module_name).router
        routes = [route for route in router.routes if route.path == path and "GET" in route.methods]
        assert routes, f"no GET {path} in {module_name}; retarget this guard"
        return set(walk(routes[0].dependant))

    return resolve


@pytest.mark.parametrize(("module_name", "path"), STREAMING_ROUTES)
def test_a_status_stream_holds_no_request_session(
    dependencies_of, get_async_session, module_name, path
):
    assert get_async_session not in dependencies_of(module_name, path), (
        f"GET {path} pins a pooled connection for its whole stream (PR #28183)"
    )


def test_a_short_lived_route_still_resolves_its_session(dependencies_of, get_async_session):
    """Guards the guard: the walk does see a session dependency where one is declared."""
    assert get_async_session in dependencies_of(*SHORT_LIVED_ROUTE)
