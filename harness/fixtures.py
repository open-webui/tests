"""Pytest fixtures over the harness, shared by the integration and browser suites.

`instance` is one scratch backend per session that the regression tests share; every test gets
a clean `upstream` script and can add accounts with `make_user`. Tests that change global
settings wrap them in `preserve`, which restores what it snapshotted. `instance_with(env)`
boots a second backend for settings that only exist as environment variables.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Generator, Iterator

import httpx
import pytest

from harness import upstream as upstream_module
from harness.actors import Actor, admin_of, create_user
from harness.instance import LaunchedInstance, free_port, launch
from harness.listener import Listener, listening
from harness.upstream import MOCK_MODEL_ID, MockUpstream

TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"

# Title, tag, follow-up and query tasks share the chat endpoint and would take a scripted reply
# meant for the chat; a test that needs one turns it on through /api/v1/tasks/config/update.
QUIET_TASKS = {
    "ENABLE_TITLE_GENERATION": "false",
    "ENABLE_TAGS_GENERATION": "false",
    "ENABLE_FOLLOW_UP_GENERATION": "false",
    "ENABLE_SEARCH_QUERY_GENERATION": "false",
    "ENABLE_RETRIEVAL_QUERY_GENERATION": "false",
}

# Global settings a test may change, by name: (read endpoint, write endpoint).
SETTINGS = {
    "permissions": ("/api/v1/users/default/permissions", "/api/v1/users/default/permissions"),
    "admin_config": ("/api/v1/auths/admin/config", "/api/v1/auths/admin/config"),
    "tasks": ("/api/v1/tasks/config", "/api/v1/tasks/config/update"),
}


@contextlib.contextmanager
def _serving_upstream() -> Iterator[MockUpstream]:
    upstream, shutdown = upstream_module.serve(free_port())
    try:
        yield upstream
    finally:
        shutdown()


def _publish_mock_model(instance: LaunchedInstance) -> None:
    """Make the provider's model readable by every account, like the admin's visibility toggle."""
    grant = {"principal_type": "user", "principal_id": "*", "permission": "read"}
    with instance.client() as client:
        client.get("/api/models").raise_for_status()  # registers the provider's models
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={"id": MOCK_MODEL_ID, "name": MOCK_MODEL_ID, "access_grants": [grant]},
        )
    assert shared.status_code == 200, f"publishing {MOCK_MODEL_ID} failed: {shared.text}"


def _launch_published(upstream: MockUpstream, env: dict[str, str]) -> Iterator[LaunchedInstance]:
    for launched in launch(upstream, {**QUIET_TASKS, **env}):
        _publish_mock_model(launched)
        yield launched


@pytest.fixture(scope="session")
def mock_upstream() -> Generator[MockUpstream, None, None]:
    """The provider behind `launched_instance` and `degraded_instance`."""
    with _serving_upstream() as upstream:
        yield upstream


@pytest.fixture(scope="session")
def launched_instance(mock_upstream: MockUpstream) -> Generator[LaunchedInstance, None, None]:
    """A scratch instance of its own for the footprint tests, which measure the process."""
    yield from launch(mock_upstream, {})


@pytest.fixture(scope="session")
def _shared_upstream() -> Generator[MockUpstream, None, None]:
    with _serving_upstream() as upstream:
        yield upstream


@pytest.fixture(scope="session")
def instance(_shared_upstream: MockUpstream) -> Generator[LaunchedInstance, None, None]:
    """The scratch instance the regression tests share."""
    yield from _launch_published(_shared_upstream, {})


@pytest.fixture
def upstream(instance: LaunchedInstance) -> MockUpstream:
    """The shared instance's provider, with an empty script and request log."""
    instance.upstream.reset()
    return instance.upstream


@pytest.fixture(scope="session")
def admin(instance: LaunchedInstance) -> Actor:
    return admin_of(instance)


@pytest.fixture(scope="session")
def user(instance: LaunchedInstance) -> Actor:
    """The regular account the browser suite signs in with."""
    return create_user(
        instance, name="Test User", email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD
    )


@pytest.fixture
def make_user(instance: LaunchedInstance) -> Callable[..., Actor]:
    """A factory for fresh accounts, so a test never inherits another test's state."""

    def factory(role: str = "user", **options) -> Actor:
        return create_user(instance, role=role, **options)

    return factory


@pytest.fixture
def preserve(admin: Actor) -> Generator[Callable[..., None], None, None]:
    """`preserve("permissions", ...)` snapshots those settings and restores them afterwards.

    A setting is a name from `SETTINGS` or a `(read endpoint, write endpoint)` pair. Pass
    `on=` an instance from `instance_with` to preserve its settings instead of the shared one's.
    """
    snapshots: list[tuple[httpx.Client, str, dict]] = []
    clients = {admin.base_url: admin.client()}

    def snapshot(*settings: str | tuple[str, str], on: LaunchedInstance | None = None) -> None:
        base_url = on.base_url if on else admin.base_url
        if base_url not in clients:
            clients[base_url] = on.client()
        client = clients[base_url]
        for setting in settings:
            read_path, write_path = SETTINGS[setting] if isinstance(setting, str) else setting
            current = client.get(read_path)
            current.raise_for_status()
            snapshots.append((client, write_path, current.json()))

    yield snapshot
    for client, write_path, value in reversed(snapshots):
        restored = client.post(write_path, json=value)
        assert restored.status_code == 200, f"restoring {write_path} failed: {restored.text}"
    for client in clients.values():
        client.close()


@pytest.fixture(scope="session")
def instance_with() -> Generator[Callable[[dict[str, str]], LaunchedInstance], None, None]:
    """`instance_with({"ENV": "value"})` boots (once per env set) an instance of its own."""
    stack = contextlib.ExitStack()
    booted: dict[frozenset, LaunchedInstance] = {}

    def factory(env: dict[str, str]) -> LaunchedInstance:
        key = frozenset(env.items())
        if key not in booted:
            provider = stack.enter_context(_serving_upstream())
            launched = contextlib.contextmanager(_launch_published)(provider, env)
            booted[key] = stack.enter_context(launched)
        booted[key].upstream.reset()
        return booted[key]

    with stack:
        yield factory


@pytest.fixture
def listener() -> Generator[Listener, None, None]:
    """A local service for the instance to call, recording every request it gets."""
    with listening() as service:
        yield service
