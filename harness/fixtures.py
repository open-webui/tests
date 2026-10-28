"""Pytest fixtures over the harness, shared by the integration and browser suites.

`instance` is one scratch backend per session that the regression tests share; every test gets
a clean `upstream` script and can add accounts with `make_user`. Tests that change global
settings wrap them in `preserve`, which restores what it snapshotted. `instance_with(env)`
boots a second backend for settings that only exist as environment variables, and
`package_instance_with(env)` keeps one for every module of a test package.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Generator, Iterator

import pytest

from harness import upstream as upstream_module
from harness.actors import Actor, admin_of, create_user
from harness.host_names import c_ares_refusal_problem
from harness.instance import LaunchedInstance, launch
from harness.listener import Listener, listening
from harness.upstream import MOCK_MODEL_ID, MockUpstream

TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"

# these tasks share the chat endpoint and would take a reply scripted for the chat
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
    "connections": ("/api/v1/configs/connections", "/api/v1/configs/connections"),
}


@contextlib.contextmanager
def _serving_upstream() -> Iterator[MockUpstream]:
    upstream, shutdown = upstream_module.serve()
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
def direct_integrations(admin: Actor, preserve) -> Callable[[bool], None]:
    """Sets the admin's Direct Integrations switch; the tool servers and terminals a user adds
    in their own settings reach the model only while it is on."""
    preserve("connections")
    read_path, write_path = SETTINGS["connections"]

    def switch(enabled: bool) -> None:
        with admin.client() as client:
            current = client.get(read_path).json()
            saved = client.post(write_path, json={**current, "ENABLE_DIRECT_INTEGRATIONS": enabled})
        assert saved.status_code == 200, saved.text

    return switch


@pytest.fixture
def preserve(request: pytest.FixtureRequest) -> Generator[Callable[..., None], None, None]:
    """`preserve("permissions", ...)` snapshots those settings and restores them afterwards.

    A setting is a name from `SETTINGS` or a `(read endpoint, write endpoint)` pair. Pass
    `on=` an instance from `instance_with` to preserve its settings instead of the shared one's.
    """
    snapshots: list[tuple[LaunchedInstance, str, dict]] = []

    def snapshot(*settings: str | tuple[str, str], on: LaunchedInstance | None = None) -> None:
        # the shared instance only boots when a test preserves its settings
        target = on or request.getfixturevalue("instance")
        with target.client() as client:
            for setting in settings:
                read_path, write_path = SETTINGS[setting] if isinstance(setting, str) else setting
                current = client.get(read_path)
                current.raise_for_status()
                snapshots.append((target, write_path, current.json()))

    yield snapshot
    failures = []
    for target, write_path, value in reversed(snapshots):
        # a new connection: one kept from the snapshot can be closed by the server's keep-alive
        # timeout just as the restore goes out on it
        with target.client() as client:
            restored = client.post(write_path, json=value)
        if restored.status_code != 200:
            failures.append(f"restoring {write_path} failed: {restored.text}")
    assert not failures, failures


@contextlib.contextmanager
def _instances_by_env() -> Iterator[Callable[[dict[str, str]], LaunchedInstance]]:
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


@pytest.fixture(scope="module")
def instance_with() -> Generator[Callable[[dict[str, str]], LaunchedInstance], None, None]:
    """`instance_with({"ENV": "value"})` boots an instance of its own, once per env set and module.

    Every call resets that instance's provider, so call it before queueing replies. The
    instances stop with the module, which keeps the suite's memory to a few instances at a time.
    """
    with _instances_by_env() as factory:
        yield factory


@pytest.fixture(scope="package")
def package_instance_with() -> Generator[Callable[[dict[str, str]], LaunchedInstance], None, None]:
    """`instance_with` for a test package whose modules share one env set, booted once for all."""
    with _instances_by_env() as factory:
        yield factory


@pytest.fixture
def milvus_service() -> Generator:
    """A Milvus for external knowledge to search, recording every call it gets."""
    pytest.importorskip("pymilvus", reason="pymilvus not installed in this env")
    from harness.milvus_server import serving_milvus

    with serving_milvus() as fake:
        yield fake


@pytest.fixture(scope="session")
def open_terminal() -> Generator:
    """A real Open Terminal for the session; skips where open-terminal is not installed."""
    from harness.open_terminal import serving_open_terminal

    with serving_open_terminal() as terminal:
        yield terminal


@pytest.fixture
def terminal_orchestrator() -> Generator:
    """A Terminals orchestrator stand-in giving each terminal context a real Open Terminal."""
    from harness.terminal_orchestrator import serving_terminal_orchestrator

    with serving_terminal_orchestrator() as orchestrator:
        yield orchestrator


@pytest.fixture
def refuses_unknown_names(resolver: str) -> None:
    """Skips a c-ares test of a name that does not resolve where this machine's DNS server keeps
    c-ares from refusing one at once, naming the reason; the probe runs once per session."""
    problem = c_ares_refusal_problem() if resolver == "c-ares" else None
    if problem:
        pytest.skip(problem)


@pytest.fixture
def listener() -> Generator[Listener, None, None]:
    """A local service for the instance to call, recording every request it gets."""
    with listening() as service:
        yield service
