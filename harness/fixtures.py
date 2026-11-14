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
from harness.file_context import file_context_ticked
from harness.host_names import c_ares_refusal_problem
from harness.instance import LaunchedInstance, launch
from harness.listener import Listener, listening
from harness.upstream import MOCK_MODEL_ID, MockUpstream
from harness.web_retrieval import RETRIEVAL_CONFIG

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

# instances from `instance_with` left running after their module, for the next module to reuse
IDLE_INSTANCES_KEPT = 2

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


@pytest.fixture
def chunked_retrieval(admin: Actor, preserve: Callable[..., None]) -> None:
    """Small files searched by their chunks on the shared instance, not sent whole."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": 0})
    assert saved.status_code == 200, f"turning the whole-file threshold off failed: {saved.text}"


@pytest.fixture
def file_context_on(admin: Actor) -> Iterator[None]:
    """File Context ticked in the shared instance's Model Defaults for the test."""
    with file_context_ticked(admin):
        yield


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


Booted = dict[frozenset, tuple[LaunchedInstance, contextlib.ExitStack]]


def _boot_published(env: dict[str, str]) -> tuple[LaunchedInstance, contextlib.ExitStack]:
    with contextlib.ExitStack() as stack:
        provider = stack.enter_context(_serving_upstream())
        launched = stack.enter_context(contextlib.contextmanager(_launch_published)(provider, env))
        return launched, stack.pop_all()


@pytest.fixture(scope="session")
def _idle_instances() -> Generator[Booted, None, None]:
    idle: Booted = {}
    yield idle
    for _, stack in idle.values():
        stack.close()


@pytest.fixture(scope="module")
def instance_with(
    _idle_instances: Booted,
) -> Generator[Callable[[dict[str, str]], LaunchedInstance], None, None]:
    """`instance_with({"ENV": "value"})` boots an instance for that env set, once per env set.

    Every call resets that instance's provider, so call it before queueing replies. After the
    module the last `IDLE_INSTANCES_KEPT` instances stay up for a later module asking for the same
    env set, which keeps the suite's memory to a few instances at a time.
    """
    booted: Booted = {}

    def factory(env: dict[str, str]) -> LaunchedInstance:
        key = frozenset(env.items())
        if key not in booted:
            booted[key] = _idle_instances.pop(key, None) or _boot_published(env)
        launched = booted[key][0]
        launched.upstream.reset()
        return launched

    yield factory
    _idle_instances.update(booted)
    while len(_idle_instances) > IDLE_INSTANCES_KEPT:
        _, stack = _idle_instances.pop(next(iter(_idle_instances)))
        stack.close()


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


@pytest.fixture(scope="session")
def agent_terminal() -> Generator:
    """A real Open Terminal with the stand-in coding agent; skips without open-terminal."""
    from harness.coding_agent import serving_agent_terminal

    with serving_agent_terminal() as terminal:
        yield terminal


@pytest.fixture(scope="session")
def native_agent_terminal() -> Generator:
    """A real Open Terminal with stand-ins for Claude Code, Codex and Pi; skips without it."""
    from harness.coding_agent import serving_agent_terminal

    with serving_agent_terminal(native=True) as terminal:
        yield terminal


@pytest.fixture(scope="session")
def full_open_terminal() -> Generator:
    """A real Open Terminal from the full image, in a container: Chromium for Chrome mode, a
    desktop with sound and notebook kernels. Skips without `OPEN_TERMINAL_FULL_IMAGE`."""
    from harness.open_terminal import serving_full_open_terminal

    with serving_full_open_terminal() as terminal:
        yield terminal


@pytest.fixture(scope="session")
def full_terminal_image() -> str:
    """The name of the full Open Terminal image, for a template's `image` on the Terminals
    orchestrator; skips without `OPEN_TERMINAL_FULL_IMAGE`."""
    import os

    from harness.open_terminal import FULL_IMAGE_ENV, FULL_REQUIRED_ENV, full_image_unavailable

    reason = full_image_unavailable()
    if reason and os.getenv(FULL_REQUIRED_ENV):
        pytest.fail(reason)
    if reason:
        pytest.skip(reason)
    return os.environ[FULL_IMAGE_ENV]


@pytest.fixture(scope="session")
def terminals_service() -> Generator:
    """The real Terminals orchestrator on Docker for the session; skips without its images."""
    from harness.terminals import serving_terminals

    with serving_terminals() as terminals:
        yield terminals


@pytest.fixture
def terminals(terminals_service, admin: Actor) -> Generator:
    """The real Terminals orchestrator; the templates and containers a test made go after it."""
    from harness.terminal_server import TERMINAL_SERVERS_CONFIG

    yield terminals_service
    # through Open WebUI first, which keeps each terminal's shares and drops them with it
    with admin.client() as client:
        saved = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
        for connection in saved or []:
            if connection.get("url") != terminals_service.base_url:
                continue
            for template in terminals_service.templates():
                client.delete(
                    f"/api/v1/terminals/templates/template:{connection['id']}:{template['id']}"
                )
    terminals_service.reset()


@pytest.fixture
def connect_terminals(admin: Actor, preserve, terminals) -> Callable[..., dict]:
    """`connect_terminals(*grants)`: the admin connects the Terminals orchestrator under
    Settings > Integrations, replacing the terminal connections there, and gives the connection
    those grants (`read_grant`, `write_grant`); returns the connection."""
    from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals

    def connect(*grants: dict, **fields) -> dict:
        preserve(TERMINAL_SERVERS_CONFIG)
        config = {**fields.pop("config", {}), "access_grants": list(grants)}
        connection = terminals.connection(config=config, **fields)
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection

    return connect


@pytest.fixture
def shared_terminal(admin: Actor, connect_terminals) -> Callable[..., str]:
    """`shared_terminal(*people, **data)`: the admin connects the Terminals orchestrator, creates a
    terminal on it (one machine per user unless `scope` says otherwise) and shares connection and
    terminal with `people`; returns the terminal's id."""
    from harness.terminal_server import read_grant
    from harness.terminals import create_template, share_template

    def share(*people: Actor, **data) -> str:
        grants = [read_grant(person.id) for person in people]
        connection = connect_terminals(*grants)
        with admin.client() as client:
            terminal_id = create_template(client, connection, **data)["terminal_id"]
            share_template(client, terminal_id, *grants)
        return terminal_id

    return share


@pytest.fixture
def connect_device(instance: LaunchedInstance) -> Generator[Callable[..., object], None, None]:
    """`connect_device(account, **options)`: a fake Tether device of that account, connected with
    a renewable session of its own (`device.session`) until the test ends. `on=` connects it to
    another instance; the options are those of `harness.tether_device.tether_device`."""
    from harness.tether_device import tether_device, tether_session

    devices = []
    with contextlib.ExitStack() as stack:

        def connect(account: Actor, on: LaunchedInstance | None = None, **options):
            target = on or instance
            session = tether_session(target, account)
            device = stack.enter_context(
                tether_device(target.base_url, session["token"], **options)
            )
            device.session = session
            devices.append(device)
            return device

        yield connect
    failures = [failure for device in devices for failure in device.failures]
    assert not failures, f"the fake device failed while serving: {failures!r}"


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
