r"""The terminal proxy refuses a path carrying a backslash before it reaches the terminal server.

Before 0.11.0 the proxy's path sanitizer normalized with `posixpath.normpath`, which splits on
`/` only, so a segment like `..\..\etc` survived as one opaque component and passed the
traversal check; a terminal server treating `\` as a separator then resolved it out of its
root. Fix `b40b6fd69` (#27198) refuses any path whose fully decoded form holds a backslash.
The paths are percent-encoded because httpx normalizes dot segments before sending. Twin of
unit/security/test_terminal_proxy_path.py.

Discriminates: passes on dev bbfa876af, fails with the backslash dropped from the sanitizer's
refused characters (`safe\segment`, `a/..\..\etc/passwd` and the encoded forms reach the fake
terminal server; a leading `..\` is still refused by the older `..` prefix check).
"""

from __future__ import annotations

from typing import Iterator

import pytest

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    serving_terminal,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        yield server


@pytest.fixture
def proxy(admin, preserve, terminal):
    """`proxy(path)` sends a GET for `path` through the proxy, as the admin, and answers it."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = terminal.connection()
    with admin.client() as client:
        configure_terminals(client, connection)
    terminal.clear()
    client = admin.client()
    yield lambda path: client.get(f"/api/v1/terminals/{connection['id']}/{path}")
    client.close()


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("a%2F..%5C..%5Cetc%2Fpasswd", id="backslash_traversal"),
        pytest.param("..%5C..%5Cetc%2Fpasswd", id="leading_backslash_traversal"),
        pytest.param("safe%5Csegment", id="backslash_in_a_name"),
        pytest.param("a%2F%255c..%255cetc", id="encoded_backslash"),
        pytest.param("a%2F%25255c..%25255cetc", id="double_encoded_backslash"),
    ],
)
def test_a_backslash_path_never_reaches_the_terminal(proxy, terminal, path):
    response = proxy(path)
    assert response.status_code == 400, (
        f"{path} was proxied; a terminal server treating '\\' as a separator resolves it out "
        "of its root (#27198)"
    )
    assert terminal.received == []


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("%2E%2E%2Fetc%2Fpasswd", id="parent"),
        pytest.param("%2E%2E", id="parent_only"),
        pytest.param("%2E", id="current_only"),
        pytest.param("%252e%252e%2Fetc", id="encoded_parent"),
    ],
)
def test_forward_slash_traversal_is_still_refused(proxy, terminal, path):
    assert proxy(path).status_code == 400
    assert terminal.received == []


@pytest.mark.parametrize(
    "path, arrives_as",
    [
        ("files/report.pdf", "/files/report.pdf"),
        ("%2Ffiles/report.pdf", "/files/report.pdf"),
        ("files/", "/files/"),
        ("files/%2E/report.pdf", "/files/report.pdf"),
        ("files/sub/%2E%2E/report.pdf", "/files/report.pdf"),
    ],
)
def test_ordinary_paths_reach_the_terminal_normalized(proxy, terminal, path, arrives_as):
    terminal.route("GET", arrives_as, json_answer({"ok": True}))
    assert proxy(path).status_code == 200
    assert [request.path for request in terminal.received] == [arrives_as]
