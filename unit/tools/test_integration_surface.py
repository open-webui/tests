"""Guard: the terminal policy route `/p/<policy>` is built in exactly one place.

Issue #26945 (fixes `9a772f42c` and `09d4cccb7`): terminal call sites each built the orchestrator
policy route themselves, some only for orchestrator connections, so a connection with a policy
could reach the root route and the orchestrator started an unintended container. The route now
comes only from `get_terminal_server_url` in `utils/terminals.py`, whose behaviour
integration/tools/test_integration_surface.py pins through the terminal proxy. This audit keeps
the next call site from building the route by hand: no string literal outside that module may
contain `/p/`.

The rest of this file moved to integration/tools/test_integration_surface.py; its
`_set_terminal_cwd` tests were dropped because that function has no caller on dev bbfa876af.

Discriminates: passes on dev bbfa876af, fails with `_set_terminal_cwd` building its URL as
`f'{base_url}/p/{connection["policy_id"]}/files/cwd'`, the shape `9a772f42c` removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROUTE_MARKER = "/p/"
ROUTE_HOME = Path("open_webui/utils/terminals.py")


def _route_literals(path: Path) -> list[int]:
    """Lines of the string literals (f-string parts included) in `path` that hold `/p/`."""
    source = path.read_text(encoding="utf-8")
    if ROUTE_MARKER not in source:
        return []
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and ROUTE_MARKER in node.value
    ]


def test_only_the_terminal_helper_builds_the_policy_route(open_webui_backend):
    builders = {
        (path.relative_to(open_webui_backend), line)
        for path in (open_webui_backend / "open_webui").rglob("*.py")
        for line in _route_literals(path)
    }
    elsewhere = sorted(f"{path}:{line}" for path, line in builders if path != ROUTE_HOME)

    assert any(path == ROUTE_HOME for path, _ in builders), (
        f"{ROUTE_HOME} no longer builds the policy route; retarget this audit at its new home"
    )
    assert elsewhere == [], f"these build the terminal policy route by hand: {elsewhere}"
