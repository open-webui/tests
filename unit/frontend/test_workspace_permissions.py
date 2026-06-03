"""Regression tests for per-control workspace permissions wiring.

The backend declares N workspace ACCESS permissions
(`USER_PERMISSIONS_WORKSPACE_<NAME>_ACCESS`); the frontend has three
places where every one of those keys has to be honoured for the user
to actually reach the section:

  1. Sidebar nav-item visibility (`src/.../layout/Sidebar.svelte`,
     case 'workspace') — the OR-chain that decides whether the
     Workspace entry shows up in the sidebar at all.
  2. `/workspace` index redirect (`src/routes/(app)/workspace/+page.svelte`)
     — the if/elif chain that picks which subpage to send the user to.
  3. Per-route guards (`src/routes/(app)/workspace/+layout.svelte`)
     — the if/elif chain that bounces non-permissioned users away from
     each subpage.

The Skills permission shipped without any of (1) or (2) (open-webui
discussion #24719 — `bwgabrielsusai`): a user with *only*
`workspace.skills` couldn't see the Workspace menu, and visiting
`/workspace` directly bounced them to `/`.

These tests scan both source trees and assert every backend ACCESS key
is wired up in each of the three frontend gates. Adding a new
workspace ACCESS permission to the backend will fail this test until
the corresponding frontend wiring is in place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ACCESS_RE = re.compile(r"USER_PERMISSIONS_WORKSPACE_([A-Z]+)_ACCESS\b")


@pytest.fixture(scope="module")
def workspace_access_keys(open_webui_backend: Path) -> set[str]:
    """The set of workspace-access permission keys declared in
    backend/open_webui/config.py — e.g. {'models', 'knowledge',
    'prompts', 'tools', 'skills'}.
    """
    config_path = open_webui_backend / "open_webui" / "config.py"
    assert config_path.is_file(), config_path
    text = config_path.read_text(encoding="utf-8")
    keys = {m.group(1).lower() for m in _BACKEND_ACCESS_RE.finditer(text)}
    assert keys, (
        "Couldn't find any USER_PERMISSIONS_WORKSPACE_<NAME>_ACCESS "
        "constants in config.py — has the naming changed?"
    )
    return keys


def _open_webui_frontend(open_webui_backend: Path) -> Path:
    """`backend` lives at `<repo>/backend`; the frontend src is at
    `<repo>/src`. The fixture resolves the backend, so the frontend
    is its parent's `src/` sibling.
    """
    return open_webui_backend.parent / "src"


# -----------------------------------------------------------------------------
# Helpers — extract the relevant predicate / chain from each frontend file
# -----------------------------------------------------------------------------


def _extract_sidebar_workspace_block(text: str) -> str:
    """Return the body of the `case 'workspace':` branch in Sidebar.svelte."""
    m = re.search(r"case\s+'workspace'\s*:(?P<body>.*?)case\s+'", text, re.DOTALL)
    assert m, "Couldn't locate `case 'workspace':` in Sidebar.svelte"
    return m.group("body")


def _extract_workspace_index_chain(text: str) -> str:
    """Return the if/else-if chain in `/workspace/+page.svelte`."""
    m = re.search(
        r"if\s*\(\s*\$user\?\.role\s*!==\s*'admin'\s*\)\s*\{(?P<body>.*?)\}\s*else\s*\{",
        text,
        re.DOTALL,
    )
    assert m, "Couldn't locate the role gate in /workspace/+page.svelte"
    return m.group("body")


def _extract_workspace_route_guards(text: str) -> str:
    """Return the onMount role-gated if/elif chain in
    `/workspace/+layout.svelte`."""
    m = re.search(
        r"if\s*\(\s*\$user\?\.role\s*!==\s*'admin'\s*\)\s*\{(?P<body>.*?)\}\s*loaded",
        text,
        re.DOTALL,
    )
    assert m, "Couldn't locate the role gate in /workspace/+layout.svelte"
    return m.group("body")


def _keys_referenced(body: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"permissions\?\.workspace\?\.([A-Za-z_]+)", body)}


# -----------------------------------------------------------------------------
# Per-control tests
# -----------------------------------------------------------------------------


@pytest.mark.regression
def test_backend_declares_expected_access_keys(workspace_access_keys: set[str]) -> None:
    """Sanity guard: the backend still defines at least the five
    documented workspace access permissions. If this fails, a rename
    landed and the other tests in this file probably need updating."""
    expected = {"models", "knowledge", "prompts", "tools", "skills"}
    missing = expected - workspace_access_keys
    assert not missing, (
        f"Backend USER_PERMISSIONS_WORKSPACE_*_ACCESS lost keys: {missing}. "
        f"Got: {sorted(workspace_access_keys)}"
    )


@pytest.mark.regression
def test_sidebar_workspace_visibility_covers_every_access_key(
    open_webui_backend: Path, workspace_access_keys: set[str]
) -> None:
    """Regression for open-webui#24719 (bwgabrielsusai).

    The Sidebar 'workspace' case must reference every workspace ACCESS
    permission key in its OR-chain — otherwise a user with only one
    of those permissions (e.g. only `workspace.skills`) can't see the
    Workspace menu.

    The original bug: only models / knowledge / prompts / tools were
    in the chain; `skills` was missing.
    """
    sidebar = (
        _open_webui_frontend(open_webui_backend)
        / "lib"
        / "components"
        / "layout"
        / "Sidebar.svelte"
    )
    body = _extract_sidebar_workspace_block(sidebar.read_text(encoding="utf-8"))
    referenced = _keys_referenced(body)
    missing = workspace_access_keys - referenced
    assert not missing, (
        f"Sidebar.svelte 'workspace' visibility check doesn't reference "
        f"these workspace permission keys: {sorted(missing)}. "
        f"Users granted ONLY those permissions won't see the Workspace menu. "
        f"Add `||$user?.permissions?.workspace?.<key>` to the case 'workspace' "
        f"OR-chain for each."
    )


@pytest.mark.regression
def test_workspace_index_redirect_covers_every_access_key(
    open_webui_backend: Path, workspace_access_keys: set[str]
) -> None:
    """The `/workspace` index-page redirect chain must include every
    workspace ACCESS key — otherwise a non-admin with only that key
    visiting `/workspace` directly gets dumped on `/`.

    Same root cause as #24719: the chain was missing `skills`.
    """
    page = (
        _open_webui_frontend(open_webui_backend) / "routes" / "(app)" / "workspace" / "+page.svelte"
    )
    body = _extract_workspace_index_chain(page.read_text(encoding="utf-8"))
    referenced = _keys_referenced(body)
    missing = workspace_access_keys - referenced
    assert not missing, (
        f"`/workspace/+page.svelte` redirect chain doesn't reference "
        f"these workspace permission keys: {sorted(missing)}. "
        f"Non-admin users with ONLY those permissions hitting `/workspace` "
        f"will be redirected to `/`. Add an `else if "
        f"($user?.permissions?.workspace?.<key>) goto('/workspace/<key>')` "
        f"for each."
    )


@pytest.mark.regression
def test_workspace_route_guards_cover_every_access_key(
    open_webui_backend: Path, workspace_access_keys: set[str]
) -> None:
    """The `/workspace/+layout.svelte` onMount role gate must check
    every ACCESS permission against its matching URL prefix —
    otherwise a non-admin without that permission can still load the
    subpage by typing the URL directly.

    This one is currently green; the test exists so that adding a new
    workspace section without adding a corresponding URL guard fails
    loudly.
    """
    layout = (
        _open_webui_frontend(open_webui_backend)
        / "routes"
        / "(app)"
        / "workspace"
        / "+layout.svelte"
    )
    body = _extract_workspace_route_guards(layout.read_text(encoding="utf-8"))
    referenced = _keys_referenced(body)
    missing = workspace_access_keys - referenced
    assert not missing, (
        f"`/workspace/+layout.svelte` per-route guard doesn't check "
        f"these workspace permission keys: {sorted(missing)}. "
        f"Users without those permissions can reach the corresponding "
        f"subpage by typing the URL. Add an `else if "
        f"($page.url.pathname.includes('/<key>') && "
        f"!$user?.permissions?.workspace?.<key>) goto('/')` clause for each."
    )
