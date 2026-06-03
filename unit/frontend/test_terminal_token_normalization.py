"""Source-contract tests for Open Terminal bearer-token normalization.

Regression for open-webui/open-webui#25613.

A bearer token configured with accidental trailing whitespace let the
REST file-API paths work (HTTP header parsing tolerates the space) but
broke the interactive terminal: the WebSocket first-message auth
`{ type: 'auth', token: '<tok> ' }` is an exact string compare on the
server, so the trailing space failed auth → "[Connection closed]".

Fix (PR #25642): a `normalizeTerminalToken` helper trims the token, and
every terminal auth path — REST headers AND the WebSocket auth message —
routes through it.

The PR's own vitest covers `normalizeTerminalToken` + one REST helper,
but NOT `XTerminal.svelte` (the actual failure site). These source-audit
tests (the external suite has no JS toolchain — same approach as
test_workspace_permissions.py) fill that gap:

  specific — XTerminal.svelte: the WebSocket auth token is normalized
  broad    — every `Bearer ${...}` in the terminal API module normalizes
             its token; no raw interpolation survives

A normalized expression is one that applies `.trim()` or
`normalizeTerminalToken(...)` — the test checks the behaviour contract,
not a specific helper name, so an inline-trim refactor still passes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _frontend(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent / "src"


# A token expression counts as normalized if it is trimmed or run through
# the terminal token normalizer.
_NORMALIZED = re.compile(r"normalizeTerminalToken\s*\(|\.trim\s*\(")

# `Bearer ${ ... }` template interpolation, capturing the inner expression.
_BEARER_INTERP = re.compile(r"Bearer \$\{([^}]*)\}")

# The WebSocket first-message auth payload's token expression.
_WS_AUTH_TOKEN = re.compile(r"type:\s*['\"]auth['\"]\s*,\s*token:\s*([^}]+?)\s*\}")


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"source file not found: {path}")
    return path.read_text(encoding="utf-8")


# =============================================================================
# Specific — open-webui#25613: the WebSocket auth path
# =============================================================================


@pytest.mark.regression
def test_xterminal_websocket_auth_token_is_normalized(open_webui_backend: Path) -> None:
    """Regression for open-webui/open-webui#25613.

    XTerminal.svelte sends the bearer token in the WebSocket first-message
    auth payload, which the server matches exactly. That token MUST be
    normalized (trimmed) or a trailing space silently closes the terminal
    while every REST path still works.
    """
    src = _read(_frontend(open_webui_backend) / "lib" / "components" / "chat" / "XTerminal.svelte")

    auth_tokens = _WS_AUTH_TOKEN.findall(src)
    assert auth_tokens, (
        "Couldn't find the WebSocket `{ type: 'auth', token: ... }` payload "
        "in XTerminal.svelte — the file structure changed; update this test."
    )
    unnormalized = [expr for expr in auth_tokens if not _NORMALIZED.search(expr)]
    assert not unnormalized, (
        "Regression of open-webui/open-webui#25613: the WebSocket auth token "
        f"is sent without trimming: token={unnormalized!r}. A trailing space "
        "in the configured bearer token will close the interactive terminal "
        "even though REST file APIs still work. Wrap it in "
        "normalizeTerminalToken(...) / .trim()."
    )


@pytest.mark.regression
def test_xterminal_applies_normalization_somewhere(open_webui_backend: Path) -> None:
    """Belt-and-suspenders: XTerminal.svelte must apply token normalization
    at all (catches a refactor that drops the helper import)."""
    src = _read(_frontend(open_webui_backend) / "lib" / "components" / "chat" / "XTerminal.svelte")
    assert _NORMALIZED.search(src), (
        "XTerminal.svelte applies no token normalization (.trim() / "
        "normalizeTerminalToken) — the terminal bearer token is used raw."
    )


# =============================================================================
# Broad — every terminal API Bearer header normalizes its token
# =============================================================================


@pytest.mark.regression
def test_all_terminal_api_bearer_headers_are_normalized(
    open_webui_backend: Path,
) -> None:
    """Broad #25613 guard: in the terminal API module, every
    `Authorization: Bearer ${...}` interpolation must normalize its token.
    A single raw helper reintroduces the trailing-whitespace footgun for
    that endpoint."""
    index = _frontend(open_webui_backend) / "lib" / "apis" / "terminal" / "index.ts"
    src = _read(index)

    interps = _BEARER_INTERP.findall(src)
    assert interps, (
        "No `Bearer ${...}` interpolations found in terminal/index.ts — "
        "the auth-header construction changed; update this test."
    )
    raw = sorted({expr.strip() for expr in interps if not _NORMALIZED.search(expr)})
    assert not raw, (
        "Regression of open-webui/open-webui#25613: terminal API helper(s) "
        f"build a Bearer header from an un-normalized token: {raw}. Every "
        "terminal auth path must trim the token (normalizeTerminalToken / "
        ".trim()) so trailing whitespace can't break some endpoints but not "
        "others."
    )


@pytest.mark.regression
def test_terminal_module_exposes_a_token_normalizer(open_webui_backend: Path) -> None:
    """The terminal API module must define a trimming normalizer that the
    other call sites (and XTerminal) can share."""
    index = _frontend(open_webui_backend) / "lib" / "apis" / "terminal" / "index.ts"
    src = _read(index)
    assert _NORMALIZED.search(src), (
        "terminal/index.ts defines no token trimming — there's nothing for "
        "the REST helpers or the WebSocket terminal to normalize through."
    )
