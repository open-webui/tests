"""Source-contract tests for Open Terminal bearer-token normalization.

Regression for open-webui/open-webui#25613.

WHAT #25613 IS ABOUT — the user-typed remote-terminal API KEY.
The footgun is the *configured remote-terminal bearer key*: a value the
user types into AddTerminalServerModal, which can pick up accidental
trailing whitespace. With a stray trailing space the REST file-API paths
still worked (HTTP header parsing tolerates it) but the interactive
terminal broke: the WebSocket first-message auth
`{ type: 'auth', token: '<key> ' }` is an exact string compare on the
server, so the space failed auth → "[Connection closed]".

Fix (#25686, inline `.trim()`): the user-typed key is normalized on save
(AddTerminalServerModal `key = key.trim()`) and at every use — the shared
`bearerHeaders(apiKey)` REST helper (`Bearer ${apiKey.trim()}`) and the
WebSocket auth message (`token: authToken.trim()`, the actual failure
site).

KEY vs JWT — what is in scope and what is NOT.
Two different bearer values flow through this feature:
  * the remote-terminal API KEY  — user-typed, may be dirty, MUST be
    trimmed. This is #25613's subject.
  * Open WebUI's own session JWT — `localStorage.token`, sent to Open
    WebUI's OWN `/terminals/` endpoint (getTerminalServers in
    terminal/index.ts, and XTerminal's system-terminal proxy path).
    Server-issued, never user-typed, always clean → trimming it is
    meaningless and it is OUT of #25613's scope.
The broad audit below therefore excludes getTerminalServers' app-JWT
`Bearer ${token}`; auditing it would just trip on a non-bug.

The upstream vitest covers the REST helper but NOT `XTerminal.svelte`
(the actual failure site). These source-audit tests (the external suite
has no JS toolchain — same approach as test_workspace_permissions.py)
fill that gap:

  specific — XTerminal.svelte: the WebSocket auth token is normalized
  broad    — every API-KEY `Bearer ${...}` in the terminal API module
             normalizes its token; no raw API-key interpolation survives
             (the app-JWT header is excluded)

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

# The app-session-JWT auth header — `Bearer ${token}`, where `token` is
# localStorage.token sent to Open WebUI's own /terminals/ endpoint
# (getTerminalServers). Server-issued, never user-typed, always clean, so it
# is out of #25613's remote-terminal-API-KEY scope and excluded from the audit.
_APP_JWT_HEADER = re.compile(r"token")

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
    """Broad #25613 guard, scoped to the user-typed remote-terminal API KEY.

    #25613 is about the *configured remote-terminal bearer key* — a value a
    user types into AddTerminalServerModal, which can carry stray trailing
    whitespace. That key reaches the remote open-terminal service through the
    `bearerHeaders(apiKey)` helper (`Bearer ${apiKey.trim()}`), so its
    interpolation MUST be normalized.

    This is deliberately NOT a blanket "every `Bearer ${...}` in the file"
    audit. `getTerminalServers(token)` builds `Bearer ${token}` inline, where
    `token` is `localStorage.token` — Open WebUI's OWN server-issued session
    JWT, sent to Open WebUI's OWN `/terminals/` endpoint. That value is never
    user-typed and is always clean, so trimming it is meaningless and it is
    explicitly out of #25613's scope. Auditing it just trips on a non-bug
    (it has no `.trim()` and never needs one).

    So: audit only interpolations of the API-KEY expression (`apiKey`),
    excluding the app-JWT expression (`token`). A raw API-key Bearer header
    reintroduces the trailing-whitespace footgun; a raw app-JWT header does
    not.
    """
    index = _frontend(open_webui_backend) / "lib" / "apis" / "terminal" / "index.ts"
    src = _read(index)

    interps = _BEARER_INTERP.findall(src)
    assert interps, (
        "No `Bearer ${...}` interpolations found in terminal/index.ts — "
        "the auth-header construction changed; update this test."
    )

    # Keep only the user-typed-API-KEY headers; drop the app-session-JWT one
    # (getTerminalServers' `Bearer ${token}`), which is out of #25613 scope.
    api_key_interps = [expr for expr in interps if not _APP_JWT_HEADER.fullmatch(expr.strip())]
    assert api_key_interps, (
        "No API-KEY `Bearer ${...}` interpolation found in terminal/index.ts "
        "(only the app-JWT `Bearer ${token}` header). The remote-terminal "
        "auth-header construction changed; update this test."
    )

    raw = sorted({expr.strip() for expr in api_key_interps if not _NORMALIZED.search(expr)})
    assert not raw, (
        "Regression of open-webui/open-webui#25613: terminal API helper(s) "
        f"build a Bearer header from an un-normalized API key: {raw}. The "
        "user-typed remote-terminal key must be trimmed (normalizeTerminalToken "
        "/ .trim()) so trailing whitespace can't break some endpoints but not "
        "others. (The app-session-JWT header — getTerminalServers' "
        "`Bearer ${token}` — is intentionally excluded: that token is "
        "server-issued and always clean.)"
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
