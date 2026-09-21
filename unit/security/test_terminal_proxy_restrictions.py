"""Regression tests for two 0.11.4 terminal-server fixes in routers/terminals.py.

* "Terminal proxy restrictions" (`51bb8cb14`): requests proxied to a terminal
  server were followed on to anywhere the upstream suggested and accepted paths
  that URL parsers rewrite. The fix refuses the server's administrative
  endpoints (ADMIN_API_PATHS, matched against the parsed target path relative
  to the configured base), disables redirect following, and rejects paths
  carrying characters a parser would rewrite (backslash, tab, CR, LF — the
  tab/newline half landed with this change's sanitizer extension).
* "Terminal sessions follow access changes" (`a1189a2d7`): a terminal session
  authenticated once and never re-checked, so removing someone's permission or
  deactivating their account left the live terminal open. The fix resolves the
  user, connection and access again on every pass (_resolve_terminal_access),
  and a watchdog re-runs it every 10 seconds for the lifetime of the session.

The proxy gate is driven through the shipped handler with the I/O boundary
faked. The recheck contract is pinned through the shipped _resolve_terminal_access
helper and the watchdog's fixed interval.

Discriminates: passes on dev 344ea5306; on the pre-fix refs the admin paths
and rewritten-character paths proxy through, and an open session never
re-resolves its access.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def terminals_router_module(owui_module):
    return owui_module("open_webui.routers.terminals")


def _user(role="admin"):
    return SimpleNamespace(id="root", role=role, email="root@example.com", name="root")


async def _proxy(terminals_router_module, path, connection=None, query=""):
    """Drive the shipped proxy gate with the I/O boundary faked out."""
    connection = connection or {
        "id": "t1",
        "url": "http://terminal.internal:8080/openapi.json",
        "enabled": True,
    }
    upstream_response = SimpleNamespace(
        status=200,
        headers={"content-type": "application/json"},
        content=SimpleNamespace(iter_any=lambda: []),
        read=AsyncMock(return_value=b'{"ok": true}'),
        release=AsyncMock(),
    )
    requested = {}

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def request(self, method, url, **kwargs):
            requested["url"] = url
            requested["allow_redirects"] = kwargs.get("allow_redirects")
            return upstream_response

        async def close(self):
            return None

    session_cls = terminals_router_module.aiohttp.ClientSession
    terminals_router_module.aiohttp.ClientSession = FakeSession
    try:
        with (
            patch.object(
                terminals_router_module.Config,
                "get",
                AsyncMock(return_value=[connection]),
            ),
            patch.object(
                terminals_router_module, "has_connection_access", AsyncMock(return_value=True)
            ),
            patch.object(
                terminals_router_module.Groups,
                "get_groups_by_member_id",
                AsyncMock(return_value=[]),
            ),
        ):
            request = SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(redis=None)),
                query_params=query,
                headers={"authorization": "Bearer k"},
                method="GET",
                body=AsyncMock(return_value=b""),
            )
            response = await terminals_router_module.proxy_terminal(
                "t1", path, request, _user()
            )
    finally:
        terminals_router_module.aiohttp.ClientSession = session_cls
    return response, requested


# ── narrow: the administrative endpoints are refused before any fetch ─────


@pytest.mark.asyncio
@pytest.mark.parametrize("admin_path", ["api/v1/policies", "api/v1/status", "api/v1/terminals"])
async def test_admin_paths_are_refused(terminals_router_module, admin_path):
    response, requested = await _proxy(terminals_router_module, admin_path)

    assert response.status_code == 403, (
        f"a proxied request to the terminal server's {admin_path} endpoint was passed "
        "through, giving a chat-level user a route to the server's administrative API "
        "(51bb8cb14)"
    )
    assert not requested


@pytest.mark.asyncio
@pytest.mark.parametrize("admin_path", ["api/v1/policies", "api/v1/terminals"])
async def test_admin_subpaths_are_refused(terminals_router_module, admin_path):
    response, requested = await _proxy(terminals_router_module, f"{admin_path}/item-1")

    assert response.status_code == 403
    assert not requested


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["a/..\\..\\etc", "a/.\t./etc", "a/.\n./etc", "a/.\r./etc"])
async def test_parser_rewritable_characters_are_refused(terminals_router_module, path):
    """The sanitizer extension: a tab or newline between two dots becomes '..'
    once a URL parser strips it, so the raw characters are refused outright."""
    response, requested = await _proxy(terminals_router_module, path)

    assert response.status_code == 400, (
        f"{path!r} was accepted; an HTTP client that strips the control character "
        "resolves it as a traversal out of the server root (51bb8cb14)"
    )
    assert not requested


@pytest.mark.asyncio
async def test_ordinary_paths_still_proxy(terminals_router_module):
    response, requested = await _proxy(terminals_router_module, "files/report.txt")

    assert response.status_code == 200
    assert "files/report.txt" in requested["url"]


# ── narrow: a session re-checks access on a timer and on revocation ───────


def test_recheck_interval_is_bounded(terminals_router_module, open_webui_backend):
    """The shipped watchdog polls access on a fixed timer; pin the interval in the
    source so a regression to an unbounded session is visible."""
    source = (
        open_webui_backend / "open_webui" / "routers" / "terminals.py"
    ).read_text(encoding="utf-8")
    assert "await asyncio.sleep(10)" in source, (
        "the terminal session watchdog no longer polls access on a timer, so a revoked "
        "user keeps their terminal indefinitely (a1189a2d7)"
    )


@pytest.mark.asyncio
async def test_resolved_access_fails_when_the_token_no_longer_verifies(
    terminals_router_module, monkeypatch
):
    """The session recheck resolves the user from the token on every pass."""
    closed = []
    ws = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(redis=None)),
        close=AsyncMock(side_effect=lambda code, reason: closed.append((code, reason))),
    )
    monkeypatch.setattr(
        terminals_router_module,
        "get_verified_user_by_token",
        AsyncMock(return_value=None),
    )

    result = await terminals_router_module._resolve_terminal_access(ws, "t1", "stale-token")

    assert result is None, (
        "a terminal session whose user no longer verifies still resolved access, so a "
        "deactivated account keeps its live terminal (a1189a2d7)"
    )
    assert closed and closed[0][0] == 4001


@pytest.mark.asyncio
async def test_resolved_access_fails_when_the_connection_is_gone(
    terminals_router_module, monkeypatch
):
    """A connection removed or disabled ends the open session on the next recheck."""
    ws = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(redis=None)),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        terminals_router_module,
        "get_verified_user_by_token",
        AsyncMock(return_value=_user()),
    )
    monkeypatch.setattr(
        terminals_router_module.Config,
        "get",
        AsyncMock(return_value=[]),
    )

    result = await terminals_router_module._resolve_terminal_access(ws, "removed-server", "tok")

    assert result is None, (
        "a terminal session whose connection was removed or disabled still resolved "
        "access (a1189a2d7)"
    )
