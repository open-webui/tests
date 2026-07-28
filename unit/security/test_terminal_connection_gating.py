"""Regression: two ways an admin-configured terminal connection reached the
wrong caller.

open-webui 0.11.0 fix `753798923` (refac): turning a terminal connection off
only removed it from `list_terminal_servers`. Every other entry point still
resolved it by id, so a user who knew the id kept browsing its files through
the HTTP proxy, opening a WebSocket shell on it, and calling its tools in a
chat. The fix rejects `enabled is False` in `proxy_terminal`,
`_resolve_authenticated_connection` and `get_terminal_tools` as well.

open-webui 0.11.0 fix `867006acc` (#27581, issues #27580 and #27064): with
`BYPASS_ADMIN_ACCESS_CONTROL` off, `has_connection_access` fell straight
through to `has_access`, which denies an empty grant list to everyone. A
freshly added connection carries no grants, so it became unreachable by every
caller including the admin who created it and the admin could no longer even
open it to add grants. The fix resolves the no-grants case to admin-only, which
is what the docstring already promised.

Discriminates: passes on v0.11.0, fails on v0.10.2 (disabled connections are
still served on all three non-list entry points, and an admin is denied a
connection that has no grants).
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from unit.security.conftest import FakeWebSocket

pytestmark = pytest.mark.regression

SERVER_ID = "terminal-1"
ADMIN = SimpleNamespace(id="admin-1", role="admin")
MEMBER = SimpleNamespace(id="member-1", role="user")


def _read_grant(user_id: str) -> dict:
    return {"principal_type": "user", "principal_id": user_id, "permission": "read"}


def _connection(enabled: bool = True, grants: list | None = None, url: str = "") -> dict:
    return {
        "id": SERVER_ID,
        "name": "Ops box",
        "url": url,
        "enabled": enabled,
        "config": {"access_grants": grants if grants is not None else []},
    }


def _patch_shared(stack, connection: dict) -> None:
    """`Config` and `Groups` are the only I/O the gates touch before deciding."""
    import open_webui.models.config as config_model
    import open_webui.models.groups as groups_model

    stack.enter_context(
        patch.object(config_model.Config, "get", AsyncMock(return_value=[connection]))
    )
    stack.enter_context(
        patch.object(groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[]))
    )


def _patch_ws_auth(stack, user) -> None:
    """v0.10.2 and v0.11.0 authenticate the socket through different helpers;
    patch every one of them on its defining module."""
    import open_webui.models.users as users_model
    import open_webui.utils.auth as auth_module

    stack.enter_context(
        patch.object(
            auth_module, "get_verified_user_by_token", AsyncMock(return_value=user), create=True
        )
    )
    stack.enter_context(
        patch.object(auth_module, "decode_token", lambda token: {"id": user.id}, create=True)
    )
    stack.enter_context(
        patch.object(auth_module, "is_valid_token", AsyncMock(return_value=True), create=True)
    )
    stack.enter_context(
        patch.object(users_model.Users, "get_user_by_id", AsyncMock(return_value=user))
    )


async def _entry_list(terminals_module, tools_module, user, connection) -> bool:
    with ExitStack() as stack:
        _patch_shared(stack, connection)
        listed = await terminals_module.list_terminal_servers(SimpleNamespace(), user=user)
    return any(entry["id"] == SERVER_ID for entry in listed)


async def _entry_http_proxy(terminals_module, tools_module, user, connection) -> bool:
    with ExitStack() as stack:
        _patch_shared(stack, connection)
        response = await terminals_module.proxy_terminal(
            SERVER_ID, "api/files", SimpleNamespace(), user=user
        )
    # The connection carries no URL, so passing the gates lands on 503.
    return response.status_code not in (403, 404)


async def _resolve_websocket(terminals_module, user, connection) -> tuple:
    ws = FakeWebSocket()
    with ExitStack() as stack:
        _patch_shared(stack, connection)
        _patch_ws_auth(stack, user)
        resolved = await terminals_module._resolve_authenticated_connection(ws, SERVER_ID)
    return resolved, ws


async def _entry_websocket_session(terminals_module, tools_module, user, connection) -> bool:
    resolved, _ = await _resolve_websocket(terminals_module, user, connection)
    return resolved is not None


async def _entry_tool_call(terminals_module, tools_module, user, connection) -> bool:
    spec = {"name": "run_command", "description": "run a shell command", "parameters": {}}
    server_data = {"id": SERVER_ID, "url": "http://terminal.invalid", "specs": [spec]}
    with ExitStack() as stack:
        _patch_shared(stack, connection)
        stack.enter_context(
            patch.object(
                tools_module, "get_terminal_servers", AsyncMock(return_value=[server_data])
            )
        )
        stack.enter_context(
            patch.object(tools_module, "get_terminal_cwd", AsyncMock(return_value=None))
        )
        stack.enter_context(
            patch.object(tools_module, "get_terminal_system_prompt", AsyncMock(return_value=None))
        )
        try:
            result = await tools_module.get_terminal_tools(SimpleNamespace(), SERVER_ID, user, {})
        except RuntimeError:
            return False
    tools = result[0] if isinstance(result, tuple) else result
    return bool(tools)


ENTRY_POINTS = {
    "list": _entry_list,
    "http_proxy": _entry_http_proxy,
    "websocket_session": _entry_websocket_session,
    "tool_call": _entry_tool_call,
}

# The three the fix had to repair; `list` already filtered on `enabled`.
GATED_BY_THE_FIX = ["http_proxy", "websocket_session", "tool_call"]


@pytest.fixture
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@pytest.fixture
def config_module(owui_module):
    return owui_module("open_webui.config")


# --- Narrow: a disabled connection must be refused on each repaired path ---


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", GATED_BY_THE_FIX)
async def test_disabled_connection_is_refused(
    entry_point, terminals_router_module, tools_module
):
    """The user holds an explicit read grant, so only `enabled` can refuse."""
    connection = _connection(enabled=False, grants=[_read_grant(MEMBER.id)])
    allowed = await ENTRY_POINTS[entry_point](
        terminals_router_module, tools_module, MEMBER, connection
    )
    assert allowed is False, (
        f"an administrator turned this terminal connection off, yet {entry_point} still "
        "served it to a user who knew its id"
    )


@pytest.mark.asyncio
async def test_disabled_connection_websocket_closes_with_disabled_reason(
    terminals_router_module, tools_module
):
    connection = _connection(enabled=False, grants=[_read_grant(MEMBER.id)])
    resolved, ws = await _resolve_websocket(terminals_router_module, MEMBER, connection)
    assert resolved is None
    assert ws.close_code == 4003
    assert "disabled" in (ws.close_reason or "").lower(), (
        "the shell socket must tell the client the connection is off, not silently "
        "open a session on it"
    )


# --- Narrow: no grants yet must stay reachable by admins (#27581) ---


@pytest.mark.asyncio
async def test_admin_reaches_connection_without_grants_when_bypass_off(
    access_control_module, config_module
):
    connection = _connection(grants=[])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        allowed = await access_control_module.has_connection_access(ADMIN, connection, set())
    assert allowed is True, (
        "a connection that has no access grants yet became unreachable by the admin "
        "who created it, so nobody could open it to add grants (#27581)"
    )


@pytest.mark.asyncio
async def test_non_admin_refused_connection_without_grants_when_bypass_off(
    access_control_module, config_module
):
    connection = _connection(grants=[])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        allowed = await access_control_module.has_connection_access(MEMBER, connection, set())
    assert allowed is False, (
        "restoring admin access must not hand an ungranted connection to ordinary users"
    )


# --- Broad: the invariants behind both fixes ---


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
async def test_no_terminal_entry_point_skips_the_enabled_check(
    entry_point, terminals_router_module, tools_module
):
    """Every path that resolves a connection by id, not just the listing."""
    connection = _connection(enabled=False, grants=[_read_grant(MEMBER.id)])
    allowed = await ENTRY_POINTS[entry_point](
        terminals_router_module, tools_module, MEMBER, connection
    )
    assert allowed is False, (
        f"{entry_point} resolves a terminal connection by id without consulting the "
        "enabled flag, so turning a server off does not actually take it out of service"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param({}, id="no_config"),
        pytest.param({"config": None}, id="config_none"),
        pytest.param({"config": {}}, id="no_grants_key"),
        pytest.param({"config": {"access_grants": []}}, id="empty_grants"),
        pytest.param({"config": {"access_grants": None}}, id="grants_none"),
    ],
)
async def test_every_empty_grants_shape_resolves_to_admin_only(
    shape, access_control_module, config_module
):
    """A connection stores its grants in several equivalent empty shapes; all of
    them mean private, so all of them must answer the same way."""
    connection = {"id": SERVER_ID, "url": "", "enabled": True, **shape}
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        admin_allowed = await access_control_module.has_connection_access(ADMIN, connection, set())
        member_allowed = await access_control_module.has_connection_access(
            MEMBER, connection, set()
        )
    assert (admin_allowed, member_allowed) == (True, False), (
        f"a connection whose grants are stored as {shape} must be admin-only, not "
        "unreachable by everyone (#27581)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
async def test_admin_reaches_ungranted_connection_on_every_entry_point(
    entry_point, terminals_router_module, tools_module, config_module
):
    connection = _connection(grants=[])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        allowed = await ENTRY_POINTS[entry_point](
            terminals_router_module, tools_module, ADMIN, connection
        )
    assert allowed is True, (
        f"{entry_point} locked the admin out of a connection that has no grants yet, "
        "which is every connection right after it is added (#27580)"
    )


# --- Nearby: behaviour that was already correct and must stay correct ---


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
async def test_enabled_connection_still_works_on_every_entry_point(
    entry_point, terminals_router_module, tools_module
):
    connection = _connection(enabled=True, grants=[_read_grant(MEMBER.id)])
    allowed = await ENTRY_POINTS[entry_point](
        terminals_router_module, tools_module, MEMBER, connection
    )
    assert allowed is True, (
        f"{entry_point} refused an enabled connection the user was granted, so the "
        "disabled-server fix over-corrected"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
async def test_missing_enabled_flag_defaults_to_enabled(
    entry_point, terminals_router_module, tools_module
):
    """Connections saved before the flag existed have no `enabled` key."""
    connection = _connection(enabled=True, grants=[_read_grant(MEMBER.id)])
    del connection["enabled"]
    allowed = await ENTRY_POINTS[entry_point](
        terminals_router_module, tools_module, MEMBER, connection
    )
    assert allowed is True, (
        f"{entry_point} treated a connection with no enabled key as off, taking "
        "existing terminal servers out of service on upgrade"
    )


@pytest.mark.asyncio
async def test_granted_non_admin_reaches_connection_when_bypass_off(
    access_control_module, config_module
):
    connection = _connection(grants=[_read_grant(MEMBER.id)])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        allowed = await access_control_module.has_connection_access(MEMBER, connection, set())
    assert allowed is True


@pytest.mark.asyncio
async def test_ungranted_non_admin_refused_when_bypass_off(access_control_module, config_module):
    connection = _connection(grants=[_read_grant("someone-else")])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", False):
        allowed = await access_control_module.has_connection_access(MEMBER, connection, set())
    assert allowed is False


@pytest.mark.asyncio
async def test_admin_bypass_on_is_unchanged(access_control_module, config_module):
    connection = _connection(grants=[_read_grant("someone-else")])
    with patch.object(config_module, "BYPASS_ADMIN_ACCESS_CONTROL", True):
        allowed = await access_control_module.has_connection_access(ADMIN, connection, set())
    assert allowed is True
