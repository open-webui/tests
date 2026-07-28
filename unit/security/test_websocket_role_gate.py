"""Regression: deactivating an account must also cut its live WebSocket access.

open-webui 0.11.0 fix `f517cc717` (PR #27537): the Socket.IO handshake in
`socket/main.py` and the terminal WebSocket in `routers/terminals.py` each
reimplemented JWT authentication as `decode_token` + `is_valid_token` +
`Users.get_user_by_id`, then accepted whatever user came back. Neither applied
the role check `get_verified_user` enforces on every HTTP route, so an account
demoted out of `user`/`admin` (deactivated, moved to `pending`) got 401 over
HTTP while the very same JWT still opened channels, notes and terminals until
it expired, four weeks by default. The fix resolves the user once in
`get_verified_user_by_token` and routes both entry points through it, with the
role set hoisted into `VERIFIED_USER_ROLES` so the two gates cannot drift.

Discriminates: passes on v0.11.0, fails on v0.10.2 (a `pending` token still
authenticates on both WebSocket entry points).
"""

import inspect
import json
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from unit.security.conftest import FakeWebSocket

pytestmark = pytest.mark.regression

TERMINAL_SERVER = {"id": "term-1", "name": "shell", "enabled": True}


@pytest.fixture(scope="session")
def auth_module(owui_module):
    return owui_module("open_webui.utils.auth")


@pytest.fixture(scope="session")
def users_module(owui_module):
    return owui_module("open_webui.models.users")


@pytest.fixture(scope="session")
def socket_module(owui_module):
    return owui_module("open_webui.socket.main")


@pytest.fixture(scope="session")
def terminals_module(owui_module):
    return owui_module("open_webui.routers.terminals")


def _user(users_module, role: str, id: str = "alice"):
    now = int(time.time())
    return users_module.UserModel(
        id=id,
        email=f"{id}@example.com",
        name=id,
        role=role,
        last_active_at=now,
        updated_at=now,
        created_at=now,
    )


def _patch_user_row(users_module, user):
    """The DB row is the only I/O boundary here; the JWT itself is real."""
    return patch.object(
        users_module.Users, "get_user_by_id", AsyncMock(return_value=user)
    )


# --- narrow: the helper itself ---------------------------------------------


@pytest.mark.asyncio
async def test_pending_role_token_resolves_to_nothing(auth_module, users_module):
    """The bug: a valid, unrevoked token belonging to a deactivated account."""
    token = auth_module.create_token({"id": "alice"})
    with _patch_user_row(users_module, _user(users_module, "pending")):
        resolved = await auth_module.get_verified_user_by_token(token)
    assert resolved is None, (
        "a deactivated (pending) account still authenticated over WebSocket, so "
        "demoting it left its channels and notes open until the token expired (#27537)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "admin"])
async def test_verified_roles_still_resolve(auth_module, users_module, role):
    token = auth_module.create_token({"id": "alice"})
    with _patch_user_row(users_module, _user(users_module, role)):
        resolved = await auth_module.get_verified_user_by_token(token)
    assert resolved is not None and resolved.role == role


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    ["", "not-a-jwt", "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6ImFsaWNlIn0.wrong-signature"],
)
async def test_malformed_token_returns_none_without_raising(
    auth_module, users_module, token
):
    with _patch_user_row(users_module, _user(users_module, "admin")):
        assert await auth_module.get_verified_user_by_token(token) is None


@pytest.mark.asyncio
async def test_expired_token_returns_none(auth_module, users_module):
    token = auth_module.create_token({"id": "alice"}, timedelta(seconds=-60))
    with _patch_user_row(users_module, _user(users_module, "admin")):
        assert await auth_module.get_verified_user_by_token(token) is None


@pytest.mark.asyncio
async def test_deleted_user_row_returns_none(auth_module, users_module):
    token = auth_module.create_token({"id": "alice"})
    with _patch_user_row(users_module, None):
        assert await auth_module.get_verified_user_by_token(token) is None


@pytest.mark.asyncio
async def test_token_without_id_claim_returns_none(auth_module, users_module):
    token = auth_module.create_token({"email": "alice@example.com"})
    with _patch_user_row(users_module, _user(users_module, "admin")):
        assert await auth_module.get_verified_user_by_token(token) is None


# --- narrow: the two entry points, behaviourally ---------------------------


@pytest.mark.asyncio
async def test_socket_connect_rejects_pending_role(
    auth_module, users_module, socket_module
):
    """Before the fix the handshake pooled the session and joined the user room."""
    mod = socket_module
    session_pool = {}
    token = auth_module.create_token({"id": "alice"})
    sio_stub = SimpleNamespace(save_session=AsyncMock(), enter_room=AsyncMock())

    with (
        _patch_user_row(users_module, _user(users_module, "pending")),
        patch.object(mod, "SESSION_POOL", session_pool),
        patch.object(mod, "sio", sio_stub),
    ):
        await mod.connect("sid-1", {}, {"token": token})

    assert session_pool == {}, (
        "the Socket.IO handshake accepted a deactivated account, giving it a live "
        "session it can use to read and write channels and shared notes (#27537)"
    )
    sio_stub.enter_room.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_websocket_rejects_pending_role(
    auth_module, users_module, terminals_module
):
    """The terminal proxy reached server resolution for a deactivated account."""
    mod = terminals_module
    token = auth_module.create_token({"id": "alice"})
    ws = FakeWebSocket(token)

    with (
        _patch_user_row(users_module, _user(users_module, "pending")),
        patch.object(mod.Config, "get", AsyncMock(return_value=[TERMINAL_SERVER])),
        patch.object(mod.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(mod, "has_connection_access", AsyncMock(return_value=True)),
    ):
        resolved = await mod._resolve_authenticated_connection(ws, TERMINAL_SERVER["id"])

    assert resolved is None, (
        "a deactivated account still opened an interactive terminal session on a "
        "server it had access to before demotion (#27537)"
    )
    assert ws.close_code == 4001
    assert ws.close_reason == "Invalid token"


# --- broad: no entry point may reimplement the token dance -----------------


def _entry_point_sources(socket_module, terminals_module):
    return {
        "socket.connect": socket_module.connect,
        "socket.user_join": socket_module.user_join,
        "socket.join_channel": socket_module.join_channel,
        "socket.join_note": socket_module.join_note,
        "terminals._resolve_authenticated_connection": (
            terminals_module._resolve_authenticated_connection
        ),
    }


@pytest.mark.parametrize(
    "handler_name",
    [
        "socket.connect",
        "socket.user_join",
        "socket.join_channel",
        "socket.join_note",
        "terminals._resolve_authenticated_connection",
    ],
)
def test_every_websocket_entry_point_uses_the_shared_gate(
    socket_module, terminals_module, handler_name
):
    """A third entry point that re-decodes the token would fail open again."""
    source = inspect.getsource(
        _entry_point_sources(socket_module, terminals_module)[handler_name]
    )
    assert "get_verified_user_by_token" in source, (
        f"{handler_name} authenticates outside the shared verified-user gate, so a "
        "role change does not reach it (#27537)"
    )
    assert "decode_token" not in source and "is_valid_token" not in source, (
        f"{handler_name} still decodes the token itself, which is how the role check "
        "was missed in the first place (#27537)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["pending", "user", "admin", "guest", ""])
async def test_websocket_gate_matches_the_http_gate(auth_module, users_module, role):
    """The two planes must accept exactly the same roles, or they drift again."""
    from fastapi import HTTPException

    user = _user(users_module, role)
    try:
        auth_module.get_verified_user(user)
        http_allows = True
    except HTTPException as error:
        assert error.status_code == 401
        http_allows = False

    token = auth_module.create_token({"id": user.id})
    with _patch_user_row(users_module, user):
        websocket_allows = await auth_module.get_verified_user_by_token(token) is not None

    assert websocket_allows == http_allows, (
        f"role {role!r} is treated differently over HTTP and over WebSocket, which is "
        "exactly the split that let deactivated accounts keep live access (#27537)"
    )


# --- nearby: behaviour that must hold on both refs -------------------------


@pytest.mark.asyncio
async def test_terminal_websocket_rejects_non_auth_first_message(terminals_module):
    ws = FakeWebSocket("")
    ws._first_message = json.dumps({"type": "input", "data": "ls"})
    resolved = await terminals_module._resolve_authenticated_connection(ws, "term-1")
    assert resolved is None
    assert ws.close_code == 4001
    assert ws.close_reason == "Expected auth message"


@pytest.mark.asyncio
async def test_terminal_websocket_admits_a_verified_user(
    auth_module, users_module, terminals_module
):
    """Positive path: the fix must not lock out ordinary users."""
    mod = terminals_module
    ws = FakeWebSocket(auth_module.create_token({"id": "alice"}))
    user = _user(users_module, "user")

    with (
        _patch_user_row(users_module, user),
        patch.object(mod.Config, "get", AsyncMock(return_value=[TERMINAL_SERVER])),
        patch.object(mod.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(mod, "has_connection_access", AsyncMock(return_value=True)),
    ):
        resolved = await mod._resolve_authenticated_connection(ws, TERMINAL_SERVER["id"])

    assert resolved is not None
    assert resolved[0].id == user.id
    assert resolved[1]["id"] == TERMINAL_SERVER["id"]


@pytest.mark.asyncio
async def test_socket_connect_admits_a_verified_user(
    auth_module, users_module, socket_module
):
    mod = socket_module
    session_pool = {}
    sio_stub = SimpleNamespace(save_session=AsyncMock(), enter_room=AsyncMock())

    with (
        _patch_user_row(users_module, _user(users_module, "user")),
        patch.object(mod, "SESSION_POOL", session_pool),
        patch.object(mod, "sio", sio_stub),
    ):
        await mod.connect("sid-1", {}, {"token": auth_module.create_token({"id": "alice"})})

    assert session_pool["sid-1"]["id"] == "alice"
    sio_stub.enter_room.assert_awaited_once_with("sid-1", "user:alice")


@pytest.mark.asyncio
async def test_socket_connect_without_auth_is_a_no_op(socket_module):
    mod = socket_module
    session_pool = {}
    with (
        patch.object(mod, "SESSION_POOL", session_pool),
        patch.object(mod, "sio", SimpleNamespace(save_session=AsyncMock(), enter_room=AsyncMock())),
    ):
        await mod.connect("sid-1", {}, None)
    assert session_pool == {}
