"""Regression: a deactivated account kept its live WebSocket access.

open-webui 0.11.0 fix `f517cc717` (PR #27537): the Socket.IO handlers (`connect`, `user-join`,
`join-channels`, `join-note`) and the terminal WebSocket each authenticated a raw token as
decode, revocation check and user lookup, without the role check `get_verified_user` applies on
every HTTP route. An account moved to `pending` got 401 over HTTP while its unexpired token still
opened a socket session, its channel and note rooms and a terminal. The fix resolves every
WebSocket token through `get_verified_user_by_token`, which applies the HTTP role set.

Each case runs twice, for a `pending` account and, as the positive control, a verified one. The
socket is opened without a token wherever a handler checks the token itself, so a failure names
that handler and not the handshake.

Twin of unit/security/test_websocket_role_gate.py.

Discriminates: passes on dev bbfa876af, fails with the role check removed from
`get_verified_user_by_token` (every pending case is admitted: the terminal gets past
authentication, `user-join` acks the account, the note and channel events arrive and the
handshake session serves the note's document); bypassing the check in one socket handler alone
fails only that handler's test.
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
import socketio
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

VERIFIED_ROLES = ("user", "admin")
VERIFIED_ONLY_ROUTE = "/api/v1/chats/"
TERMINAL_URL = "/api/v1/terminals/no-such-server/api/terminals/session-1"
REFUSED = (4001, "Invalid token")
# Past authentication, stopped at the server lookup because none is configured.
PAST_AUTHENTICATION = (4004, "Terminal server not found")
ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0


def _set_role(admin, account, role: str) -> None:
    with admin.client() as client:
        updated = client.post(f"/api/v1/users/{account.id}/update", json={"role": role})
    assert updated.status_code == 200, f"setting role {role!r} failed: {updated.text}"


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": "private note", "data": {"content": {"md": "only mine"}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _terminal_close(instance, first_message: dict) -> tuple[int, str]:
    url = instance.base_url.replace("http", "ws", 1) + TERMINAL_URL
    with connect(url, open_timeout=30) as terminal:
        terminal.send(json.dumps(first_message))
        with pytest.raises(ConnectionClosed) as closed:
            terminal.recv(timeout=30)
    return closed.value.rcvd.code, closed.value.rcvd.reason


@contextmanager
def _socket(
    instance, token: str | None = None, listen: tuple[str, ...] = ()
) -> Iterator[tuple[socketio.Client, dict[str, threading.Event]]]:
    """A Socket.IO connection like a browser tab's, flagging each `listen` event on arrival."""
    client = socketio.Client(reconnection=False)
    arrived = {event: threading.Event() for event in listen}
    for event, flag in arrived.items():
        client.on(event, lambda *_, flag=flag: flag.set())
    client.connect(
        instance.base_url,
        socketio_path="/ws/socket.io",
        transports=["websocket"],
        auth={"token": token} if token else None,
        wait_timeout=30,
    )
    try:
        yield client, arrived
    finally:
        client.disconnect()


def _arrives(flag: threading.Event, expected: bool) -> bool:
    # A refused account is given a bounded quiet period, an admitted one the full timeout.
    return flag.wait(ARRIVAL_TIMEOUT if expected else QUIET_PERIOD)


@pytest.mark.parametrize("role", ["pending", "user"])
def test_the_terminal_websocket_authenticates_only_verified_accounts(
    instance, admin, make_user, role
):
    account = make_user()
    _set_role(admin, account, role)

    closed = _terminal_close(instance, {"type": "auth", "token": account.token})

    expected = PAST_AUTHENTICATION if role in VERIFIED_ROLES else REFUSED
    assert closed == expected, (
        f"a {role} account's token closed the terminal socket with {closed}, expected "
        f"{expected}: a deactivated account still opens terminal sessions (#27537)"
    )


def test_the_terminal_websocket_wants_an_auth_message_first(instance):
    closed = _terminal_close(instance, {"type": "input", "data": "ls"})
    assert closed == (4001, "Expected auth message")


@pytest.mark.parametrize("role", ["pending", "user", "admin"])
def test_user_join_admits_exactly_the_roles_http_admits(instance, admin, make_user, role):
    account = make_user()
    _set_role(admin, account, role)
    verified = role in VERIFIED_ROLES

    with account.client() as client:
        http_status = client.get(VERIFIED_ONLY_ROUTE).status_code
    with _socket(instance) as (client, _):
        joined = client.call("user-join", {"auth": {"token": account.token}}, timeout=30)

    assert http_status == (200 if verified else 401)
    assert joined == ({"id": account.id, "name": account.name} if verified else None), (
        f"user-join answered {joined} for a {role} account that HTTP answers {http_status}: "
        "the socket and the API disagree on who is signed in (#27537)"
    )


@pytest.mark.parametrize("role", ["pending", "user"])
def test_the_handshake_gives_only_a_verified_account_a_session(instance, admin, make_user, role):
    """The collaborative document handlers serve whoever the handshake admitted."""
    owner = make_user()
    note_id = _create_note(owner)
    _set_role(admin, owner, role)
    verified = role in VERIFIED_ROLES

    with _socket(instance, owner.token, listen=("ydoc:document:state",)) as (client, arrived):
        client.call("ydoc:document:join", {"document_id": f"note:{note_id}"}, timeout=30)
        served = _arrives(arrived["ydoc:document:state"], verified)

    assert served == verified, (
        f"a socket opened with a {role} account's token was "
        f"{'served' if served else 'refused'} its note's document (#27537)"
    )


@pytest.mark.parametrize("role", ["pending", "user"])
def test_join_note_joins_only_a_verified_account(instance, admin, make_user, role):
    owner = make_user()
    note_id = _create_note(owner)
    _set_role(admin, owner, role)
    verified = role in VERIFIED_ROLES

    with _socket(instance, listen=("events:note",)) as (client, arrived):
        client.call("join-note", {"auth": {"token": owner.token}, "note_id": note_id}, timeout=30)
        with admin.client() as admin_client:
            edited = admin_client.post(
                f"/api/v1/notes/{note_id}/update", json={"title": "edited by the admin"}
            )
        assert edited.status_code == 200, edited.text
        received = _arrives(arrived["events:note"], verified)

    assert received == verified, (
        f"after join-note with a {role} account's token the live edits of its note "
        f"{'arrived' if received else 'did not arrive'} (#27537)"
    )


@pytest.mark.parametrize("role", ["pending", "user"])
def test_join_channels_joins_only_a_verified_account(instance, admin, make_user, preserve, role):
    preserve("admin_config")
    member = make_user()
    with admin.client() as admin_client:
        config = admin_client.get("/api/v1/auths/admin/config").json()
        admin_client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_CHANNELS": True}
        ).raise_for_status()
        channel = admin_client.post(
            "/api/v1/channels/create",
            json={"name": f"gate-{uuid.uuid4().hex[:8]}", "type": "group", "user_ids": [member.id]},
        )
    assert channel.status_code == 200, channel.text
    channel_id = channel.json()["id"]
    _set_role(admin, member, role)
    verified = role in VERIFIED_ROLES

    with _socket(instance, listen=("events:channel",)) as (client, arrived):
        client.call("join-channels", {"auth": {"token": member.token}}, timeout=30)
        with admin.client() as admin_client:
            posted = admin_client.post(
                f"/api/v1/channels/{channel_id}/messages/post",
                json={"content": "hello members"},
            )
        assert posted.status_code == 200, posted.text
        received = _arrives(arrived["events:channel"], verified)

    assert received == verified, (
        f"after join-channels with a {role} account's token its channel's messages "
        f"{'arrived' if received else 'did not arrive'} (#27537)"
    )


@pytest.mark.parametrize("credential", ["deleted account", "forged signature", "not a token"])
def test_a_token_of_no_live_account_opens_nothing(instance, admin, make_user, credential):
    account = make_user()
    token = account.token
    if credential == "deleted account":
        with admin.client() as client:
            client.delete(f"/api/v1/users/{account.id}").raise_for_status()
    elif credential == "forged signature":
        token = token.rsplit(".", 1)[0] + ".forged-signature"
    else:
        token = "not-a-jwt"

    with _socket(instance) as (client, _):
        joined = client.call("user-join", {"auth": {"token": token}}, timeout=30)

    assert joined is None
    assert _terminal_close(instance, {"type": "auth", "token": token}) == REFUSED
