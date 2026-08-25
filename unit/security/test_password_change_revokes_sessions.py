"""Regression: changing a password left every other signed-in device working.

open-webui 0.11.1 fix `21e390561` (#28725): a password change wrote nothing to the
token revocation list, so every session issued before the change kept working until
its JWT expired on its own, up to four weeks on the default `JWT_EXPIRES_IN=4w`.
Token validation already understood a per-user `auth:user:<id>:revoked_at` marker,
but only sign-out and OIDC back-channel logout ever wrote one. Both password paths,
self-service `PUT /api/v1/auths/update/password` and an admin resetting someone's
password through `POST /api/v1/users/{id}/update`, now stamp that marker through the
shared `revoke_user_tokens` helper.

The marker lives in Redis. Without Redis nothing can be revoked, and the point of the
fix is that this is now said out loud: the backend logs a warning instead of letting a
security control silently no-op. That is the case the third narrow test pins.

Tests drive the real router coroutines against the real `revoke_user_tokens` and the
real `is_valid_token`, with only the database layer mocked, so the assertions are on
whether a token issued before the change still authenticates.

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix the routers never touch
the revocation list, so the old token stays valid and no warning is logged).
"""

from __future__ import annotations

import ast
import logging
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.regression

NEW_PASSWORD = 'NewPassw0rd!'
OLD_PASSWORD = 'OldPassw0rd!'


class FakeRedis:
    """Records what the revocation helper writes, and replays it to `is_valid_token`."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int | None] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.expiries[key] = ex


@pytest.fixture(scope='module')
def auth_utils(owui_module):
    # Importing config runs the alembic migrations, so `Config.get` has a table to read.
    owui_module('open_webui.config')
    return owui_module('open_webui.utils.auth')


@pytest.fixture(scope='module')
def auths_router(owui_module):
    return owui_module('open_webui.routers.auths')


@pytest.fixture(scope='module')
def users_router(owui_module):
    return owui_module('open_webui.routers.users')


@pytest.fixture
def user_id():
    """Unique id so tests sharing a Redis stub or the scratch database cannot collide."""
    return f'user-{uuid4().hex[:12]}'


def _request(redis):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=redis)))


def _revocation_key(auth_utils, user_id: str) -> str:
    return f'{auth_utils.REDIS_KEY_PREFIX}:auth:user:{user_id}:revoked_at'


def _issue_token(auth_utils, user_id: str) -> dict:
    """A session token as a signed-in device holds it, decoded the way the gate sees it."""
    return auth_utils.decode_token(
        auth_utils.create_token({'id': user_id}, timedelta(weeks=4)),
    )


async def _noop(*args, **kwargs):
    return None


async def change_own_password(auths_router, request, user_id, *, write_succeeds=True):
    """Drive the self-service password change with only the database layer mocked."""
    user = SimpleNamespace(id=user_id, email=f'{user_id}@example.com', role='user')

    async def authenticate_user(*args, **kwargs):
        return user

    async def update_password(*args, **kwargs):
        return write_succeeds

    with (
        patch.object(auths_router.Auths, 'authenticate_user', authenticate_user),
        patch.object(auths_router.Auths, 'update_user_password_by_id', update_password),
        patch.object(auths_router, 'get_password_hash', _noop),
        patch.object(auths_router, 'publish_event', _noop),
    ):
        return await auths_router.update_password(
            request=request,
            form_data=auths_router.UpdatePasswordForm(
                password=OLD_PASSWORD, new_password=NEW_PASSWORD
            ),
            session_user=user,
            db=None,
        )


async def admin_resets_password(users_router, request, user_id, *, write_succeeds=True):
    """Drive the admin reset path with only the database layer mocked."""
    target = SimpleNamespace(id=user_id, email=f'{user_id}@example.com', role='user')
    admin = SimpleNamespace(id='acting-admin', email='admin@example.com', role='admin')

    async def get_first_user(*args, **kwargs):
        return SimpleNamespace(id='primary-admin')

    async def get_user_by_id(*args, **kwargs):
        return target

    async def update_password(*args, **kwargs):
        return write_succeeds

    with (
        patch.object(users_router.Users, 'get_first_user', get_first_user),
        patch.object(users_router.Users, 'get_user_by_id', get_user_by_id),
        patch.object(users_router.Auths, 'update_user_password_by_id', update_password),
        patch.object(users_router, 'get_password_hash', _noop),
        patch.object(users_router, 'publish_event', _noop),
    ):
        return await users_router.update_user_by_id(
            request=request,
            user_id=user_id,
            form_data=users_router.UserUpdateForm(password=NEW_PASSWORD),
            session_user=admin,
            db=None,
        )


# --- Narrow: the bug itself ---


@pytest.mark.asyncio
async def test_changing_your_own_password_stops_your_other_devices(
    auth_utils, auths_router, user_id
):
    """The reason people change a password: the old session must stop working."""
    redis = FakeRedis()
    old_session = _issue_token(auth_utils, user_id)

    await change_own_password(auths_router, _request(redis), user_id)

    assert await auth_utils.is_valid_token(old_session, redis) is False, (
        'a session issued before the password change still authenticates: whoever holds a '
        'token stolen under the old password keeps full access for the whole JWT lifetime, '
        'four weeks by default, and changing the password does nothing to stop them (#28725)'
    )


@pytest.mark.asyncio
async def test_an_admin_reset_stops_the_users_existing_devices(auth_utils, users_router, user_id):
    """An admin resetting a compromised account must cut the intruder off, not just the owner."""
    redis = FakeRedis()
    old_session = _issue_token(auth_utils, user_id)

    await admin_resets_password(users_router, _request(redis), user_id)

    assert await auth_utils.is_valid_token(old_session, redis) is False, (
        "an admin reset left the account's existing sessions valid: an admin resetting the "
        'password of an account they believe is compromised does not actually evict the '
        'intruder (#28725)'
    )


@pytest.mark.asyncio
async def test_without_redis_the_backend_says_nothing_was_revoked(
    auth_utils, auths_router, user_id, caplog
):
    """The worst outcome is a security control that silently no-ops, so it must be logged."""
    old_session = _issue_token(auth_utils, user_id)

    with caplog.at_level(logging.WARNING, logger=auth_utils.log.name):
        succeeded = await change_own_password(auths_router, _request(None), user_id)

    assert succeeded, 'the password change itself must still go through without Redis'
    assert await auth_utils.is_valid_token(old_session, None) is True, (
        'a deployment without Redis has nowhere to record the revocation, so the old session '
        'necessarily stays valid; that is exactly why the warning has to be emitted'
    )

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and 'redis' in record.getMessage().lower()
    ]
    assert warnings, (
        'changing a password on a deployment without Redis revoked nothing and logged nothing: '
        'the operator is left believing the other sessions were signed out when every one of '
        'them keeps working (#28725)'
    )
    assert any(user_id in message for message in warnings), (
        'the no-Redis warning does not name the user whose sessions were left alive, so an '
        'operator cannot tell which accounts are exposed (#28725)'
    )


# --- Broad: the contract around the revocation, for both entry points ---


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['self', 'admin'])
async def test_a_password_change_revokes_only_that_users_sessions(
    auth_utils, auths_router, users_router, user_id, path
):
    """Blast radius: one account's change must not sign the rest of the instance out."""
    redis = FakeRedis()
    bystander_id = f'bystander-{uuid4().hex[:12]}'
    bystander_session = _issue_token(auth_utils, bystander_id)

    if path == 'self':
        await change_own_password(auths_router, _request(redis), user_id)
    else:
        await admin_resets_password(users_router, _request(redis), user_id)

    assert list(redis.values) == [_revocation_key(auth_utils, user_id)], (
        f'the {path} password change wrote {sorted(redis.values)} to the revocation list; only '
        f'the marker for {user_id} belongs there'
    )
    assert await auth_utils.is_valid_token(bystander_session, redis) is True, (
        f'one user changing their password via the {path} path signed another user out'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['self', 'admin'])
async def test_nothing_is_revoked_when_the_password_write_fails(
    auth_utils, auths_router, users_router, user_id, path
):
    """No password change happened, so the sessions must survive."""
    redis = FakeRedis()
    session = _issue_token(auth_utils, user_id)

    if path == 'self':
        await change_own_password(auths_router, _request(redis), user_id, write_succeeds=False)
    else:
        await admin_resets_password(users_router, _request(redis), user_id, write_succeeds=False)

    assert redis.values == {}, (
        f'the {path} path revoked sessions after the password write reported failure, signing '
        'the user out of everything while their password is unchanged'
    )
    assert await auth_utils.is_valid_token(session, redis) is True


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['self', 'admin'])
async def test_the_revocation_marker_outlives_the_tokens_it_revokes(
    auth_utils, auths_router, users_router, owui_module, user_id, path
):
    """A marker that expires first hands every revoked token back its access."""
    redis = FakeRedis()
    parse_duration = owui_module('open_webui.utils.misc').parse_duration
    jwt_lifetime = parse_duration(await auth_utils.Config.get('auth.jwt_expiry'))

    if path == 'self':
        await change_own_password(auths_router, _request(redis), user_id)
    else:
        await admin_resets_password(users_router, _request(redis), user_id)

    ttl = redis.expiries.get(_revocation_key(auth_utils, user_id), 'no marker written')
    expected = int(jwt_lifetime.total_seconds()) if jwt_lifetime else None
    assert ttl == expected, (
        f'the {path} revocation marker expires after {ttl}s while tokens live for {expected}s: '
        'once the marker is gone every session it revoked authenticates again (#28725)'
    )


def test_every_password_write_in_the_routers_revokes(auths_router):
    """Guards the next password-writing route from shipping without the revocation."""
    routers_dir = Path(auths_router.__file__).resolve().parent
    call_sites = 0
    unrevoked = []
    for path in sorted(routers_dir.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            } | {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if 'update_user_password_by_id' not in called:
                continue
            call_sites += 1
            if 'revoke_user_tokens' not in called:
                unrevoked.append(f'{path.name}:{node.lineno} {node.name}')

    assert not unrevoked, (
        f'{unrevoked} writes a new password without revoking the existing sessions, so every '
        'device signed in under the old password keeps working (#28725)'
    )
    assert call_sites == 2, (
        f'expected the two known password-writing routes, found {call_sites}: a route moved or '
        'was added and is no longer covered by this check (#28725)'
    )


# --- Nearby: adjacent behaviour that must keep working ---


@pytest.mark.asyncio
async def test_signing_in_again_after_the_change_works(auth_utils, auths_router, user_id):
    """The revocation is a cut-off point, not a permanent lockout."""
    redis = FakeRedis()
    await change_own_password(auths_router, _request(redis), user_id)

    # The marker has one-second resolution and rejects `iat <= revoked_at`, so stand in
    # for a sign-in a second later rather than sleeping through it.
    revoked_at = await redis.get(_revocation_key(auth_utils, user_id))
    fresh_session = _issue_token(auth_utils, user_id)
    if revoked_at:
        fresh_session['iat'] = int(revoked_at) + 1

    assert await auth_utils.is_valid_token(fresh_session, redis) is True, (
        'the session created by signing in with the new password was rejected: the user can '
        'never get back in after changing their password'
    )


@pytest.mark.asyncio
async def test_a_wrong_current_password_revokes_nothing(auth_utils, auths_router, user_id):
    """Failing the current-password check must not become a way to sign someone out."""
    redis = FakeRedis()
    session = _issue_token(auth_utils, user_id)

    async def authenticate_user(*args, **kwargs):
        return None

    with (
        patch.object(auths_router.Auths, 'authenticate_user', authenticate_user),
        patch.object(auths_router, 'publish_event', _noop),
        pytest.raises(auths_router.HTTPException) as raised,
    ):
        await auths_router.update_password(
            request=_request(redis),
            form_data=auths_router.UpdatePasswordForm(
                password='wrong-password', new_password=NEW_PASSWORD
            ),
            session_user=SimpleNamespace(id=user_id, email=f'{user_id}@example.com'),
            db=None,
        )

    assert raised.value.status_code == 400
    assert redis.values == {}, (
        'a rejected password change still revoked the sessions: anyone who can reach the '
        'endpoint with a stolen session could sign the real user out at will'
    )
    assert await auth_utils.is_valid_token(session, redis) is True


@pytest.mark.asyncio
async def test_the_per_token_signout_revocation_still_works(auth_utils, user_id):
    """Sign-out revokes one jti; it shares the key namespace with the new per-user marker."""
    redis = FakeRedis()
    token = auth_utils.create_token({'id': user_id}, timedelta(weeks=4))
    decoded = auth_utils.decode_token(token)
    other_session = _issue_token(auth_utils, user_id)

    await auth_utils.invalidate_token(_request(redis), token)

    assert await auth_utils.is_valid_token(decoded, redis) is False
    assert await auth_utils.is_valid_token(other_session, redis) is True, (
        'signing out one device revoked the other sessions too'
    )


@pytest.mark.asyncio
async def test_token_validation_without_redis_is_unchanged(auth_utils, user_id):
    """No Redis means no revocation list to consult; validation must not start failing."""
    session = _issue_token(auth_utils, user_id)
    assert await auth_utils.is_valid_token(session, None) is True
