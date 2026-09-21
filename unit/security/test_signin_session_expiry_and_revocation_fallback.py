"""Regression: an SSO session expired with whichever of its two tokens ran out first,
and a Redis outage made every request with a signed-in user fail.

open-webui 0.11.4 fixes `aaaf26fb8ede28854dd660b11b85ba6bafe64007` and
`c1615bec2f5f143084b5fcc04f42ebfa0dade9df` + `6a85abb3f5e002f3069007213e16e4ef60776890`:

- `aaaf26fb8`: `_normalize_token_expiry` capped the session's stored `expires_at` at the
  id_token's own `exp`, so the session record expired with whichever of its two tokens ran
  out first; an id_token shorter than the access token renewed the session early and
  re-capped it after every renewal. The stored value now tracks the access token the
  session actually calls with, and `get_oauth_token` applies the id_token cap at read time
  instead, when it decides whether to refresh before handing the token to an SSO
  integration.
- `c1615bec` + `6a85abb3`: `is_valid_token` dereferenced the redis client with no error
  handling, so a Redis outage turned every authenticated request into a 500. It now
  accepts the token when Redis raises, logging one rate-limited warning, so signing out
  may not take effect until Redis is back.

Discriminates: passes on v0.11.4 (344ea5306), fails on v0.11.4~ (aaaf26fb8^ /
c1615bec^): pre-fix `_normalize_token_expiry` caps the stored expiry at the id_token exp,
`get_oauth_token` refreshes only on the stored value so a session with a near-expiry
id_token is handed out stale to SSO integrations, and a Redis outage raises out of
`is_valid_token` instead of accepting the token.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt as pyjwt
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"
SECRET = "unit-test-key-of-a-decent-length-0123456789"


@pytest.fixture(scope="module")
def oauth_module(owui_module):
    return owui_module("open_webui.utils.oauth")


@pytest.fixture(scope="module")
def auth_utils(owui_module):
    return owui_module("open_webui.utils.auth")


def _id_token(exp: int | None) -> str:
    claims = {"sub": "u1"}
    if exp is not None:
        claims["exp"] = exp
    return pyjwt.encode(claims, SECRET, algorithm="HS256")


def _session(oauth_module, *, expires_at, id_token):
    return SimpleNamespace(
        id="s1",
        provider="oidc",
        expires_at=expires_at,
        token={"access_token": "live", "id_token": id_token, "refresh_token": "r"},
    )


def _manager(oauth_module):
    return oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))


# ── 4. session expiry accuracy (aaaf26fb8) ──────────────────────────────────


@pytest.mark.asyncio
async def test_id_token_near_expiry_triggers_the_read_time_refresh(oauth_module):
    """NARROW: the refresh decision caps at the id_token exp at read time, so an SSO
    integration is never handed a session whose id_token is about to die, even though
    the stored expiry is the access token's own."""
    now = int(time.time())
    session = _session(oauth_module, expires_at=now + 7200, id_token=_id_token(now + 60))
    manager = _manager(oauth_module)
    refreshed = {"access_token": "fresh"}
    with (
        patch.object(
            oauth_module.OAuthSessions,
            "get_session_by_id_and_user_id",
            AsyncMock(return_value=session),
        ),
        patch.object(manager, "_refresh_token", AsyncMock(return_value=refreshed)) as refresh,
    ):
        result = await manager.get_oauth_token(ALICE, "s1")

    assert refresh.await_count == 1, "a near-expiry id_token did not trigger a refresh"
    assert result == refreshed


@pytest.mark.asyncio
async def test_id_token_outside_the_refresh_window_is_left_alone(oauth_module):
    """NEARBY: the read-time cap respects the 5-minute window; a distant id_token expiry
    does not refresh a session whose access token is live."""
    now = int(time.time())
    session = _session(oauth_module, expires_at=now + 7200, id_token=_id_token(now + 600))
    manager = _manager(oauth_module)
    with (
        patch.object(
            oauth_module.OAuthSessions,
            "get_session_by_id_and_user_id",
            AsyncMock(return_value=session),
        ),
        patch.object(manager, "_refresh_token", AsyncMock()) as refresh,
    ):
        result = await manager.get_oauth_token(ALICE, "s1")

    assert refresh.await_count == 0
    assert result == session.token


@pytest.mark.asyncio
async def test_access_token_at_edge_of_expiry_still_refreshes(oauth_module):
    """NEARBY: the access token's own expiry still triggers the refresh it always did."""
    now = int(time.time())
    session = _session(oauth_module, expires_at=now + 300, id_token=_id_token(now + 7200))
    manager = _manager(oauth_module)
    refreshed = {"access_token": "fresh"}
    with (
        patch.object(
            oauth_module.OAuthSessions,
            "get_session_by_id_and_user_id",
            AsyncMock(return_value=session),
        ),
        patch.object(manager, "_refresh_token", AsyncMock(return_value=refreshed)) as refresh,
    ):
        result = await manager.get_oauth_token(ALICE, "s1")

    assert refresh.await_count == 1, "an access token inside the refresh window was not refreshed"
    assert result == refreshed


def test_normalize_token_expiry_no_longer_caps_at_the_id_token_exp(oauth_module):
    """NARROW: the stored expiry is the access token's own; the id_token no longer caps it."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry(
        {"expires_in": 7200, "id_token": _id_token(now + 600), "refresh_token": "r"}
    )

    assert abs(result["expires_at"] - (now + 7200)) <= 5, (
        "the stored expiry was capped at the id_token exp"
    )


def test_normalize_token_expiry_with_only_an_id_token_outliving_it(oauth_module):
    """NEARBY: the cap is gone in both directions; a longer id_token does not extend anything."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry(
        {"expires_in": 600, "id_token": _id_token(now + 7200)}
    )

    assert abs(result["expires_at"] - (now + 600)) <= 5


# ── 5. revocation list fallback (c1615bec, 6a85abb3) ────────────────────────


class _BrokenRedis:
    """A redis client that cannot be reached, the way an outage looks to the call."""

    def __init__(self, error):
        self._error = error

    async def get(self, key):
        raise self._error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [RedisConnectionError("connection refused"), RedisTimeoutError("timed out")],
    ids=["connection-refused", "timeout"],
)
async def test_unreachable_redis_accepts_the_token(auth_utils, error):
    """NARROW: a Redis outage accepts the token instead of failing the request."""
    decoded = {"id": ALICE, "jti": "jti-1", "iat": int(time.time())}

    assert await auth_utils.is_valid_token(decoded, _BrokenRedis(error)) is True


@pytest.mark.asyncio
async def test_live_revocation_still_rejects_the_token(auth_utils):
    """NEARBY: with Redis up, a revoked jti is still refused, fallback or not."""
    decoded = {"id": ALICE, "jti": "jti-1", "iat": int(time.time())}
    redis = SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                "1",  # jti revoked marker
                None,  # (not reached)
            ]
        )
    )

    assert await auth_utils.is_valid_token(decoded, redis) is False


@pytest.mark.asyncio
async def test_no_redis_client_still_accepts_the_token(auth_utils):
    """NEARBY: a deployment without Redis keeps accepting tokens, as before."""
    decoded = {"id": ALICE, "jti": "jti-1", "iat": int(time.time())}

    assert await auth_utils.is_valid_token(decoded, None) is True
