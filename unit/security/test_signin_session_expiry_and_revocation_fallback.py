"""Regression: an SSO session expired with whichever of its two tokens ran out first.

open-webui 0.11.4 fix `aaaf26fb8` (following c055203f2, #27520): `_normalize_token_expiry`
capped the stored `expires_at` at the id_token's own `exp`, so an id_token shorter than the
access token renewed the session early and re-capped it after every renewal. The stored value
now tracks the access token, and `get_oauth_token` applies the id_token cap when it decides
whether to refresh before handing the tokens to an SSO integration. Nothing reaches this over
HTTP without waiting out a token, so it is driven here against a real OAuth session in an
in-memory database and a local token endpoint.

The Redis half of this file (c1615bec2, 6a85abb3f: an outage fails open, a live sign-out still
revokes) moved to integration/security/test_signin_session_expiry_and_revocation_fallback.py.

Discriminates: passes on dev bbfa876af; with aaaf26fb8 reverted in a backend copy the session
with a near-expiry id_token is handed out without a refresh and the stored expiry is capped
at the id_token's.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import patch

import jwt
import pytest
from fastapi import FastAPI

from harness.listener import json_answer, listening
from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"


def id_token_expiring_in(seconds: int) -> str:
    claims = {"sub": "u1", "exp": int(time.time()) + seconds}
    return jwt.encode(claims, "unit-test-key-of-a-decent-length-0123456789", algorithm="HS256")


@pytest.fixture
def token_endpoint():
    """The provider's discovery document and token endpoint, recording every refresh."""
    with listening() as provider:
        discovery = {"issuer": provider.base_url, "token_endpoint": f"{provider.base_url}/token"}
        provider.route("GET", "/.well-known/openid-configuration", json_answer(discovery))
        provider.route(
            "POST", "/token", json_answer({"access_token": "refreshed", "expires_in": 3600})
        )
        yield provider


@asynccontextmanager
async def signed_in_session(owui_module, provider, *, expires_in: int, id_token: str):
    """A real OAuthManager and one stored SSO session whose tokens live as long as asked."""
    oauth = owui_module("open_webui.utils.oauth")
    oauth_sessions = owui_module("open_webui.models.oauth_sessions")
    config = owui_module("open_webui.models.config").Config

    def register(registry):
        discovery_url = f"{provider.base_url}/.well-known/openid-configuration"
        return registry.register(
            name="oidc", client_id="owui", client_secret="secret", server_metadata_url=discovery_url
        )

    app = FastAPI()
    app.state.redis = None
    token = {
        "access_token": "stored",
        "refresh_token": "r",
        "id_token": id_token,
        "expires_at": int(time.time()) + expires_in,
    }
    async with memory_database(owui_module, oauth_sessions.OAuthSession, config):
        with patch.dict(oauth.OAUTH_PROVIDERS, {"oidc": {"register": register}}, clear=True):
            manager = oauth.OAuthManager(app=app)
            stored = await oauth_sessions.OAuthSessions.create_session(
                user_id=ALICE, provider="oidc", token=token
            )
            yield manager, stored


@pytest.mark.parametrize(
    ("access_token_ttl", "id_token_ttl", "refreshed"),
    [(7200, 60, True), (7200, 600, False), (300, 7200, True), (7200, None, False)],
    ids=["id-token-about-to-expire", "both-live", "access-token-about-to-expire", "unreadable"],
)
@pytest.mark.asyncio
async def test_a_session_is_refreshed_exactly_when_a_token_it_hands_out_is_expiring(
    owui_module, token_endpoint, access_token_ttl, id_token_ttl, refreshed
):
    """Narrow (id-token-about-to-expire): the id_token's exp counts at hand-out time, so an SSO
    integration never gets a dying id_token. Nearby: a live pair and an unreadable id_token are
    handed out as they are, and the access token's own expiry still refreshes."""
    id_token = id_token_expiring_in(id_token_ttl) if id_token_ttl else "not-a-jwt"
    async with signed_in_session(
        owui_module, token_endpoint, expires_in=access_token_ttl, id_token=id_token
    ) as (manager, stored):
        handed_out = await manager.get_oauth_token(user_id=ALICE, session_id=stored.id)

    assert handed_out["access_token"] == ("refreshed" if refreshed else "stored")
    assert len(token_endpoint.requests_to("/token")) == int(refreshed)


@pytest.mark.parametrize("id_token_ttl", [600, 7200], ids=["shorter-id-token", "longer-id-token"])
def test_the_stored_expiry_is_the_access_tokens_own(owui_module, id_token_ttl):
    """Narrow (shorter): the id_token no longer caps the stored expiry. Nearby (longer): nor
    does it ever extend it."""
    oauth = owui_module("open_webui.utils.oauth")
    token = oauth._normalize_token_expiry(
        {"expires_in": 7200, "id_token": id_token_expiring_in(id_token_ttl), "refresh_token": "r"}
    )
    assert token["expires_at"] == pytest.approx(time.time() + 7200, abs=5)
