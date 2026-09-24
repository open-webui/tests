"""Regression: the OAuth sign-in lifecycle fixes the HTTP twin cannot reach.

integration/security/test_oauth_session_lifecycle.py (and its e2e twin) pin the OAuth switch,
the key rotation and the MCP authorize and callback flows over HTTP. What stays here:

- Profile-picture SSRF (#26699, commit 5dcca59ae): the provider-supplied picture was fetched with
  a plain `aiohttp.ClientSession` after `validate_url()` had vetted only the first DNS answer,
  so a host that re-resolved to an internal address between check and fetch was reached. The
  fetch now goes through `get_ssrf_safe_session()`, which re-checks the address it connects to.
  Only a resolver can play that trick, so this is driven with the two DNS answers scripted.
- MCP authorize without a state (c2107e5bb): the flow is bound to its initiator through the
  state, so a client that produced none must not send the person to the provider. authlib
  always makes a state, so only a stand-in client reaches this branch.
- Duplicate account on first sign-in (#27571, issue #27117, commits b190dcf + 50e050e):
  `insert_new_auth` committed with no integrity handling, so the losing half of a race left a
  shared session unusable, and nothing stopped two accounts for one address in different case.
  The race is a timing window, so it is driven at the model and migration level.
- SSO settings from env vars (#26928, commit e398ba350): `Config.seed_defaults` seeded rows for
  the `oauth.*` keys while their persistence was off, and once it was turned on the stale rows
  shadowed the environment for good. Seeding only runs at boot.
- Never-expiring sign-in tokens (#26802, issue #26141, commit 98656b7): a provider sending
  neither an expiry nor a refresh token got a fabricated one-hour lifetime, after which the
  session died unrecoverably. `_normalize_token_expiry` is a pure function with no route.

Discriminates: passes on dev bbfa876af; with each fix reverted in a backend copy the picture
is fetched from loopback, the stateless authorize redirects, the shared session raises
PendingRollbackError, the migrated database takes one address twice, the seeded row wins over
the environment and the refresh-less token gets a one-hour expiry.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
import sqlalchemy as sa
from authlib.integrations.starlette_client import StarletteOAuth2App
from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError

from harness.listener import listening
from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
REBOUND_HOST = "avatars.rebind.example"


@pytest.fixture
def oauth(owui_module):
    return owui_module("open_webui.utils.oauth")


# ── profile-picture SSRF (#26699) ──────────────────────────────────────────


@pytest.fixture
def rebinding_dns(monkeypatch):
    """A resolver that tells the URL check a public address and the connection loopback."""
    real_getaddrinfo = socket.getaddrinfo

    def public_answer(host, *args, **kwargs):
        if host == REBOUND_HOST:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        return real_getaddrinfo(host, *args, **kwargs)

    class LoopbackResolver(aiohttp.ThreadedResolver):
        async def resolve(self, host, port=0, family=socket.AF_INET):
            return await super().resolve(
                "127.0.0.1" if host == REBOUND_HOST else host, port, family
            )

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    monkeypatch.setattr(aiohttp.connector, "DefaultResolver", LoopbackResolver)
    for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(proxy, raising=False)


@pytest.mark.asyncio
async def test_a_picture_host_that_rebinds_to_loopback_is_never_fetched(oauth, rebinding_dns):
    """Narrow: the address the fetch connects to is checked again, not just the first answer."""
    manager = oauth.OAuthManager(app=FastAPI())
    with listening() as internal:
        internal.route("GET", "/avatar.png", (200, {"Content-Type": "image/png"}, PNG))
        picture = await manager._process_picture_url(
            picture_url=f"http://{REBOUND_HOST}:{internal.port}/avatar.png", access_token="t"
        )

    assert internal.requests_to("/avatar.png") == [], "the picture was fetched from loopback"
    assert picture == "/user.png"


@pytest.mark.asyncio
async def test_no_picture_url_means_the_default_picture(oauth):
    """Nearby: a provider without a picture still gets the default one, fetching nothing."""
    manager = oauth.OAuthManager(app=FastAPI())
    assert await manager._process_picture_url(picture_url="") == "/user.png"


# ── MCP authorize without a state (c2107e5bb) ──────────────────────────────


@pytest.mark.asyncio
async def test_an_mcp_authorization_without_a_state_never_leaves(oauth):
    """Narrow: with no state to bind the initiator to, nothing is saved and nobody is sent."""
    client = create_autospec(StarletteOAuth2App, instance=True)
    client.create_authorization_url.return_value = {"url": "https://mcp.example/authorize"}
    manager = oauth.OAuthClientManager(app=FastAPI())
    client_info = oauth.OAuthClientInformationFull(
        client_id="mcp-client", redirect_uris=["https://owui.example/oauth/clients/mcp:x/callback"]
    )
    manager.clients["mcp:x"] = {"client": client, "client_info": client_info}

    with pytest.raises(HTTPException) as refused:
        await manager.handle_authorize(request=None, client_id="mcp:x", user_id="alice")

    assert refused.value.status_code == 500
    client.save_authorize_data.assert_not_awaited()


# ── duplicate account on first sign-in (#27571) ───────────────────────────


@pytest.mark.asyncio
async def test_the_losing_half_of_a_sign_up_race_leaves_the_shared_session_usable(owui_module):
    """Narrow: the duplicate insert is rolled back, so the caller can still find the winner."""
    auths = owui_module("open_webui.models.auths")
    users = owui_module("open_webui.models.users")
    db_module = owui_module("open_webui.internal.db")
    async with memory_database(owui_module, users.User, auths.Auth) as sessions:
        async with sessions() as session:
            # what migration f0bd01a18a3d adds
            await session.execute(sa.text('CREATE UNIQUE INDEX email_ci ON "user" (lower(email))'))
            with patch.object(db_module, "DATABASE_ENABLE_SESSION_SHARING", True):
                winner = await auths.Auths.insert_new_auth(
                    email="racer@example.com", password="hash", name="Racer", db=session
                )
                with pytest.raises(IntegrityError):
                    await auths.Auths.insert_new_auth(
                        email="Racer@example.com", password="hash", name="Racer", db=session
                    )
                found = await users.Users.get_user_by_email("racer@example.com", db=session)

    assert found is not None and found.id == winner.id


def migrate_to_head(backend: Path, database_url: str, data_dir: Path) -> None:
    """Run the migration chain out of process; this process already holds a different DB."""
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ['DATABASE_URL'] = {database_url!r}
        os.environ['DATA_DIR'] = {str(data_dir)!r}
        sys.path.insert(0, {str(backend)!r})
        from alembic import command
        from alembic.config import Config
        from open_webui.env import OPEN_WEBUI_DIR
        config = Config(OPEN_WEBUI_DIR / 'alembic.ini')
        config.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
        command.upgrade(config, 'head')
        """
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "WEBUI_SECRET_KEY": "test"}
    migrated = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env
    )
    assert migrated.returncode == 0, f"alembic upgrade head failed\n{migrated.stderr[-3000:]}"


def test_a_migrated_database_refuses_one_address_in_two_cases(open_webui_backend, tmp_path):
    """Narrow: the same address in different case is one account; distinct and missing
    addresses are still allowed."""
    database_url = f"sqlite:///{(tmp_path / 'webui.db').resolve().as_posix()}"
    (tmp_path / "data").mkdir()
    migrate_to_head(open_webui_backend, database_url, tmp_path / "data")
    insert = sa.text('INSERT INTO "user" (id, name, email, role) VALUES (:id, :id, :email, :role)')
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            for user_id, email in [
                ("u1", "a@x.com"),
                ("u2", "b@x.com"),
                ("u3", None),
                ("u4", None),
            ]:
                connection.execute(insert, {"id": user_id, "email": email, "role": "user"})
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(insert, {"id": "u5", "email": "A@x.com", "role": "user"})
    finally:
        engine.dispose()


# ── SSO settings from env vars (#26928) ────────────────────────────────────


@pytest.mark.parametrize("oauth_persistence", [False, True], ids=["persistence-off", "on"])
@pytest.mark.asyncio
async def test_seeding_leaves_oauth_settings_to_whoever_is_authoritative(
    owui_module, oauth_persistence
):
    """Narrow (off): a boot with persistence off seeds no `oauth.*` row, so turning it on
    later still reads the environment. Nearby (on): with it on, the seeded row is the setting."""
    config = owui_module("open_webui.models.config").Config
    seeds = {"oauth.enable_signup": "first boot", "ui.default_user_role": "seeded"}
    async with memory_database(owui_module, config):
        with patch.object(config, "PERSISTENT_ENABLED", True):
            with patch.object(config, "OAUTH_PERSISTENT_ENABLED", oauth_persistence):
                await config.seed_defaults(seeds)
            with (
                patch.object(config, "OAUTH_PERSISTENT_ENABLED", True),
                patch.dict(config.DEFAULTS, {"oauth.enable_signup": "from env"}),
            ):
                signup = await config.get("oauth.enable_signup")
                default_role = await config.get("ui.default_user_role")

    assert signup == ("first boot" if oauth_persistence else "from env")
    assert default_role == "seeded"


# ── tokens without an expiry (#26802) ──────────────────────────────────────


def test_a_token_that_cannot_be_refreshed_is_not_given_an_invented_expiry(oauth):
    """Narrow: no expiry and no refresh token means non-expiring, not dead in an hour."""
    token = oauth._normalize_token_expiry({"access_token": "abc"})
    assert token["expires_at"] > time.time() + 365 * 86400


def test_a_refreshable_token_without_an_expiry_keeps_the_one_hour_default(oauth):
    """Nearby: with a refresh path the conservative hour still applies."""
    token = oauth._normalize_token_expiry({"access_token": "abc", "refresh_token": "r"})
    assert token["expires_at"] == pytest.approx(time.time() + 3600, abs=5)


@pytest.mark.parametrize(
    "token",
    [
        {"expires_at": 1900000000},
        {"expires_in": 300},
        {"access_token": "abc"},
        {"access_token": "abc", "refresh_token": "r"},
    ],
)
def test_every_branch_stamps_an_int_expiry_and_an_issued_at(oauth, token):
    """Broad: every resolution path leaves a usable numeric expiry and issue time."""
    normalized = oauth._normalize_token_expiry(dict(token))
    assert isinstance(normalized["expires_at"], int) and normalized["expires_at"] > 0
    assert normalized["issued_at"] == pytest.approx(time.time(), abs=5)
