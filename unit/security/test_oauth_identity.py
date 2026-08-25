"""Regression: OAuth/OIDC sign-in identity handling.

open-webui 0.11.1 gathers six fixes to the provider sign-in path:

* `5462c02af` (#28065): only `client_id` was registered as a known JOSE header, so an ID token
  carrying any other vendor header (CyberArk `app_id`) was rejected by joserfc and the sign-in
  died with "the email or password provided is incorrect". The fix disables joserfc's
  unknown-header rejection globally at `open_webui.utils.oauth` import time.
* `73c1f5806` (#28624): the SQLite branch of `get_user_by_oauth_sub` used
  `User.oauth.contains(...)`, which on a JSON column degrades to a substring `LIKE`. A sub that
  is a `LIKE` pattern matched a different person's account, and a provider entry carrying any
  key besides `sub` stopped matching at all.
* `a6834f089` (#28954, issue #27760): providers that identify people by a number (GitHub) store
  the sub as a JSON number, which never equalled the string sub the lookup was given. The lookup
  now coerces to `str` and, on SQLite, also compares against the integer; the writer merges into
  the existing provider entry and stores `str(sub)`.
* `d799e81ed` / `e96844581`: `token_exchange` returned a session immediately after finding the
  user, so OAuth role and group mapping never ran for that sign-in path. `get_user_role` also
  reset an existing account to the default role when the provider sent no roles claim.
* `aeda6ff13`: back-channel logout fetched the provider JWKS through an anonymous
  `PyJWKClient`, so signature validation failed for providers whose key endpoint needs client
  authentication and the person stayed signed in. It now fetches through the authlib client.
* `c2107e5bb3`: the OAuth-client callback bound the connection to whoever returned with the
  authorization instead of the account that started it, and `signout` left the `owui-session`
  cookie and the server-side session in place.

Discriminates: passes on v0.11.1, fails on v0.11.0 (private JOSE headers are fatal, the sub
lookup is a substring/type match, token exchange skips role and group mapping, an existing
account is reset to the default role, back-channel logout fetches keys anonymously, the callback
stores the token for the wrong account and signout leaves `owui-session` behind).
"""

from __future__ import annotations

import inspect
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"
MALLORY = "mallory-user-id"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def oauth_module(owui_module):
    """`open_webui.utils.oauth`. Importing it is what installs the joserfc relaxation."""
    return owui_module("open_webui.utils.oauth")


@pytest.fixture(scope="module")
def users_module(owui_module):
    """`open_webui.models.users` (get_user_by_oauth_sub, update_user_oauth_by_id)."""
    return owui_module("open_webui.models.users")


@pytest.fixture(scope="module")
def db_module(owui_module):
    """`open_webui.internal.db` (session sharing flag)."""
    return owui_module("open_webui.internal.db")


@pytest.fixture(scope="module")
def auths_module(owui_module):
    """`open_webui.routers.auths` (signout, token_exchange)."""
    return owui_module("open_webui.routers.auths")


# --------------------------------------------------------------------------- db helper


@asynccontextmanager
async def _user_db(users_module, db_module, rows):
    """A throwaway in-memory SQLite holding just the user table.

    `rows` maps a user name to the raw `oauth` JSON to store. The session is handed to the
    model layer explicitly so nothing touches the suite's shared database.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    user_table = users_module.User
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(user_table.__table__.create)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            ids = {}
            now = int(time.time())
            for name, oauth in rows.items():
                ids[name] = str(uuid.uuid4())
                session.add(
                    user_table(
                        id=ids[name],
                        name=name,
                        email=f"{name}@example.com",
                        role="user",
                        profile_image_url="",
                        oauth=oauth,
                        created_at=now,
                        updated_at=now,
                        last_active_at=now,
                    )
                )
            await session.commit()
            # get_async_db_context only reuses the passed session when sharing is enabled.
            with patch.object(db_module, "DATABASE_ENABLE_SESSION_SHARING", True):
                yield session, ids
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- jose headers


def _id_token(secret: str, headers: dict) -> str:
    import jwt as pyjwt

    return pyjwt.encode({"sub": "1"}, secret, algorithm="HS256", headers=headers)


def _verify(token: str, secret: str, algorithms=("HS256",)):
    from joserfc import jws as joserfc_jws
    from joserfc.jwk import OctKey

    return joserfc_jws.deserialize_compact(
        token, OctKey.import_key(secret), algorithms=list(algorithms)
    )


SECRET = "unit-test-hmac-secret-of-decent-length"


@pytest.mark.parametrize("header_name", ["app_id", "client_id", "x-vendor-thing"])
def test_private_jose_header_does_not_break_verification(oauth_module, header_name):
    """Narrow (#28065): a vendor header in the ID token JOSE header is ignored, not fatal."""
    assert oauth_module is not None
    token = _id_token(SECRET, {header_name: "vendor-value"})
    assert b'"sub"' in _verify(token, SECRET).payload


def test_crit_header_still_rejected(oauth_module):
    """Nearby: relaxing unknown headers must not disarm `crit`."""
    from joserfc.errors import JoseError

    assert oauth_module is not None
    token = _id_token(SECRET, {"crit": ["app_id"], "app_id": "x"})
    with pytest.raises(JoseError):
        _verify(token, SECRET)


def test_signature_still_verified(oauth_module):
    """Nearby: a token signed with the wrong key is still rejected."""
    from joserfc.errors import BadSignatureError

    assert oauth_module is not None
    token = _id_token(SECRET, {})
    with pytest.raises(BadSignatureError):
        _verify(token, "a-completely-different-secret-value")


def test_algorithm_allowlist_still_enforced(oauth_module):
    """Nearby: an algorithm outside the allowlist is still rejected."""
    from joserfc.errors import JoseError

    assert oauth_module is not None
    token = _id_token(SECRET, {})
    with pytest.raises(JoseError):
        _verify(token, SECRET, algorithms=("HS512",))


# --------------------------------------------------------------------------- sub lookup


@pytest.mark.asyncio
async def test_like_wildcard_sub_does_not_match_another_account(users_module, db_module):
    """Narrow (#28624): an underscore in a sub is a literal, not a single-character wildcard."""
    rows = {"alice": {"oidc": {"sub": "abc12345"}}}
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("oidc", "abc_2345", db=session)
        assert found is None


@pytest.mark.asyncio
async def test_percent_sub_does_not_match_any_account(users_module, db_module):
    """Narrow (#28624): `%` is a literal sub, not "match whoever is first"."""
    rows = {"alice": {"oidc": {"sub": "abc12345"}}}
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("oidc", "%", db=session)
        assert found is None


@pytest.mark.asyncio
async def test_sub_matches_when_provider_entry_carries_other_keys(users_module, db_module):
    """Narrow (#28624): the match is on the `sub` value, not on the serialised provider entry."""
    rows = {"bob": {"oidc": {"sub": "12345", "email": "bob@example.com"}}}
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("oidc", "12345", db=session)
        assert found is not None and found.name == "bob"


@pytest.mark.asyncio
async def test_numeric_stored_sub_matches_string_lookup(users_module, db_module):
    """Narrow (#28954): a GitHub id stored as a JSON number matches the string sub."""
    rows = {"carol": {"github": {"sub": 777}}}
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("github", "777", db=session)
        assert found is not None and found.name == "carol"


@pytest.mark.asyncio
async def test_integer_lookup_matches_string_stored_sub(users_module, db_module):
    """Narrow (#28954): the lookup coerces an int sub, so a stored string still matches."""
    rows = {"dave": {"github": {"sub": "888"}}}
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("github", 888, db=session)
        assert found is not None and found.name == "dave"


@pytest.mark.asyncio
async def test_oauth_writer_keeps_other_provider_keys_and_stores_string(users_module, db_module):
    """Narrow (#28954): linking a sub merges into the provider entry instead of replacing it."""
    rows = {"bob": {"oidc": {"sub": "12345", "email": "bob@example.com"}}}
    async with _user_db(users_module, db_module, rows) as (session, ids):
        updated = await users_module.Users.update_user_oauth_by_id(
            ids["bob"], "oidc", 999, db=session
        )
        assert updated.oauth["oidc"] == {"sub": "999", "email": "bob@example.com"}


@pytest.mark.asyncio
async def test_exact_sub_still_matches_and_unknown_sub_does_not(users_module, db_module):
    """Nearby: the ordinary hit and miss are unchanged."""
    rows = {
        "alice": {"oidc": {"sub": "abc12345"}},
        "erin": {"other": {"sub": "abc12345"}},
    }
    async with _user_db(users_module, db_module, rows) as (session, _ids):
        found = await users_module.Users.get_user_by_oauth_sub("oidc", "abc12345", db=session)
        assert found is not None and found.name == "alice"
        assert await users_module.Users.get_user_by_oauth_sub("oidc", "nope", db=session) is None
        other = await users_module.Users.get_user_by_oauth_sub("nope", "abc12345", db=session)
        assert other is None


# --------------------------------------------------------------------------- role default


def _runtime_config(**overrides):
    values = {
        "ENABLE_OAUTH_ROLE_MANAGEMENT": True,
        "OAUTH_ROLES_CLAIM": "roles",
        "OAUTH_ALLOWED_ROLES": ["user"],
        "OAUTH_ADMIN_ROLES": ["admin"],
        "DEFAULT_USER_ROLE": "user",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@asynccontextmanager
async def _role_env(oauth_module, user_count=5, **config_overrides):
    config = _runtime_config(**config_overrides)
    with patch.object(
        oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)
    ), patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=user_count)):
        yield


@pytest.mark.asyncio
async def test_existing_admin_keeps_role_when_provider_sends_no_roles(oauth_module):
    """Narrow: an account is not demoted just because the token carries no roles claim."""
    manager = oauth_module.OAuthManager.__new__(oauth_module.OAuthManager)
    user = SimpleNamespace(id=ALICE, role="admin")
    async with _role_env(oauth_module):
        assert await manager.get_user_role(user, {"email": "a@example.com"}) == "admin"


@pytest.mark.asyncio
async def test_provider_roles_still_decide_the_role(oauth_module):
    """Nearby: when the token does carry roles, they win over the existing role."""
    manager = oauth_module.OAuthManager.__new__(oauth_module.OAuthManager)
    admin = SimpleNamespace(id=ALICE, role="admin")
    async with _role_env(oauth_module):
        assert await manager.get_user_role(admin, {"roles": ["user"]}) == "user"
        assert await manager.get_user_role(admin, {"roles": ["admin"]}) == "admin"


@pytest.mark.asyncio
async def test_new_user_still_gets_the_default_role(oauth_module):
    """Nearby: with no existing account the default role is still the fallback."""
    manager = oauth_module.OAuthManager.__new__(oauth_module.OAuthManager)
    async with _role_env(oauth_module, DEFAULT_USER_ROLE="pending"):
        assert await manager.get_user_role(None, {"email": "new@example.com"}) == "pending"


# --------------------------------------------------------------------------- token exchange


class _FakeOAuthManager:
    """Stands in for `request.app.state.oauth_manager`: records what token exchange drives."""

    def __init__(self, client):
        self._client = client
        self.role_calls = []
        self.group_calls = []

    def get_client(self, provider):
        return self._client

    async def update_user_role_from_oauth(self, request, user, user_data, provider, db=None):
        self.role_calls.append(provider)
        return user

    async def update_user_groups(self, request, user, user_data, default_permissions, db=None):
        self.group_calls.append(user)


@asynccontextmanager
async def _token_exchange_env(auths_module, manager, user, config):
    async def _config_get(key, default=None):
        return config.get(key, default)

    users = SimpleNamespace(
        get_user_by_oauth_sub=AsyncMock(return_value=user),
        get_user_by_email=AsyncMock(return_value=None),
        update_user_oauth_by_id=AsyncMock(return_value=user),
    )
    with patch.object(auths_module, "ENABLE_OAUTH_TOKEN_EXCHANGE", True), patch.object(
        auths_module, "token_exchange_rate_limiter", None
    ), patch.object(auths_module, "OAUTH_PROVIDERS", {"oidc": {"sub_claim": "sub"}}), patch.object(
        auths_module, "OAUTH_TOKEN_EXCHANGE_TRUSTED_CLIENT_IDS", []
    ), patch.object(auths_module.Config, "get", AsyncMock(side_effect=_config_get)), patch.object(
        auths_module, "Users", users
    ), patch.object(
        auths_module, "create_session_response", AsyncMock(return_value={"ok": True})
    ):
        yield users


def _exchange_request(manager):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(oauth_manager=manager)),
        client=SimpleNamespace(host="127.0.0.1"),
    )


@pytest.mark.asyncio
async def test_token_exchange_runs_role_and_group_mapping(auths_module):
    """Narrow: signing in from another application applies role and group mapping too."""
    provider_client = SimpleNamespace(
        userinfo=AsyncMock(return_value={"sub": "12345", "email": "alice@example.com"})
    )
    manager = _FakeOAuthManager(provider_client)
    user = SimpleNamespace(id=ALICE, role="user", oauth={"oidc": {"sub": "12345"}})
    config = {
        "oauth.email_claim": "email",
        "oauth.sub_claim": "sub",
        "oauth.allowed_domains": ["*"],
        "oauth.merge_accounts_by_email": False,
        "oauth.enable_group_mapping": True,
        "user.permissions": {},
    }
    async with _token_exchange_env(auths_module, manager, user, config):
        result = await auths_module.token_exchange(
            request=_exchange_request(manager),
            response=SimpleNamespace(),
            provider="oidc",
            form_data=SimpleNamespace(token="provider-access-token"),
            db=None,
        )

    assert result == {"ok": True}
    assert manager.role_calls == ["oidc"], "role mapping did not run on the token exchange path"
    assert manager.group_calls, "group mapping did not run on the token exchange path"


@pytest.mark.asyncio
async def test_token_exchange_skips_group_mapping_when_disabled(auths_module):
    """Nearby: group mapping stays off when the setting is off."""
    provider_client = SimpleNamespace(
        userinfo=AsyncMock(return_value={"sub": "12345", "email": "alice@example.com"})
    )
    manager = _FakeOAuthManager(provider_client)
    user = SimpleNamespace(id=ALICE, role="user", oauth={"oidc": {"sub": "12345"}})
    config = {
        "oauth.email_claim": "email",
        "oauth.sub_claim": "sub",
        "oauth.allowed_domains": ["*"],
        "oauth.merge_accounts_by_email": False,
        "oauth.enable_group_mapping": False,
        "user.permissions": {},
    }
    async with _token_exchange_env(auths_module, manager, user, config):
        await auths_module.token_exchange(
            request=_exchange_request(manager),
            response=SimpleNamespace(),
            provider="oidc",
            form_data=SimpleNamespace(token="provider-access-token"),
            db=None,
        )

    assert manager.group_calls == []


@pytest.mark.asyncio
async def test_token_exchange_still_rejects_a_disallowed_email_domain(auths_module):
    """Nearby: the domain allowlist still bites before any mapping runs."""
    from fastapi import HTTPException

    provider_client = SimpleNamespace(
        userinfo=AsyncMock(return_value={"sub": "12345", "email": "alice@evil.example"})
    )
    manager = _FakeOAuthManager(provider_client)
    user = SimpleNamespace(id=ALICE, role="user", oauth={"oidc": {"sub": "12345"}})
    config = {
        "oauth.email_claim": "email",
        "oauth.sub_claim": "sub",
        "oauth.allowed_domains": ["example.com"],
        "oauth.merge_accounts_by_email": False,
        "oauth.enable_group_mapping": True,
        "user.permissions": {},
    }
    async with _token_exchange_env(auths_module, manager, user, config):
        with pytest.raises(HTTPException) as excinfo:
            await auths_module.token_exchange(
                request=_exchange_request(manager),
                response=SimpleNamespace(),
                provider="oidc",
                form_data=SimpleNamespace(token="provider-access-token"),
                db=None,
            )
    assert excinfo.value.status_code == 403
    assert manager.role_calls == []


# --------------------------------------------------------------------------- signout


class _FakeSession(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleared = False

    def clear(self):
        self.cleared = True
        super().clear()


def _deleted_cookie_names(response) -> set[str]:
    return {
        raw_value.decode().split("=", 1)[0].strip()
        for raw_key, raw_value in response.raw_headers
        if raw_key.decode().lower() == "set-cookie"
    }


@pytest.mark.asyncio
async def test_signout_clears_the_server_side_session_and_its_cookie(auths_module):
    """Narrow: signing out drops the starlette session and the `owui-session` cookie."""
    from starlette.responses import Response

    session = _FakeSession({"user_id": ALICE})
    request = SimpleNamespace(headers={}, cookies={}, session=session, app=None)
    response = Response()

    with patch.object(auths_module, "WEBUI_AUTH_SIGNOUT_REDIRECT_URL", None):
        await auths_module.signout(request=request, response=response, db=None)

    deleted = _deleted_cookie_names(response)
    assert "owui-session" in deleted
    assert session.cleared and not session


@pytest.mark.asyncio
async def test_signout_still_clears_the_legacy_cookies(auths_module):
    """Nearby: the cookies signout already cleared are still cleared."""
    from starlette.responses import Response

    request = SimpleNamespace(headers={}, cookies={}, session=_FakeSession(), app=None)
    response = Response()

    with patch.object(auths_module, "WEBUI_AUTH_SIGNOUT_REDIRECT_URL", None):
        await auths_module.signout(request=request, response=response, db=None)

    assert {"token", "oui-session", "oauth_id_token"} <= _deleted_cookie_names(response)


# --------------------------------------------------------------------------- client callback


def _fake_oauth_client(state_user_id):
    framework = SimpleNamespace(
        get_state_data=AsyncMock(
            return_value={"user_id": state_user_id} if state_user_id else None
        ),
        clear_state_data=AsyncMock(return_value=None),
    )
    return SimpleNamespace(
        framework=framework,
        client_id="client-1",
        authorize_access_token=AsyncMock(return_value={"access_token": "at", "expires_in": 60}),
    )


@asynccontextmanager
async def _client_callback_env(oauth_module, client, sessions):
    async def _config_get(key, default=None):
        return "http://localhost:8080" if key == "webui.url" else default

    with patch.object(oauth_module, "OAuthSessions", sessions), patch.object(
        oauth_module.Config, "get", AsyncMock(side_effect=_config_get)
    ), patch.object(
        oauth_module, "should_send_oauth_resource", lambda client_info: False
    ), patch.object(
        oauth_module,
        "get_verified_user_by_id",
        AsyncMock(side_effect=lambda user_id: SimpleNamespace(id=user_id, role="user")),
        create=True,
    ), patch.object(
        oauth_module,
        "get_optional_verified_user_from_request",
        AsyncMock(return_value=None),
        create=True,
    ):
        manager = oauth_module.OAuthClientManager.__new__(oauth_module.OAuthClientManager)
        manager.get_client = AsyncMock(return_value=client)
        manager.get_client_info = AsyncMock(return_value=SimpleNamespace(resource=None))
        yield manager


@pytest.mark.asyncio
async def test_client_callback_binds_the_token_to_the_user_who_started_authorize(oauth_module):
    """Narrow: the connection lands on the account that began the flow, never on the visitor.

    Each ref is called the way its own `main.py` calls it: pre-fix that means handing in the
    verified user of the returning request, which is exactly the substitution the fix removes.
    """
    from starlette.responses import Response

    client = _fake_oauth_client(state_user_id=ALICE)
    sessions = SimpleNamespace(
        get_sessions_by_user_id=AsyncMock(return_value=[]),
        delete_session_by_id=AsyncMock(return_value=True),
        create_session=AsyncMock(return_value=SimpleNamespace(id="session-1")),
    )
    request = SimpleNamespace(
        query_params={"state": "state-token"},
        session={},
        base_url="http://localhost:8080/",
        cookies={},
        headers={},
        app=SimpleNamespace(state=SimpleNamespace(redis=None)),
        state=SimpleNamespace(token=None),
    )

    async with _client_callback_env(oauth_module, client, sessions) as manager:
        kwargs = {"client_id": "client-1", "response": Response()}
        if "user_id" in inspect.signature(manager.handle_callback).parameters:
            kwargs["user_id"] = MALLORY
        await manager.handle_callback(request, **kwargs)

    stored_for = [call.kwargs.get("user_id") for call in sessions.create_session.await_args_list]
    assert stored_for == [ALICE], f"OAuth token was stored for {stored_for}, expected the initiator"


# --------------------------------------------------------------------------- back-channel logout


class _FakeMetadataResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status = 200

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAiohttpSession:
    """Serves the provider's discovery document, so the pre-fix path matches the issuer."""

    def __init__(self, payload):
        self._payload = payload

    def get(self, url, **kwargs):
        return _FakeMetadataResponse(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_backchannel_logout_fetches_keys_through_the_authenticated_client(oauth_module):
    """Narrow: the JWKS is fetched through the authlib client, not an anonymous PyJWKClient."""
    import jwt as pyjwt

    issuer = "https://idp.example.com"
    metadata = {"issuer": issuer, "jwks_uri": f"{issuer}/jwks"}
    logout_token = pyjwt.encode(
        {
            "iss": issuer,
            "aud": "client-1",
            "iat": int(time.time()),
            "events": {"http://schemas.openid.net/event/backchannel-logout": {}},
            "sub": "12345",
        },
        "irrelevant-signing-secret",
        algorithm="HS256",
        headers={"kid": "key-1"},
    )

    client = SimpleNamespace(
        client_id="client-1",
        load_server_metadata=AsyncMock(return_value=metadata),
        fetch_jwk_set=AsyncMock(return_value={"keys": []}),
    )
    manager = oauth_module.OAuthManager.__new__(oauth_module.OAuthManager)
    manager.get_client = lambda name: client
    manager.get_server_metadata_url = lambda name: f"{issuer}/.well-known/openid-configuration"

    request = SimpleNamespace(form=AsyncMock(return_value={"logout_token": logout_token}))

    anonymous_key_client = AsyncMock()
    with patch.object(oauth_module, "OAUTH_PROVIDERS", {"oidc": {}}), patch.object(
        oauth_module.aiohttp, "ClientSession", lambda **kwargs: _FakeAiohttpSession(metadata)
    ), patch.object(pyjwt, "PyJWKClient", anonymous_key_client):
        await manager.handle_backchannel_logout(request, db=None)

    assert client.fetch_jwk_set.await_count == 1, "JWKS was not fetched through the OAuth client"
    assert anonymous_key_client.call_count == 0, "JWKS was fetched anonymously"
