"""Regression: provider roles were read only in some setups, and an unreadable roles
claim on the token-exchange path was let through.

open-webui 0.11.4 fix `10d1cfe6375f207acaa531e857edb575ded2cfc3`: the roles claim
extraction on the sign-in path only looked at the provider's userinfo response, with a
nested-walk that degraded to `None` for anything but a list/str/int, so providers that
put roles on the access token rather than in userinfo left the account at its default
role, and there was no read of the access token's own claims at all. `get_user_role` now
takes the `access_token`, reads the claim from the token itself when userinfo does not
carry it, and token-exchange sign-ins whose roles claim cannot be read from either
source are refused with 403 instead of being let through at the default role.

Discriminates: passes on v0.11.4 (344ea5306), fails on v0.11.4~ (10d1cfe63^): pre-fix
`get_user_role` has no `access_token` parameter, roles living only in the token are
never applied, and a sign-in with no readable roles claim anywhere still returns a
role instead of raising 403.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt as pyjwt
import pytest

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"


@pytest.fixture(scope="module")
def oauth_module(owui_module):
    return owui_module("open_webui.utils.oauth")


@pytest.fixture(scope="module")
def auths_module(owui_module):
    return owui_module("open_webui.routers.auths")


def _runtime_config(**overrides):
    values = {
        "ENABLE_OAUTH_ROLE_MANAGEMENT": True,
        "OAUTH_ROLES_CLAIM": "roles",
        "OAUTH_ALLOWED_ROLES": ["user"],
        "OAUTH_ADMIN_ROLES": ["admin"],
        "DEFAULT_USER_ROLE": "pending",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _token_with_claims(claims: dict) -> str:
    return pyjwt.encode(claims, "unit-test-key-of-a-decent-length-0123456789", algorithm="HS256")


def _make_manager(oauth_module):
    return oauth_module.OAuthManager.__new__(oauth_module.OAuthManager)


async def _num_users(_self=None):
    return 5


def _patched(oauth_module, config):
    return patch.object(
        oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)
    ), patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5))


@pytest.mark.asyncio
async def test_roles_claim_from_access_token_is_applied(oauth_module):
    """NARROW: roles carried only in the access token's claims reach the role mapping."""
    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="pending")
    access_token = _token_with_claims({"sub": "12345", "roles": ["user"]})
    config = _runtime_config()
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        role = await manager.get_user_role(
            user, {"email": "a@example.com"}, access_token=access_token
        )
    assert role == "user", "roles on the access token were ignored and the default kept"


@pytest.mark.asyncio
async def test_roles_claim_nested_path_from_access_token_is_applied(oauth_module):
    """NARROW: a dotted roles claim on the token is read at its nested path."""
    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="pending")
    access_token = _token_with_claims({"sub": "12345", "realm_access": {"roles": ["admin"]}})
    config = _runtime_config(OAUTH_ROLES_CLAIM="realm_access.roles")
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        role = await manager.get_user_role(
            user, {"email": "a@example.com"}, access_token=access_token
        )
    assert role == "admin", "nested roles on the access token were ignored"


@pytest.mark.asyncio
async def test_unreadable_roles_claim_refuses_the_sign_in(oauth_module):
    """NARROW: a sign-in whose roles claim cannot be read is refused, not let through."""
    from fastapi import HTTPException

    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="pending")
    access_token = _token_with_claims({"sub": "12345"})  # no roles claim anywhere
    config = _runtime_config()
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await manager.get_user_role(
                user, {"email": "a@example.com"}, access_token=access_token
            )
    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_undecodable_token_with_no_userinfo_roles_is_refused(oauth_module):
    """NARROW: a token that is not a decodable JWT and carries no roles claim is refused."""
    from fastapi import HTTPException

    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="pending")
    config = _runtime_config()
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await manager.get_user_role(
                user, {"email": "a@example.com"}, access_token="opaque-provider-token"
            )
    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_no_roles_anywhere_without_access_token_keeps_the_existing_role(oauth_module):
    """NEARBY: the browser callback path still keeps an existing account at its role."""
    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="user")
    config = _runtime_config()
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        role = await manager.get_user_role(user, {"email": "a@example.com"})
    assert role == "user"


def _role_call_kwargs(manager):
    """`access_token` exists only after 10d1cfe63; pass it when the checkout has it."""
    params = inspect.signature(manager.get_user_role).parameters
    token = _token_with_claims({"sub": "12345", "roles": ["user"]})
    return {"access_token": token} if "access_token" in params else {}


@pytest.mark.asyncio
async def test_userinfo_roles_still_win_over_token_roles(oauth_module):
    """NEARBY: when userinfo does carry roles they still decide, over the token or not."""
    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="admin")
    config = _runtime_config()
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        role = await manager.get_user_role(
            user, {"email": "a@example.com", "roles": ["admin"]}, **_role_call_kwargs(manager)
        )
    assert role == "admin"


@pytest.mark.asyncio
async def test_wildcard_allowed_roles_still_admit_a_token_without_roles(oauth_module):
    """NEARBY: `*` in the allowed list still admits a sign-in that has no roles claim."""
    manager = _make_manager(oauth_module)
    user = SimpleNamespace(id=ALICE, role="user")
    config = _runtime_config(OAUTH_ALLOWED_ROLES=["*"])
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module.Users, "get_num_users", AsyncMock(return_value=5)),
    ):
        role = await manager.get_user_role(
            user, {"email": "a@example.com"}, **_role_call_kwargs(manager)
        )
    assert role == "user"


# ── the token-exchange router now hands the access token to the role mapping ──


class _StubOauthManager:
    """Records what the router hands to `update_user_role_from_oauth`."""

    def __init__(self):
        self.role_calls = []

    def get_client(self, provider):
        return SimpleNamespace(
            userinfo=AsyncMock(
                return_value={"sub": "12345", "email": "alice@example.com"}
            )
        )

    async def update_user_role_from_oauth(
        self, request, user, user_data, provider, *, access_token=None, db=None
    ):
        self.role_calls.append({"provider": provider, "access_token": access_token})
        return user

    async def update_user_groups(self, request, user, user_data, default_permissions, db=None):
        return None


def _exchange_request(manager):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(oauth_manager=manager)),
        client=SimpleNamespace(host="127.0.0.1"),
    )


@pytest.mark.asyncio
async def test_token_exchange_passes_the_access_token_to_role_mapping(auths_module):
    """NARROW: the router hands the presented provider token to `get_user_role` via the manager."""
    signature = inspect.signature(auths_module.token_exchange)
    assert signature.parameters, "token_exchange not importable in this checkout"
    manager = _StubOauthManager()
    user = SimpleNamespace(id=ALICE, role="user", oauth={"oidc": {"sub": "12345"}})
    users = SimpleNamespace(
        get_user_by_oauth_sub=AsyncMock(return_value=user),
        get_user_by_email=AsyncMock(return_value=None),
    )

    async def _config_get(key, default=None):
        values = {
            "oauth.email_claim": "email",
            "oauth.sub_claim": "sub",
            "oauth.allowed_domains": ["*"],
            "oauth.merge_accounts_by_email": False,
            "oauth.enable_group_mapping": False,
            "user.permissions": {},
        }
        return values.get(key, default)

    with (
        patch.object(auths_module, "ENABLE_OAUTH_TOKEN_EXCHANGE", True),
        patch.object(auths_module, "token_exchange_rate_limiter", None),
        patch.object(auths_module, "OAUTH_PROVIDERS", {"oidc": {"sub_claim": "sub"}}),
        patch.object(auths_module, "OAUTH_TOKEN_EXCHANGE_TRUSTED_CLIENT_IDS", []),
        patch.object(auths_module.Config, "get", AsyncMock(side_effect=_config_get)),
        patch.object(auths_module, "Users", users),
        patch.object(auths_module, "create_session_response", AsyncMock(return_value={"ok": True})),
    ):
        await auths_module.token_exchange(
            request=_exchange_request(manager),
            response=SimpleNamespace(),
            provider="oidc",
            form_data=SimpleNamespace(token="provider-access-token"),
            db=None,
        )

    assert manager.role_calls == [
        {"provider": "oidc", "access_token": "provider-access-token"}
    ], "the router did not hand the access token to the role mapping"
