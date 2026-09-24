"""Regression: a token-exchange sign-in ignored roles carried on the access token itself.

open-webui 0.11.4 fix `10d1cfe63`: role mapping only read the roles claim from the provider's
userinfo answer, so a provider that puts roles on the access token (Keycloak's
`realm_access.roles`) left the account at its old role, and an exchange whose roles could not be
read anywhere was let through. `get_user_role` now also reads the access token's claims, and an
exchange with no readable roles claim is refused with 403.

Twin of unit/security/test_signin_role_claim_resolution.py.

Discriminates: passes on dev bbfa876af; with 10d1cfe63 reverted in a copy the token's roles
never reach the account (it stays pending) and the roleless exchange gets a session instead of
a 403.
"""

from __future__ import annotations

import httpx
import pytest

from harness.oidc_provider import (
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EXCHANGE = "/api/v1/auths/oauth/oidc/token/exchange"


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


@pytest.fixture
def person(sso, idp) -> dict:
    """Someone who signed in through the browser once, so the exchange finds an account."""
    person = idp.sign_in_as()
    first = sign_in(sso)
    assert first.token, f"the first sign-in failed: {first.error}"
    assert session_user(sso, first.token)["role"] == "pending"
    return person


def exchange(sso, idp, userinfo: dict, **token_options) -> httpx.Response:
    token = idp.issue_access_token(userinfo, **token_options)
    return httpx.post(f"{sso.base_url}{EXCHANGE}", json={"token": token}, timeout=60.0)


@pytest.mark.parametrize(
    ("roles_claim", "token_claims", "expected_role"),
    [
        ("roles", {"roles": ["user"]}, "user"),
        ("realm_access.roles", {"realm_access": {"roles": ["admin"]}}, "admin"),
    ],
    ids=["flat", "nested"],
)
def test_roles_carried_only_on_the_access_token_decide_the_role(
    sso, idp, person, roles_claim, token_claims, expected_role
):
    """Narrow: with no roles in userinfo, the access token's roles claim is read."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True, OAUTH_ROLES_CLAIM=roles_claim):
        answer = exchange(sso, idp, person, claims=token_claims)
    assert answer.status_code == 200, answer.text
    assert answer.json()["role"] == expected_role, "the token's roles were ignored"


@pytest.mark.parametrize("opaque", [False, True], ids=["jwt-without-roles", "opaque-token"])
def test_an_exchange_with_no_readable_roles_is_refused(sso, idp, person, opaque):
    """Narrow: roles neither in userinfo nor on the token refuse the sign-in."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        answer = exchange(sso, idp, person, opaque=opaque)
    assert answer.status_code == 403, answer.text


def test_userinfo_roles_still_win_over_the_token(sso, idp, person):
    """Nearby: roles the provider's userinfo answers with still decide."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        answer = exchange(sso, idp, {**person, "roles": ["admin"]}, claims={"roles": ["user"]})
    assert answer.status_code == 200, answer.text
    assert answer.json()["role"] == "admin"


def test_a_wildcard_allowed_role_admits_an_exchange_without_roles(sso, idp, person):
    """Nearby: `*` in the allowed roles still lets a roleless exchange through."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True, OAUTH_ALLOWED_ROLES="*"):
        answer = exchange(sso, idp, person)
    assert answer.status_code == 200, answer.text
    assert answer.json()["role"] == "pending"
