"""Regressions in how SSO group claims become groups, fixed for 0.11.5.

- Token exchange group mapping (fix `f41253875`): an exchange read the roles claim from the
  access token when the provider's userinfo answer lacked it, but group mapping only ever read
  userinfo, so a provider that puts groups on the access token alone left the account out of
  every group. Groups now fall back to the token's claims the same way roles do.
- Blocked groups created (fix `fcb0af3fd`, open-webui/open-webui#31316, issue
  open-webui/open-webui#29558): with automatic group creation on, every group named in the claim
  was created at sign-in, including those `OAUTH_BLOCKED_GROUPS` excludes. Membership sync
  already skipped them, so a provider sending a user's whole directory membership filled the
  groups list with empty groups. A blocked group is no longer created.
- Exchange without a groups claim (fix `184eada9e`, #33846): with group management on, an
  exchange whose userinfo and token carried no groups claim signed in and took the account out
  of every group. For an account in a group it is now refused with 403 and the groups stay; the
  same exchange with the claim on the token still signs in.

Discriminates: passes on dev efe63bd34; with f41253875 reverted in a backend copy the
token-claims exchange test fails (the account joins no group), and with fcb0af3fd reverted the
blocked-creation test fails (the blocked group exists after the sign-in). The missing-claim
test passes on dev f659c0d12 and fails with 184eada9e reverted (the exchange answers 200 and the
account leaves its group). The nearby tests pass on both.
"""

from __future__ import annotations

import secrets

import httpx
import pytest

from harness.oidc_provider import (
    group_member_ids,
    group_named,
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
def group(sso):
    with group_named(sso, f"team-{secrets.token_hex(4)}") as created:
        yield created


def _signed_in_user(sso) -> dict:
    result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return session_user(sso, result.token)


def _groups_named(sso, *names: str) -> list[dict]:
    with sso.client() as admin:
        listed = admin.get("/api/v1/groups/")
    listed.raise_for_status()
    return [group for group in listed.json() if group["name"] in names]


def _delete_groups(sso, groups: list[dict]) -> None:
    with sso.client() as admin:
        for group in groups:
            admin.delete(f"/api/v1/groups/id/{group['id']}/delete")


def _exchange(sso, idp, userinfo: dict, **token_options) -> httpx.Response:
    token = idp.issue_access_token(userinfo, **token_options)
    return httpx.post(f"{sso.base_url}{EXCHANGE}", json={"token": token}, timeout=60.0)


# --- token exchange reads groups from the token (f41253875) ----------------------------------


def test_an_exchange_joins_the_groups_carried_only_on_the_access_token(sso, idp, group):
    person = idp.sign_in_as()
    account = _signed_in_user(sso)

    with oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True):
        answer = _exchange(sso, idp, person, claims={"groups": [group["name"]]})

    assert answer.status_code == 200, answer.text
    assert account["id"] in group_member_ids(sso, group), (
        "the access token's groups claim was ignored because userinfo carried none"
    )


def test_an_exchange_prefers_the_userinfo_groups_over_the_token(sso, idp):
    person = idp.sign_in_as()
    account = _signed_in_user(sso)
    tag = secrets.token_hex(3)
    with (
        group_named(sso, f"from-userinfo-{tag}") as from_userinfo,
        group_named(sso, f"from-token-{tag}") as from_token,
    ):
        with oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True):
            answer = _exchange(
                sso,
                idp,
                {**person, "groups": [from_userinfo["name"]]},
                claims={"groups": [from_token["name"]]},
            )
        assert answer.status_code == 200, answer.text
        memberships = (
            account["id"] in group_member_ids(sso, from_userinfo),
            account["id"] in group_member_ids(sso, from_token),
        )

    assert memberships == (True, False)


def test_an_opaque_token_without_groups_still_signs_in(sso, idp, group):
    person = idp.sign_in_as()
    account = _signed_in_user(sso)

    with oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True):
        answer = _exchange(sso, idp, person, opaque=True)

    assert answer.status_code == 200, answer.text
    assert account["id"] not in group_member_ids(sso, group)


# --- an exchange with no readable groups claim keeps the groups (184eada9e) ----------------


@pytest.mark.parametrize("token", ["signed", "opaque"])
def test_an_exchange_without_a_groups_claim_is_refused_for_a_group_member(sso, idp, group, token):
    with oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True):
        person = idp.sign_in_as(groups=[group["name"]])
        account = _signed_in_user(sso)
        assert account["id"] in group_member_ids(sso, group)
        without_groups = {key: value for key, value in person.items() if key != "groups"}

        refused = _exchange(sso, idp, without_groups, opaque=token == "opaque")
        still_member = account["id"] in group_member_ids(sso, group)
        with_groups = _exchange(sso, idp, without_groups, claims={"groups": [group["name"]]})

    assert refused.status_code == 403, (
        f"an exchange whose {token} token and userinfo carry no groups claim signed in "
        f"(HTTP {refused.status_code}) (#33846)"
    )
    assert still_member, "an exchange with no groups claim took the account out of its group"
    assert with_groups.status_code == 200, with_groups.text
    assert account["id"] in group_member_ids(sso, group)


# --- blocked groups are not created (fcb0af3fd) ----------------------------------------------


def test_group_creation_skips_blocked_groups(sso, idp):
    tag = secrets.token_hex(3)
    blocked, allowed = f"blocked-{tag}", f"allowed-{tag}"
    with oauth_settings(
        sso,
        ENABLE_OAUTH_GROUP_MANAGEMENT=True,
        ENABLE_OAUTH_GROUP_CREATION=True,
        OAUTH_BLOCKED_GROUPS=f"blocked-{tag}*",
    ):
        idp.sign_in_as(groups=[blocked, allowed])
        account = _signed_in_user(sso)

    created = _groups_named(sso, blocked, allowed)
    try:
        assert [group["name"] for group in created if group["name"] == blocked] == [], (
            "a group matching OAUTH_BLOCKED_GROUPS was created at sign-in (#29558)"
        )
        [allowed_group] = [group for group in created if group["name"] == allowed]
        assert account["id"] in group_member_ids(sso, allowed_group)
    finally:
        _delete_groups(sso, created)


def test_an_existing_blocked_group_is_still_not_joined(sso, idp):
    tag = secrets.token_hex(3)
    with group_named(sso, f"blocked-{tag}") as blocked:
        with oauth_settings(
            sso,
            ENABLE_OAUTH_GROUP_MANAGEMENT=True,
            ENABLE_OAUTH_GROUP_CREATION=True,
            OAUTH_BLOCKED_GROUPS=f"blocked-{tag}",
        ):
            idp.sign_in_as(groups=[blocked["name"]])
            account = _signed_in_user(sso)

        assert account["id"] not in group_member_ids(sso, blocked)
