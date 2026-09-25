"""Journey: what an OIDC sign-in writes onto the account, and the provider's back-channel logout.

With group management on, each sign-in makes the account's memberships match the provider's
groups claim, sent as a list or as one `;`-separated string; groups the claim leaves out are
left, and an absent or empty claim changes nothing. Group creation adds a missing group for the
claim. The name and picture come from the provider when the account is created and, with the
update switches on, again on every later sign-in; the picture is fetched with the provider's
access token and stored as an image, and anything that is not an image falls back to the default
avatar. Role mapping reads a separated string or a nested claim, honours the configured role
names and `*`, and refuses a sign-in whose roles match none of them without creating the account.

The provider's back-channel logout ends the SSO sessions of the account whose `sub` it names and
no one else's. A logout token that is unsigned by the provider, meant for another client, lacks
the logout event, carries a nonce or names nobody is refused and ends nothing.

Discriminates: in a backend copy, splitting a string groups claim on `,` instead of the
configured separator turns the separated-string test red, dropping the name update on login
turns the name test red, and decoding the logout token unverified turns the foreign-key and
other-audience cases red.
"""

from __future__ import annotations

import base64
import secrets

import httpx
import pytest

from harness.listener import text_answer
from harness.oidc_provider import (
    group_member_ids,
    group_named,
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
LOGOUT = "/oauth/backchannel-logout"


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    # the provider's picture is served from localhost
    return instance_with({**sso_env(idp), "ENABLE_LOCAL_WEB_FETCH": "true"})


@pytest.fixture
def pictures(listener):
    listener.route("GET", "/avatar.png", (200, {"Content-Type": "image/png"}, PNG))
    listener.route("GET", "/other.png", (200, {"Content-Type": "image/png"}, PNG[::-1]))
    listener.route("GET", "/page.html", text_answer("<html>not a picture</html>"))
    return listener


def signed_in_user(sso) -> dict:
    result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return session_user(sso, result.token)


def is_member(sso, account: dict, group: dict) -> bool:
    return account["id"] in group_member_ids(sso, group)


# --------------------------------------------------------------------------- groups


@pytest.fixture
def red_and_blue(sso):
    with (
        group_named(sso, f"red-{secrets.token_hex(3)}") as red,
        group_named(sso, f"blue-{secrets.token_hex(3)}") as blue,
        oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True),
    ):
        yield red, blue


def test_a_groups_list_joins_and_a_later_sign_in_leaves_what_it_drops(sso, idp, red_and_blue):
    red, blue = red_and_blue
    person = idp.sign_in_as(groups=[red["name"], blue["name"]])
    account = signed_in_user(sso)
    joined = (is_member(sso, account, red), is_member(sso, account, blue))

    idp.sign_in_as(**{**person, "groups": [blue["name"]]})
    signed_in_user(sso)
    after_drop = (is_member(sso, account, red), is_member(sso, account, blue))

    assert joined == (True, True)
    assert after_drop == (False, True), "a group the claim left out kept the account"


@pytest.mark.parametrize("claim", [None, []], ids=["absent", "empty"])
def test_an_absent_or_empty_groups_claim_changes_nothing(sso, idp, red_and_blue, claim):
    red, _ = red_and_blue
    person = idp.sign_in_as(groups=[red["name"]])
    account = signed_in_user(sso)

    later = {key: value for key, value in person.items() if key != "groups"}
    idp.sign_in_as(**later, **({} if claim is None else {"groups": claim}))
    signed_in_user(sso)

    assert is_member(sso, account, red)


def test_a_separated_groups_string_is_split(sso, idp, red_and_blue):
    red, blue = red_and_blue
    idp.sign_in_as(groups=f"{red['name']};{blue['name']}")
    account = signed_in_user(sso)

    assert (is_member(sso, account, red), is_member(sso, account, blue)) == (True, True)


def test_a_single_group_string_joins_that_group(sso, idp, red_and_blue):
    red, blue = red_and_blue
    idp.sign_in_as(groups=red["name"])
    account = signed_in_user(sso)

    assert (is_member(sso, account, red), is_member(sso, account, blue)) == (True, False)


@pytest.mark.parametrize("creation", [True, False], ids=["creation-on", "creation-off"])
def test_group_creation_adds_a_missing_group_with_the_member(sso, idp, creation):
    name = f"from-idp-{secrets.token_hex(3)}"
    with oauth_settings(
        sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True, ENABLE_OAUTH_GROUP_CREATION=creation
    ):
        idp.sign_in_as(groups=[name])
        account = signed_in_user(sso)

    with sso.client() as admin:
        created = [group for group in admin.get("/api/v1/groups/").json() if group["name"] == name]
        try:
            if not creation:
                assert created == [], "a group was created with creation off"
                return
            [group] = created
            assert account["id"] in group_member_ids(sso, group)
        finally:
            for group in created:
                admin.delete(f"/api/v1/groups/id/{group['id']}/delete")


# --------------------------------------------------------------------------- name and picture


@pytest.mark.parametrize("update", [True, False], ids=["update-on", "update-off"])
def test_the_name_follows_the_provider_when_updates_are_on(sso, idp, update):
    person = idp.sign_in_as(name="Ada Before")
    account = signed_in_user(sso)

    with oauth_settings(sso, OAUTH_UPDATE_NAME_ON_LOGIN=update):
        idp.sign_in_as(**{**person, "name": "Ada After"})
        later = signed_in_user(sso)

    assert later["id"] == account["id"]
    assert later["name"] == ("Ada After" if update else "Ada Before")


def profile_image(sso, account: dict) -> bytes:
    with sso.client() as admin:
        answer = admin.get(f"/api/v1/users/{account['id']}/profile/image")
    answer.raise_for_status()
    return answer.content


def test_a_new_account_gets_the_providers_picture(sso, idp, pictures):
    idp.sign_in_as(picture=f"{pictures.base_url}/avatar.png")
    account = signed_in_user(sso)

    assert account["profile_image_url"].startswith("data:image/png;base64,")
    assert profile_image(sso, account) == PNG
    [fetch] = pictures.requests_to("/avatar.png")
    assert fetch.headers["Authorization"] == f"Bearer {idp.issued[-1]['access_token']}"


@pytest.mark.parametrize("update", [True, False], ids=["update-on", "update-off"])
def test_the_picture_follows_the_provider_when_updates_are_on(sso, idp, pictures, update):
    person = idp.sign_in_as(picture=f"{pictures.base_url}/avatar.png")
    signed_in_user(sso)

    with oauth_settings(sso, OAUTH_UPDATE_PICTURE_ON_LOGIN=update):
        idp.sign_in_as(**{**person, "picture": f"{pictures.base_url}/other.png"})
        later = signed_in_user(sso)

    assert profile_image(sso, later) == (PNG[::-1] if update else PNG)


def test_a_picture_that_is_not_an_image_becomes_the_default_avatar(sso, idp, pictures):
    idp.sign_in_as(picture=f"{pictures.base_url}/page.html")
    assert signed_in_user(sso)["profile_image_url"] == "/user.png"


# --------------------------------------------------------------------------- roles


@pytest.mark.parametrize(
    ("settings", "claims", "role"),
    [
        ({}, {"roles": "staff,admin"}, "admin"),
        (
            {"OAUTH_ROLES_CLAIM": "realm_access.roles"},
            {"realm_access": {"roles": ["admin"]}},
            "admin",
        ),
        ({"OAUTH_ALLOWED_ROLES": "*"}, {"roles": ["contractor"]}, "user"),
        ({"OAUTH_ADMIN_ROLES": "owners"}, {"roles": ["owners"]}, "admin"),
        ({"OAUTH_ADMIN_ROLES": "owners"}, {"roles": ["admin", "user"]}, "user"),
    ],
    ids=["separated-string", "nested-claim", "wildcard", "custom-admin-role", "renamed-admin"],
)
def test_role_mapping_reads_the_claim_as_configured(sso, idp, settings, claims, role):
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True, **settings):
        idp.sign_in_as(**claims)
        assert signed_in_user(sso)["role"] == role


def test_roles_matching_nothing_refuse_the_sign_in_and_create_no_account(sso, idp):
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        person = idp.sign_in_as(roles=["guest"])
        result = sign_in(sso)

    assert result.token is None and result.error
    with sso.client() as admin:
        found = admin.get("/api/v1/users/", params={"query": person["email"]}).json()["users"]
    assert found == []


# --------------------------------------------------------------------------- back-channel logout


def has_sso_session(sso, account: dict) -> bool:
    with sso.client() as admin:
        return admin.get(f"/api/v1/users/{account['id']}/oauth/sessions").status_code == 200


def logout(sso, **form) -> httpx.Response:
    return httpx.post(f"{sso.base_url}{LOGOUT}", data=form, timeout=60.0)


@pytest.fixture
def two_people(sso, idp):
    """Two accounts signed in through the provider, each holding an SSO session."""
    leaving = idp.sign_in_as()
    leaving_account = signed_in_user(sso)
    idp.sign_in_as()
    staying_account = signed_in_user(sso)
    assert has_sso_session(sso, leaving_account) and has_sso_session(sso, staying_account)
    return leaving, leaving_account, staying_account


def test_a_logout_ends_that_accounts_sessions_and_no_one_elses(sso, idp, two_people):
    person, leaving, staying = two_people

    answer = logout(sso, logout_token=idp.logout_token(person["sub"]))

    assert answer.status_code == 200, answer.text
    assert not has_sso_session(sso, leaving), "the logout left the account's SSO session"
    assert has_sso_session(sso, staying), "the logout ended someone else's session"


def test_a_logout_naming_an_unknown_sub_or_only_a_sid_ends_nothing(sso, idp, two_people):
    _, leaving, _ = two_people

    unknown = logout(sso, logout_token=idp.logout_token(f"sub-{secrets.token_hex(4)}"))
    sid_only = logout(sso, logout_token=idp.logout_token(None, sid=secrets.token_hex(8)))

    assert (unknown.status_code, sid_only.status_code) == (200, 200)
    assert has_sso_session(sso, leaving)


REFUSED_TOKENS = {
    "foreign-key": lambda idp, sub: idp.logout_token(sub, signature="foreign-key"),
    "other-audience": lambda idp, sub: idp.logout_token(sub, aud="another-client"),
    "unknown-issuer": lambda idp, sub: idp.logout_token(sub, iss="https://idp.invalid"),
    "no-logout-event": lambda idp, sub: idp.logout_token(sub, events={"other": {}}),
    "nonce": lambda idp, sub: idp.logout_token(sub, nonce="n-0"),
    "neither-sub-nor-sid": lambda idp, sub: idp.logout_token(None),
    "not-a-jwt": lambda idp, sub: "not.a.jwt",
}


@pytest.mark.parametrize("make_token", REFUSED_TOKENS.values(), ids=REFUSED_TOKENS.keys())
def test_a_logout_token_that_fails_a_check_is_refused_and_ends_nothing(
    sso, idp, two_people, make_token
):
    person, leaving, _ = two_people

    answer = logout(sso, logout_token=make_token(idp, person["sub"]))

    assert answer.status_code == 400, answer.text
    assert answer.json()["error"] == "invalid_request"
    assert has_sso_session(sso, leaving), "a refused logout token ended the session"


def test_a_logout_without_a_token_is_refused(sso):
    answer = logout(sso)
    assert answer.status_code == 400
    assert answer.json()["error"] == "invalid_request"
