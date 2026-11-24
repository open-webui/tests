"""Regression: who an SSO sign-in becomes, driven through a real OIDC sign-in.

open-webui 0.11.1 gathered these fixes to the provider sign-in path:

* `5462c02af` (#28065, issue #28062): joserfc rejected any JOSE header it did not know, so an
  ID token carrying a vendor header (CyberArk `app_id`) failed the sign-in. Unknown headers are
  now ignored; `crit` and the signature are still checked.
* `73c1f5806` (#28624): the SQLite sub lookup used `contains()`, a substring `LIKE`, so a sub
  like `abc_2345` or `%` signed in as someone else's account.
* `a6834f089` (#28954, issue #27760): a provider's numeric sub was stored as a JSON number,
  which SQLite never equals to a text sub, so the provider's back-channel logout (a JWT, its sub
  always text) never found the account. The sub is now stored as text. Signing in twice with a
  numeric sub keeps one account on both refs, since both sides of that lookup are numbers.
* `d799e81ed` / `e96844581`: token exchange skipped role and group mapping, and a provider that
  sent no roles claim reset an existing account to the default role.
* `c2107e5bb`: signout left the `owui-session` cookie and its server-side session behind.
* `aeda6ff13`: back-channel logout fetched the provider's keys with an anonymous `PyJWKClient`,
  so a provider whose key endpoint turns away such clients never signed anyone out. The keys
  now come through the provider's configured client.
* Issue #34922: OIDC lets a provider sign the ID token HS256 with the client secret, which no
  published key set carries. Open WebUI checked it against the provider's published keys only and
  every sign-in failed with "Algorithm 'HS256' requires 'oct' key". Red on dev dc9886799 until the
  fix (branch fix-oidc-hs256-id-token) merges.

Accounts linked by releases before `a6834f089` are written into the instance's database the way
those releases left them (the sub as a JSON number, other keys beside it) and then signed in to.

Twin of unit/security/test_oauth_identity.py.

Discriminates: passes on dev bbfa876af; with each fix reverted in a copy, the matching narrow
tests fail (the vendor header refuses the sign-in, the pattern sub lands on the victim's
account, the back-channel logout misses the numeric sub's session, the exchange leaves the group
unjoined, the admin comes back pending and signout leaves `owui-session` in the browser); with
a6834f089 reverted the JSON-number account is missed and its entry loses its other keys, with
73c1f5806 reverted the entry with other keys is missed, and with the key fetch put back on
`PyJWKClient` the logout answers 400.
"""

from __future__ import annotations

import secrets

import httpx
import pytest

from harness.actors import create_user
from harness.oidc_provider import (
    group_member_ids,
    group_named,
    oauth_entry,
    oauth_settings,
    session_user,
    serve,
    shared_provider,
    sign_in,
    sso_env,
    store_oauth_entry,
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


@pytest.fixture(scope="module")
def secret_signing_idp():
    """A provider that, like the one in #34922, advertises HS256 and publishes only an RSA key."""
    provider, shutdown = serve()
    provider.id_token_algs = ["HS256", "RS256", "none"]
    yield provider
    shutdown()


@pytest.fixture
def group(sso):
    with group_named(sso, f"team-{secrets.token_hex(4)}") as created:
        yield created


def signed_in_user(sso) -> dict:
    result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return session_user(sso, result.token)


# --------------------------------------------------------------------------- JOSE headers


@pytest.mark.parametrize("header_name", ["app_id", "client_id"])
def test_an_id_token_with_a_vendor_jose_header_signs_in(sso, idp, header_name):
    """Narrow (#28065): CyberArk's `app_id` (and CAS's `client_id`) is ignored, not fatal."""
    person = idp.sign_in_as(jose_header={header_name: "vendor-value"})
    assert signed_in_user(sso)["email"] == person["email"]


@pytest.mark.parametrize(
    "tampering",
    [
        {"jose_header": {"crit": ["app_id"], "app_id": "x"}},
        {"signature": "foreign-key"},
        {"signature": "client-secret"},
    ],
    ids=["unknown-critical-header", "foreign-key", "hs256-with-client-secret"],
)
def test_an_id_token_the_relaxation_must_not_accept_is_refused(sso, idp, tampering):
    """Nearby: ignoring unknown headers leaves `crit`, the signature and the alg list armed."""
    idp.sign_in_as(**tampering)
    result = sign_in(sso)
    assert result.token is None and result.error


def test_an_id_token_signed_with_the_client_secret_signs_in(instance_with, secret_signing_idp):
    """Narrow (#34922): HS256 checks against the client secret; another secret is still refused."""
    idp = secret_signing_idp
    sso = instance_with(sso_env(idp))

    person = idp.sign_in_as(signature="client-secret")
    assert signed_in_user(sso)["email"] == person["email"]

    idp.sign_in_as(signature="wrong-secret")
    forged = sign_in(sso)
    assert forged.token is None and forged.error, "an ID token signed with another secret signed in"

    person = idp.sign_in_as(signature="provider")
    assert signed_in_user(sso)["email"] == person["email"], "RS256 stopped working beside HS256"


# --------------------------------------------------------------------------- sub lookup


@pytest.mark.parametrize("pattern", ["{tag}_2345", "%"], ids=["underscore", "percent"])
def test_a_sub_that_is_a_like_pattern_gets_its_own_account(sso, idp, pattern):
    """Narrow (#28624): a sub is compared as a value, never as a LIKE pattern."""
    tag = secrets.token_hex(3)
    idp.sign_in_as(sub=f"{tag}12345")
    victim = signed_in_user(sso)

    newcomer = idp.sign_in_as(sub=pattern.format(tag=tag))
    arrived_as = signed_in_user(sso)

    assert arrived_as["id"] != victim["id"], "the pattern sub signed in as another account"
    assert arrived_as["email"] == newcomer["email"]


def sign_in_with_numeric_sub(idp, **claims) -> int:
    numeric_sub = 700_000_000 + secrets.randbelow(10**8)
    # without an email in the ID token the userinfo answer, with its numeric sub, is used
    idp.sign_in_as(sub=numeric_sub, id_token_claims={"sub": str(numeric_sub)}, **claims)
    return numeric_sub


def test_a_back_channel_logout_reaches_an_account_with_a_numeric_sub(sso, idp):
    """Narrow (#28954): the sub is stored as text, so the logout's string sub finds the account."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        numeric_sub = sign_in_with_numeric_sub(idp, roles=["user"])
        account = sign_in(sso)
    assert account.token, f"the sign-in failed: {account.error}"

    logout = httpx.post(
        f"{sso.base_url}/oauth/backchannel-logout",
        data={"logout_token": idp.logout_token(str(numeric_sub))},
        timeout=60.0,
    )

    assert logout.status_code == 200, logout.text
    with sso.client(account.token) as client:
        disconnect = client.delete("/api/v1/auths/oauth/sessions/oidc")
    # with Redis a logout that found the account also revokes its token
    expected = 401 if sso.redis_url else 404
    assert disconnect.status_code == expected, "the logout missed the account's SSO session"


def verified_sign_in(sso, idp):
    """A fresh person signed in with the user role, so their session may use the API."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["user"])
        account = sign_in(sso)
    assert account.token, f"the sign-in failed: {account.error}"
    return account


def test_a_back_channel_logout_fetches_the_keys_through_the_providers_client(sso, idp):
    """Narrow (aeda6ff13): a provider whose key endpoint refuses anonymous scripts still gets
    its logout honoured, because the keys are fetched the way sign-in fetches them."""
    account = verified_sign_in(sso, idp)
    sub = idp.claims["sub"]
    idp.jwks_refused_agent = "Python-urllib"

    logout = httpx.post(
        f"{sso.base_url}/oauth/backchannel-logout",
        data={"logout_token": idp.logout_token(sub)},
        timeout=60.0,
    )

    assert logout.status_code == 200, f"the logout was refused: {logout.text}"
    with sso.client(account.token) as client:
        disconnect = client.delete("/api/v1/auths/oauth/sessions/oidc")
    expected = 401 if sso.redis_url else 404
    assert disconnect.status_code == expected, "the logout missed the account's SSO session"


def test_a_logout_signed_with_a_key_the_provider_never_published_is_refused(sso, idp):
    """Nearby: fetching keys another way still checks the signature against them."""
    account = verified_sign_in(sso, idp)

    logout = httpx.post(
        f"{sso.base_url}/oauth/backchannel-logout",
        data={"logout_token": idp.logout_token(idp.claims["sub"], signature="foreign-key")},
        timeout=60.0,
    )

    assert logout.status_code == 400, logout.text
    with sso.client(account.token) as client:
        assert client.delete("/api/v1/auths/oauth/sessions/oidc").status_code == 200


def test_a_numeric_sub_signs_in_to_the_same_account_twice(sso, idp):
    """Nearby: a provider sending the sub as a JSON number keeps one account."""
    sign_in_with_numeric_sub(idp)
    first = signed_in_user(sso)
    assert signed_in_user(sso)["id"] == first["id"]


def test_the_same_sub_is_the_same_account_and_a_new_sub_a_new_one(sso, idp):
    """Nearby: the ordinary hit and miss are unchanged."""
    person = idp.sign_in_as()
    first = signed_in_user(sso)
    idp.sign_in_as(**person)
    assert signed_in_user(sso)["id"] == first["id"]
    idp.sign_in_as()
    assert signed_in_user(sso)["id"] != first["id"]


# --------------------------------------------------------------------------- legacy rows


def test_an_account_stored_with_a_numeric_sub_still_signs_in(sso, idp):
    """Narrow (#28954): the JSON number an older release stored matches the provider's text sub,
    and the sign-in rewrites it as text without dropping the entry's other keys."""
    numeric_sub = 700_000_000 + secrets.randbelow(10**8)
    account = create_user(sso)
    store_oauth_entry(sso, account.id, {"oidc": {"sub": numeric_sub, "email": account.email}})

    idp.sign_in_as(sub=str(numeric_sub), email=account.email)
    result = sign_in(sso)

    assert result.token, f"the account stored with a numeric sub could not sign in: {result.error}"
    assert session_user(sso, result.token)["id"] == account.id
    assert oauth_entry(sso, account.email) == {
        "oidc": {"sub": str(numeric_sub), "email": account.email}
    }, "the sign-in did not store the sub as text beside the entry's other keys"


def test_an_entry_with_keys_besides_the_sub_still_matches(sso, idp):
    """Narrow (#28624): the match is on the sub's value, not on the serialised entry."""
    sub = f"sub-{secrets.token_hex(4)}"
    account = create_user(sso)
    store_oauth_entry(sso, account.id, {"oidc": {"sub": sub, "email": account.email}})

    idp.sign_in_as(sub=sub, email=account.email)
    result = sign_in(sso)

    assert result.token, f"the entry with other keys was not matched: {result.error}"
    assert session_user(sso, result.token)["id"] == account.id


def test_linking_by_email_keeps_the_other_providers_entries(sso, idp):
    """Nearby: merging a sign-in into an account by email adds the provider's entry beside the
    others instead of replacing them."""
    account = create_user(sso)
    store_oauth_entry(sso, account.id, {"github": {"sub": "4242"}})
    sub = f"sub-{secrets.token_hex(4)}"

    with oauth_settings(sso, OAUTH_MERGE_ACCOUNTS_BY_EMAIL=True):
        idp.sign_in_as(sub=sub, email=account.email)
        result = sign_in(sso)

    assert result.token, f"the sign-in was not merged into the account: {result.error}"
    assert session_user(sso, result.token)["id"] == account.id
    assert oauth_entry(sso, account.email) == {"github": {"sub": "4242"}, "oidc": {"sub": sub}}


# --------------------------------------------------------------------------- roles


def test_an_admin_whose_provider_sends_no_roles_stays_admin(sso, idp):
    """Narrow: a sign-in without a roles claim keeps the account's role; roles still decide."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        person = idp.sign_in_as(roles=["admin"])
        assert signed_in_user(sso)["role"] == "admin"

        idp.sign_in_as(**{key: value for key, value in person.items() if key != "roles"})
        assert signed_in_user(sso)["role"] == "admin", "the admin was reset to the default role"

        idp.sign_in_as(**{**person, "roles": ["user"]})
        assert signed_in_user(sso)["role"] == "user"


def test_a_new_account_without_roles_gets_the_default_role(sso, idp):
    """Nearby: with no account yet, the default role is still the fallback."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as()
        assert signed_in_user(sso)["role"] == "pending"


# --------------------------------------------------------------------------- token exchange


def exchange(sso, idp, person: dict, **claims) -> httpx.Response:
    """Sign in from another application with a provider access token for `person`."""
    token = idp.issue_access_token({**person, **claims})
    return httpx.post(f"{sso.base_url}{EXCHANGE}", json={"token": token}, timeout=60.0)


@pytest.mark.parametrize("group_mapping", [True, False], ids=["mapping-on", "mapping-off"])
def test_token_exchange_applies_the_providers_groups(sso, idp, group, group_mapping):
    """Narrow: signing in from another application runs group mapping like the callback does."""
    person = idp.sign_in_as()
    account = signed_in_user(sso)
    with oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=group_mapping):
        answer = exchange(sso, idp, person, groups=[group["name"]])
    assert answer.status_code == 200, answer.text
    assert (account["id"] in group_member_ids(sso, group)) is group_mapping


def test_token_exchange_still_refuses_a_disallowed_email_domain(sso, idp):
    """Nearby: the domain allowlist bites on the exchange path too."""
    person = idp.sign_in_as()
    signed_in_user(sso)
    with oauth_settings(sso, OAUTH_ALLOWED_DOMAINS="example.org"):
        assert exchange(sso, idp, person).status_code == 403


# --------------------------------------------------------------------------- signout


def test_signing_out_drops_the_session_cookie_and_the_sign_in_cookies(sso, idp):
    """Narrow: signout clears the server session behind `owui-session` and the cookie itself."""
    idp.sign_in_as()
    result = sign_in(sso)
    browser = result.browser
    browser.get("/oauth/oidc/login")  # a second sign-in started, so the session holds a state
    assert "owui-session" in browser.cookies

    signed_out = browser.post("/api/v1/auths/signout")

    assert signed_out.status_code == 200, signed_out.text
    assert "owui-session" not in browser.cookies, "signout left the session cookie behind"
    assert "token" not in browser.cookies
    deleted = {header.split("=", 1)[0] for header in signed_out.headers.get_list("set-cookie")}
    assert {"token", "owui-session", "oui-session", "oauth_id_token"} <= deleted
