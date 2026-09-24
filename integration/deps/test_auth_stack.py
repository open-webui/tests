"""Dependency smoke: the sign-in stack, each library driven through the feature that uses it.

bcrypt (the default) and argon2-cffi hash the password at sign-up and check it at sign-in.
PyJWT signs and checks the session token, whose `iat` and `exp` come from pytz's UTC clock.
authlib builds the SSO redirect and completes the code exchange, while itsdangerous signs the
`owui-session` cookie that carries its state from one to the other. The completed SSO sign-in
also verifies the provider's RS256 ID token and encrypts the stored OAuth session, which is
what cryptography does on this path (the OAuth twins under integration/security complete many
more). A bump that breaks one of them fails a sign-in here, not only an API check in unit/deps.

The argon2 instance runs in a zone far from UTC, so a clock that is not UTC shows in the token.

Discriminates: passes on dev bbfa876af; in a backend copy, `bcrypt.checkpw` answering True lets
the wrong password in, argon2 verification answering True does the same on its instance,
`jwt.decode` without signature and expiry checks accepts the flipped and the expired token, a
naive `datetime.now()` for `exp` stretches the lifetime by the zone's 5 h 45 min and dropping
`SessionMiddleware` fails the SSO sign-in at its first step.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
import uuid

import httpx
import pytest

from harness.actors import sign_in
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD, LaunchedInstance
from harness.oidc_provider import browser_for, session_user, shared_provider, sso_env

pytestmark = [
    pytest.mark.depcheck,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
# a zone off UTC by hours and minutes, so a naive clock cannot pass for UTC
ARGON2_OFF_UTC = {"PASSWORD_HASH_ALGORITHM": "argon2", "TZ": "Asia/Kathmandu"}
# past bcrypt's 72 bytes, so only a hash of the whole password tells the last byte apart
LONG_PASSWORD = "argon2-" + "x" * 72 + "!"
FOUR_WEEKS = 4 * 7 * 24 * 3600


@pytest.fixture
def argon2_instance(instance_with):
    return instance_with(ARGON2_OFF_UTC)


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def _sign_in(instance: LaunchedInstance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def _session_status(instance: LaunchedInstance, token: str) -> int:
    # not /api/v1/auths/, which checks `exp` itself and would hide PyJWT's own check
    with instance.client(token) as client:
        return client.get("/api/v1/users/user/settings").status_code


def _claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def _flip_middle(segment: str) -> str:
    """One base64url character changed mid-segment, where every bit counts."""
    middle = len(segment) // 2
    replacement = "A" if segment[middle] != "A" else "B"
    return segment[:middle] + replacement + segment[middle + 1 :]


def _with_flipped_signature(signed: str) -> str:
    *signed_part, signature = signed.split(".")
    return ".".join([*signed_part, _flip_middle(signature)])


def _save_admin_config(instance: LaunchedInstance, **changes) -> None:
    with instance.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def test_bcrypt_checks_the_password_at_sign_in(instance, make_user):
    account = make_user(password="bcrypt-password-1")

    assert _sign_in(instance, account.email, "bcrypt-password-1").status_code == 200
    assert _sign_in(instance, account.email, "bcrypt-password-2").status_code == 400


def test_argon2_hashes_the_whole_password_at_sign_up(argon2_instance, preserve):
    preserve("admin_config", on=argon2_instance)
    _save_admin_config(argon2_instance, ENABLE_SIGNUP=True)
    email = f"argon2-{uuid.uuid4().hex[:8]}@example.com"

    signed_up = httpx.post(
        f"{argon2_instance.base_url}/api/v1/auths/signup",
        json={"name": "Argon", "email": email, "password": LONG_PASSWORD},
        timeout=60.0,
    )

    assert signed_up.status_code == 200, signed_up.text
    assert _sign_in(argon2_instance, email, LONG_PASSWORD).status_code == 200
    assert _sign_in(argon2_instance, email, LONG_PASSWORD[:-1] + "?").status_code == 400


def test_a_session_token_with_a_flipped_signature_is_refused(instance, make_user):
    account = make_user()

    assert _session_status(instance, account.token) == 200
    assert _session_status(instance, _with_flipped_signature(account.token)) == 401


def test_a_session_token_stops_working_once_it_expires(instance, make_user, preserve):
    account = make_user()
    preserve("admin_config")
    _save_admin_config(instance, JWT_EXPIRES_IN="2s")

    token = sign_in(instance, account.email, account.password)
    claims = _claims(token)

    assert claims["exp"] - claims["iat"] in (1, 2)
    assert _session_status(instance, token) == 200
    time.sleep(3)  # the token's own lifetime: nothing to wait on but the clock
    assert _session_status(instance, token) == 401


def test_the_session_token_runs_four_weeks_from_now_in_utc(argon2_instance):
    signed_in = _sign_in(argon2_instance, ADMIN_EMAIL, ADMIN_PASSWORD)
    signed_in.raise_for_status()
    claims = _claims(signed_in.json()["token"])

    assert abs(claims["iat"] - time.time()) < 60, "the token's iat is not the current UTC time"
    assert abs(claims["exp"] - claims["iat"] - FOUR_WEEKS) <= 1


def test_sso_starts_at_the_provider_with_a_signed_session_cookie(sso, idp):
    with browser_for(sso) as browser:
        start = browser.get("/oauth/oidc/login")
        session_cookie = browser.cookies.get("owui-session")

    assert start.status_code == 302, start.text
    authorize = urllib.parse.urlsplit(start.headers["location"])
    query = dict(urllib.parse.parse_qsl(authorize.query))
    assert f"{authorize.scheme}://{authorize.netloc}{authorize.path}" == f"{idp.base_url}/authorize"
    assert query["client_id"] == idp.client_id
    assert query["response_type"] == "code"
    assert query["state"]
    # itsdangerous: data.timestamp.signature
    assert session_cookie and session_cookie.count(".") == 2


def _approved_callback(sso: LaunchedInstance) -> tuple[str, str]:
    """The provider's redirect back to Open WebUI, and the session cookie set on the way out."""
    with browser_for(sso) as browser:
        start = browser.get("/oauth/oidc/login")
        approved = browser.get(start.headers["location"])
        return approved.headers["location"], browser.cookies["owui-session"]


def _finish(callback_url: str, session_cookie: str) -> httpx.Response:
    return httpx.get(
        callback_url, headers={"Cookie": f"owui-session={session_cookie}"}, timeout=60.0
    )


def test_a_forged_session_cookie_fails_the_sso_callback_and_an_intact_one_signs_in(sso, idp):
    person = idp.sign_in_as()
    callback_url, session_cookie = _approved_callback(sso)
    forged = _finish(callback_url, _with_flipped_signature(session_cookie))

    assert "token" not in forged.cookies, "a session cookie with a broken signature was trusted"
    assert "error=" in forged.headers["location"]

    callback_url, session_cookie = _approved_callback(sso)
    token = _finish(callback_url, session_cookie).cookies.get("token")
    assert token, "the SSO sign-in with the intact session cookie failed"
    assert session_user(sso, token)["email"] == person["email"]
