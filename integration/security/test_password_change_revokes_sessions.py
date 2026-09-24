"""Regression: changing a password left every other signed-in session working.

open-webui 0.11.1 fix `21e390561` (#28725): a password change wrote nothing to the token
revocation list, so every session issued before it kept working until its JWT expired, four
weeks by default. Both password paths, `POST /api/v1/auths/update/password` and an admin reset
through `POST /api/v1/users/{id}/update`, now stamp the per-user `revoked_at` marker that token
validation already checks, for as long as a token can live. The marker lives in Redis; without
Redis nothing can be revoked and the backend now logs a warning naming the user.

The revocation tests run on an instance of their own backed by `StatefulRedis`; the no-Redis
warning is read from the shared instance's log.

Twin of unit/security/test_password_change_revokes_sessions.py.

Discriminates: passes on dev bbfa876af, fails with both `revoke_user_tokens` calls removed
from the password routes (earlier sessions keep answering 200, no marker is stored and no
warning is logged), with the marker kept for a fixed 30 days (it expires before the 8-week
tokens) and with the admin reset revoking even when no password was written.
"""

from __future__ import annotations

import base64
import json
import time
import uuid

import httpx
import pytest

from harness.actors import create_user, sign_in
from integration.stateful_redis import StatefulRedis

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NEW_PASSWORD = "N3w-password-after-the-change"
PASSWORD_CHANGES = ["self-service", "admin reset"]
SCIM_TOKEN = "scim-provisioning-token"


@pytest.fixture(scope="session")
def revocation_store():
    store = StatefulRedis()
    yield store
    store.close()


@pytest.fixture
def redis_instance(revocation_store, instance_with):
    return instance_with(
        {
            "REDIS_URL": revocation_store.url,
            # Tokens that outlive the fixed 30 days the marker used to be kept for.
            "JWT_EXPIRES_IN": "8w",
            # Provisions accounts without a password row, as single sign-on does.
            "ENABLE_SCIM": "true",
            "SCIM_TOKEN": SCIM_TOKEN,
            "SCIM_AUTH_PROVIDER": "oidc",
        }
    )


def _session_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/auths/").status_code


def _change_own_password(instance, account, current_password: str) -> httpx.Response:
    with instance.client(account.token) as client:
        return client.post(
            "/api/v1/auths/update/password",
            json={"password": current_password, "new_password": NEW_PASSWORD},
        )


def _change_password(instance, account, change: str) -> None:
    if change == "self-service":
        response = _change_own_password(instance, account, account.password)
    else:
        with instance.client() as admin_client:
            response = admin_client.post(
                f"/api/v1/users/{account.id}/update", json={"password": NEW_PASSWORD}
            )
    assert response.status_code == 200, f"{change} failed: {response.text}"


def _expiry_of(token: str) -> int:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]


def _keys_naming(store: StatefulRedis, user_id: str) -> list[str]:
    return [key for key in store.keys() if user_id in key]


@pytest.mark.parametrize("change", PASSWORD_CHANGES)
def test_a_password_change_signs_out_every_earlier_session(redis_instance, change):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)
    bystander = create_user(redis_instance)

    _change_password(redis_instance, account, change)

    assert _session_status(redis_instance, other_device) == 401, (
        f"a session issued before the {change} still authenticates: a token stolen under the "
        "old password keeps working for the whole JWT lifetime (#28725)"
    )
    assert _session_status(redis_instance, account.token) == 401
    assert _session_status(redis_instance, bystander.token) == 200, (
        f"the {change} of one account signed another account out"
    )


@pytest.mark.parametrize("change", PASSWORD_CHANGES)
def test_the_revocation_outlives_every_token_it_revokes(redis_instance, revocation_store, change):
    account = create_user(redis_instance)
    token_expiry = _expiry_of(account.token)

    _change_password(redis_instance, account, change)

    markers = _keys_naming(revocation_store, account.id)
    assert markers, f"the {change} stored no revocation for the account (#28725)"
    for marker in markers:
        expires_at = revocation_store.expires_at(marker)
        assert expires_at is None or expires_at >= token_expiry, (
            f"the {change} revocation expires before the tokens it revokes: once it is gone "
            "every one of them authenticates again (#28725)"
        )


def test_a_wrong_current_password_is_refused_and_signs_nothing_out(
    redis_instance, revocation_store
):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)

    refused = _change_own_password(redis_instance, account, "not-the-current-password")

    assert refused.status_code == 400
    assert _session_status(redis_instance, account.token) == 200
    assert _session_status(redis_instance, other_device) == 200, (
        "a refused password change signed the account out: a stolen session could log the "
        "real user out at will"
    )
    assert _keys_naming(revocation_store, account.id) == []


def test_a_reset_that_writes_no_password_revokes_nothing(redis_instance, revocation_store):
    """Nothing changed for an account without a password row, so its sessions must survive."""
    suffix = uuid.uuid4().hex[:8]
    email = f"sso-{suffix}@example.com"
    provisioned = httpx.post(
        f"{redis_instance.base_url}/api/v1/scim/v2/Users",
        headers={"Authorization": f"Bearer {SCIM_TOKEN}"},
        json={
            "userName": email,
            "displayName": f"SSO {suffix}",
            "emails": [{"value": email, "primary": True}],
        },
        timeout=60.0,
    )
    assert provisioned.status_code == 201, provisioned.text
    account_id = provisioned.json()["id"]

    with redis_instance.client() as admin_client:
        reset = admin_client.post(
            f"/api/v1/users/{account_id}/update", json={"password": NEW_PASSWORD}
        )

    assert reset.status_code == 200, reset.text
    assert _keys_naming(revocation_store, account_id) == [], (
        "an admin reset that wrote no password still revoked the account's sessions, signing a "
        "single sign-on user out of everything for nothing"
    )


def test_signing_in_with_the_new_password_works_after_the_change(redis_instance):
    account = create_user(redis_instance)
    _change_password(redis_instance, account, "self-service")
    changed_in_second = int(time.time())

    # The marker has one-second resolution and also rejects a token issued in its own second.
    time.sleep(max(0.0, changed_in_second + 1 - time.time()))
    fresh = sign_in(redis_instance, account.email, NEW_PASSWORD)
    stale = httpx.post(
        f"{redis_instance.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": account.password},
        timeout=60.0,
    )

    assert _session_status(redis_instance, fresh) == 200, (
        "the session from signing in with the new password was rejected: the account is locked "
        "out after its own password change"
    )
    assert stale.status_code == 400


def test_signing_out_one_session_leaves_the_others(redis_instance):
    account = create_user(redis_instance)
    other_device = sign_in(redis_instance, account.email, account.password)

    with redis_instance.client(account.token) as client:
        client.post("/api/v1/auths/signout").raise_for_status()

    assert _session_status(redis_instance, account.token) == 401
    assert _session_status(redis_instance, other_device) == 200, (
        "signing out one device signed out the account's other sessions too"
    )


def test_without_redis_the_change_logs_that_nothing_was_revoked(instance, make_user):
    account = make_user()
    other_device = sign_in(instance, account.email, account.password)
    log_offset = instance.log_size()

    _change_password(instance, account, "self-service")

    # Without Redis the earlier session necessarily survives, which is why the warning exists.
    assert _session_status(instance, other_device) == 200
    deadline = time.monotonic() + 10
    warnings: list[str] = []
    while not warnings and time.monotonic() < deadline:
        log_lines = instance.log_since(log_offset).splitlines()
        warnings = [line for line in log_lines if "WARNING" in line and account.id in line]
        time.sleep(0.1)
    assert any("redis" in line.lower() for line in warnings), (
        "a password change without Redis revoked nothing and logged nothing naming the account: "
        "the operator believes the other sessions were signed out while every one keeps "
        "working (#28725)"
    )
