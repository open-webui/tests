"""Journey: the password rules set in the environment hold on every route that stores a password.

On an instance that checks new passwords against a pattern (at least 12 characters with a digit)
and gives a hint, sign-up, the admin's Add User, an account changing its own password and the
admin editing another account each refuse a password that breaks the rule with 400 and the hint.
Nothing changes: no account appears and the old password still signs in. A password that meets
the rule passes on each route and signs in afterwards.

Discriminates: passes on dev ebc6add67; in a backend copy where the password check skips the
pattern (the length limit stays), every refusal test goes red (the route answers 200 and the
account or the new password exists) and the four tests with a fitting password stay green.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import Actor, create_user
from harness.instance import LaunchedInstance

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

HINT = "Use at least 12 characters with a digit."
RULES = {
    "ENABLE_PASSWORD_VALIDATION": "true",
    "PASSWORD_VALIDATION_REGEX_PATTERN": r"^(?=.*\d).{12,}$",
    "PASSWORD_VALIDATION_HINT": HINT,
}
ADMIN_CONFIG = "/api/v1/auths/admin/config"
BROKEN = "short-1"
NO_DIGIT = "long-enough-but-no-digit"
FITTING = "fitting-password-42"


@pytest.fixture
def ruled(instance_with) -> LaunchedInstance:
    launched = instance_with(RULES)
    with launched.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        opened = client.post(
            ADMIN_CONFIG,
            json={**current.json(), "ENABLE_SIGNUP": True, "DEFAULT_USER_ROLE": "user"},
        )
    assert opened.status_code == 200, opened.text
    return launched


def _new_email() -> str:
    return f"rules-{uuid.uuid4().hex[:8]}@example.com"


def _sign_up(instance: LaunchedInstance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signup",
        json={"name": "New Person", "email": email, "password": password},
        timeout=60.0,
    )


def _sign_in(instance: LaunchedInstance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def _add(instance: LaunchedInstance, email: str, password: str) -> httpx.Response:
    with instance.client() as client:
        return client.post(
            "/api/v1/auths/add",
            json={"name": "Added Person", "email": email, "password": password, "role": "user"},
        )


def _change_own(account: Actor, new_password: str) -> httpx.Response:
    with account.client() as client:
        return client.post(
            "/api/v1/auths/update/password",
            json={"password": account.password, "new_password": new_password},
        )


def _edit_by_admin(instance: LaunchedInstance, account: Actor, password: str) -> httpx.Response:
    with instance.client() as client:
        return client.post(f"/api/v1/users/{account.id}/update", json={"password": password})


def _accounts_named(instance: LaunchedInstance, email: str) -> list[dict]:
    with instance.client() as client:
        found = client.get("/api/v1/users/", params={"query": email})
    found.raise_for_status()
    return found.json()["users"]


@pytest.mark.parametrize("password", [BROKEN, NO_DIGIT])
def test_a_sign_up_that_breaks_the_rule_is_refused_with_the_hint_and_creates_no_account(
    ruled, password
):
    email = _new_email()

    refused = _sign_up(ruled, email, password)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == HINT
    assert _accounts_named(ruled, email) == []
    assert _sign_in(ruled, email, password).status_code == 400


def test_a_sign_up_that_meets_the_rule_creates_the_account(ruled):
    email = _new_email()

    signed_up = _sign_up(ruled, email, FITTING)

    assert signed_up.status_code == 200, signed_up.text
    assert _sign_in(ruled, email, FITTING).status_code == 200


@pytest.mark.parametrize("password", [BROKEN, NO_DIGIT])
def test_an_admin_adding_an_account_with_a_password_that_breaks_the_rule_is_refused_with_the_hint(
    ruled, password
):
    email = _new_email()

    refused = _add(ruled, email, password)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == HINT
    assert _accounts_named(ruled, email) == []


def test_an_admin_adding_an_account_with_a_password_that_meets_the_rule_creates_it(ruled):
    email = _new_email()

    added = _add(ruled, email, FITTING)

    assert added.status_code == 200, added.text
    assert _sign_in(ruled, email, FITTING).status_code == 200


@pytest.mark.parametrize("password", [BROKEN, NO_DIGIT])
def test_changing_your_own_password_to_one_that_breaks_the_rule_is_refused_and_keeps_the_old_one(
    ruled, password
):
    account = create_user(ruled)

    refused = _change_own(account, password)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == HINT
    assert _sign_in(ruled, account.email, account.password).status_code == 200
    assert _sign_in(ruled, account.email, password).status_code == 400


def test_changing_your_own_password_to_one_that_meets_the_rule_replaces_the_old_one(ruled):
    account = create_user(ruled)

    changed = _change_own(account, FITTING)

    assert changed.status_code == 200, changed.text
    assert _sign_in(ruled, account.email, FITTING).status_code == 200
    assert _sign_in(ruled, account.email, account.password).status_code == 400


@pytest.mark.parametrize("password", [BROKEN, NO_DIGIT])
def test_an_admin_setting_another_accounts_password_to_one_that_breaks_the_rule_is_refused(
    ruled, password
):
    account = create_user(ruled)

    refused = _edit_by_admin(ruled, account, password)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == HINT
    assert _sign_in(ruled, account.email, account.password).status_code == 200
    assert _sign_in(ruled, account.email, password).status_code == 400


def test_an_admin_setting_another_accounts_password_to_one_that_meets_the_rule_replaces_it(ruled):
    account = create_user(ruled)

    edited = _edit_by_admin(ruled, account, FITTING)

    assert edited.status_code == 200, edited.text
    assert _sign_in(ruled, account.email, FITTING).status_code == 200
    assert _sign_in(ruled, account.email, account.password).status_code == 400
