"""Journey: who may sign up, and the role a new account starts with.

The first account on an instance becomes its admin and switches self-service sign-up off. With
sign-up switched back on, a later account gets the default user role, `pending` unless the admin
changed it, and a pending account can see its own session but is refused on every user route
until an admin lets it in. With sign-up off the form is refused and no account appears, while an
admin can still add accounts of any role from the admin panel.

Discriminates: fails with the `ui.enable_signup` check in the signup route removed (a sign-up
with sign-up off is answered 200 and the account exists).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import create_user
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def save_admin_config(admin, **changes) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def sign_up(instance, email: str | None = None) -> httpx.Response:
    """Fill in the sign-up form, without a session."""
    email = email or f"signup-{uuid.uuid4().hex[:8]}@example.com"
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signup",
        json={"name": "New Person", "email": email, "password": "signup-password-1"},
        timeout=60.0,
    )


def sign_in(instance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def user_route_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/chats/").status_code


@pytest.fixture
def signup_open(admin, preserve):
    preserve("admin_config")
    save_admin_config(admin, ENABLE_SIGNUP=True)


def test_the_first_account_is_the_admin_and_sign_up_starts_closed(instance, admin):
    signed_in = sign_in(instance, ADMIN_EMAIL, ADMIN_PASSWORD)

    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["role"] == "admin"
    with admin.client() as client:
        assert client.get(ADMIN_CONFIG).json()["ENABLE_SIGNUP"] is False
    assert sign_up(instance).status_code == 403


def test_a_later_sign_up_is_pending_and_refused_on_user_routes(instance, signup_open):
    signed_up = sign_up(instance)

    assert signed_up.status_code == 200, signed_up.text
    session = signed_up.json()
    assert session["role"] == "pending"
    with instance.client(session["token"]) as client:
        assert client.get("/api/v1/auths/").json()["role"] == "pending"
    assert user_route_status(instance, session["token"]) == 401


def test_a_pending_account_gets_in_once_an_admin_approves_it(instance, admin, signup_open):
    session = sign_up(instance).json()

    with admin.client() as client:
        approved = client.post(f"/api/v1/users/{session['id']}/update", json={"role": "user"})

    assert approved.status_code == 200, approved.text
    assert user_route_status(instance, session["token"]) == 200


def test_a_sign_up_gets_the_default_role_the_admin_chose(instance, admin, signup_open):
    save_admin_config(admin, DEFAULT_USER_ROLE="user")

    signed_up = sign_up(instance)

    assert signed_up.status_code == 200, signed_up.text
    assert signed_up.json()["role"] == "user"
    assert user_route_status(instance, signed_up.json()["token"]) == 200


def test_sign_up_switched_off_is_refused_and_creates_no_account(instance, admin, preserve):
    preserve("admin_config")
    save_admin_config(admin, ENABLE_SIGNUP=False)
    email = f"refused-{uuid.uuid4().hex[:8]}@example.com"

    refused = sign_up(instance, email)

    assert refused.status_code == 403, refused.text
    assert sign_in(instance, email, "signup-password-1").status_code == 400
    with admin.client() as client:
        listed = client.get("/api/v1/users/", params={"query": email}).json()["users"]
    assert listed == []


@pytest.mark.parametrize("role", ["pending", "user", "admin"])
def test_the_admin_adds_accounts_with_sign_up_switched_off(instance, admin, preserve, role):
    preserve("admin_config")
    save_admin_config(admin, ENABLE_SIGNUP=False)

    added = create_user(instance, role=role)

    assert sign_in(instance, added.email, added.password).json()["role"] == role
    expected = 401 if role == "pending" else 200
    assert user_route_status(instance, added.token) == expected


def test_an_email_already_taken_cannot_sign_up_again(instance, make_user, signup_open):
    existing = make_user()

    assert sign_up(instance, existing.email.upper()).status_code == 400
