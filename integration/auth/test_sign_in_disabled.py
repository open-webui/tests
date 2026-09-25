"""Journey: an instance started with `WEBUI_AUTH=False`, the single-user mode without sign-in.

The web client sees `auth: false` and signs in with an empty form; the first such sign-in creates
the one account, `admin@localhost`, as admin, and every later sign-in returns that same account
whatever the form says. Sign-up is refused once the account exists, and a request without the
session token is still refused, so the API never opens to anonymous callers. An instance that
already holds other accounts refuses the sign-in, since authentication can only be turned off on
a fresh installation.

Discriminates: in a backend copy, letting the `WEBUI_AUTH=False` sign-in read the form's email
and password turns the any-credentials test red, and dropping its existing-users check turns the
existing-accounts test red (the sign-in creates a second account).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import create_user

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

NO_AUTH = {"WEBUI_AUTH": "False"}
SOLE_ACCOUNT = "admin@localhost"


@pytest.fixture
def open_instance(instance_with):
    return instance_with(NO_AUTH)


def sign_in(instance, email: str = "", password: str = "") -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def test_the_client_is_told_there_is_no_sign_in(open_instance):
    features = httpx.get(f"{open_instance.base_url}/api/config", timeout=60.0).json()["features"]
    assert features["auth"] is False
    assert features["enable_signup"] is False


def test_the_empty_sign_in_is_the_sole_admin_account(open_instance):
    signed_in = sign_in(open_instance)

    assert signed_in.status_code == 200, signed_in.text
    session = signed_in.json()
    assert (session["email"], session["role"]) == (SOLE_ACCOUNT, "admin")
    with open_instance.client() as seeded:
        assert seeded.get("/api/v1/auths/").json()["id"] == session["id"]
    with open_instance.client(session["token"]) as client:
        assert client.get("/api/v1/users/").status_code == 200


def test_any_credentials_sign_in_to_that_same_account(open_instance):
    first = sign_in(open_instance).json()
    added = create_user(open_instance)

    as_someone_else = sign_in(open_instance, added.email, added.password)
    with_nonsense = sign_in(open_instance, "nobody@example.com", "wrong")

    assert as_someone_else.json()["id"] == first["id"]
    assert with_nonsense.json()["id"] == first["id"]


def test_sign_up_is_refused_and_creates_no_account(open_instance):
    email = f"late-{uuid.uuid4().hex[:8]}@example.com"

    refused = httpx.post(
        f"{open_instance.base_url}/api/v1/auths/signup",
        json={"name": "Late", "email": email, "password": "signup-password-1"},
        timeout=60.0,
    )

    assert refused.status_code == 403, refused.text
    with open_instance.client() as client:
        assert client.get("/api/v1/users/", params={"query": email}).json()["users"] == []


def test_a_request_without_the_session_token_is_refused(open_instance):
    anonymous = httpx.Client(base_url=open_instance.base_url, timeout=60.0)
    with anonymous:
        answers = [anonymous.get(path).status_code for path in ("/api/v1/auths/", "/api/v1/chats/")]
    assert answers == [401, 401]


def test_an_instance_that_already_has_accounts_refuses_the_sign_in(instance_with):
    reused = instance_with({**NO_AUTH, "REGRESSION_THROWAWAY_INSTANCE": "no-auth-with-accounts"})
    with reused.client() as client:
        owner_id = client.get("/api/v1/auths/").json()["id"]
        # the account now looks like one made on an installation with sign-in on
        renamed = client.post(
            f"/api/v1/users/{owner_id}/update", json={"email": "owner@example.com"}
        )
    assert renamed.status_code == 200, renamed.text

    refused = sign_in(reused)

    assert refused.status_code == 400, refused.text
    assert "existing users" in refused.json()["detail"]
    with reused.client() as client:
        emails = [user["email"] for user in client.get("/api/v1/users/").json()["users"]]
    assert emails == ["owner@example.com"], (
        "the sign-in created an account next to the existing one"
    )
