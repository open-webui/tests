"""Journey: the admin's JWT Expiration decides how long a signed-in session lasts.

A sign-in answers the moment its session ends (`expires_at`), the admin's duration after the
sign-in, and once that moment passes the session is refused. With `-1` a session never ends: the
answer carries no `expires_at` and the token works. A duration the server cannot read is not
saved, so the one before it stays. A session issued before the admin shortens the duration keeps
the end it was issued with.

Discriminates: on dev ebc6add67, a backend copy that signs session tokens without an expiry turns
both expiry tests red (the session still answers after its end), one whose duration parser
reads `-1` as a number turns the `-1` test red and one whose settings save skips the duration
check turns the unreadable test red.
"""

from __future__ import annotations

import time

import pytest

from harness.actors import sign_in

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def save_session_length(admin, duration: str) -> dict:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "JWT_EXPIRES_IN": duration})
    assert saved.status_code == 200, saved.text
    return saved.json()


def signed_in(instance, account) -> dict:
    with instance.client() as client:
        answer = client.post(
            "/api/v1/auths/signin", json={"email": account.email, "password": account.password}
        )
    assert answer.status_code == 200, answer.text
    return answer.json()


def session_status(instance, token: str) -> int:
    with instance.client(token) as client:
        return client.get("/api/v1/auths/").status_code


def wait_for_refusal(instance, token: str, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if session_status(instance, token) == 401:
            return True
        time.sleep(0.25)
    return False


@pytest.fixture
def session_length(admin, preserve):
    preserve("admin_config")
    return lambda duration: save_session_length(admin, duration)


def test_a_session_ends_at_the_moment_the_sign_in_announced(instance, make_user, session_length):
    account = make_user()
    session_length("3s")

    signed_in_at = time.time()
    session = signed_in(instance, account)

    assert 1 <= session["expires_at"] - signed_in_at <= 5
    assert session_status(instance, session["token"]) == 200
    assert wait_for_refusal(instance, session["token"], within=10), (
        "the session still answers after its announced end"
    )
    assert time.time() >= session["expires_at"]


def test_minus_one_issues_sessions_that_never_end(instance, make_user, session_length):
    account = make_user()
    session_length("-1")

    session = signed_in(instance, account)

    assert session["expires_at"] is None
    assert session_status(instance, session["token"]) == 200


def test_an_unreadable_duration_is_not_saved(admin, session_length):
    session_length("2h")

    saved = session_length("soon")

    assert saved["JWT_EXPIRES_IN"] == "2h"
    with admin.client() as client:
        assert client.get(ADMIN_CONFIG).json()["JWT_EXPIRES_IN"] == "2h"


def test_a_session_keeps_the_end_it_was_issued_with(instance, make_user, session_length):
    account = make_user()
    session_length("1h")
    session = signed_in(instance, account)

    session_length("2s")
    later = sign_in(instance, account.email, account.password)

    assert wait_for_refusal(instance, later, within=10)
    assert session_status(instance, session["token"]) == 200
