"""Journey: repeated sign-in attempts for one email are cut off, and only for that email.

The sign-in form allows 15 attempts per email within three minutes and answers 429 to the next
one, even with the right password, so guessing a password takes a fresh email every 15 tries.
The count is kept per email, so someone else signing in at the same time is not held up. The
test spends exactly one attempt past the limit on accounts of its own, so it never waits on the
window.

Discriminates: fails with the `signin_rate_limiter.is_limited` check removed from the signin
route (the 16th attempt with the right password signs in).
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ALLOWED_ATTEMPTS = 15


def sign_in(instance, email: str, password: str) -> httpx.Response:
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )


def test_repeated_wrong_passwords_block_that_email_and_no_other(instance, make_user):
    target, bystander = make_user(), make_user()

    wrong_answers = [
        sign_in(instance, target.email, f"guess-{attempt}").status_code
        for attempt in range(ALLOWED_ATTEMPTS)
    ]
    blocked = sign_in(instance, target.email, target.password)

    assert wrong_answers == [400] * ALLOWED_ATTEMPTS
    assert blocked.status_code == 429, "the right password still signed in after 15 wrong ones"
    assert sign_in(instance, bystander.email, bystander.password).status_code == 200


def test_the_count_ignores_the_case_of_the_email(instance, make_user):
    target = make_user()

    for attempt in range(ALLOWED_ATTEMPTS):
        email = target.email.upper() if attempt % 2 else target.email
        assert sign_in(instance, email, f"guess-{attempt}").status_code == 400

    assert sign_in(instance, target.email.upper(), target.password).status_code == 429


def test_a_few_wrong_passwords_still_let_the_right_one_in(instance, make_user):
    account = make_user()

    for attempt in range(3):
        assert sign_in(instance, account.email, f"guess-{attempt}").status_code == 400

    assert sign_in(instance, account.email, account.password).status_code == 200
