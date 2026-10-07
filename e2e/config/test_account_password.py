"""Journey: a person changes their own password under Settings > Account > Change Password.

A change with the right current password and a matching confirmation sends the browser to the
sign-in page, where the new password signs in and the old one is refused with the sign-in error.
A wrong current password is refused with its own message, a confirmation that does not match is
caught before anything is sent, and in both cases the browser stays signed in and the old
password keeps working. That a change signs out the account's other browsers is pinned in
e2e/security/test_password_change_revokes_sessions.py.

Discriminates: on dev ebc6add67, a frontend copy whose password form skips comparing the two new
passwords turns the mismatch test red (the password changes) and one whose form drops the
server's error turns the wrong-password test red; a backend copy whose password route stores
nothing turns the change test red (the old password still signs in).
"""

from __future__ import annotations

import re

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NEW_PASSWORD = "harbour-lights-42"
INVALID_CREDENTIALS = "The email or password provided is incorrect."


def _password_form(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="Account").click()
    form = page.locator("form").filter(has_text="Change Password")
    form.get_by_role("button", name="Show").click()
    return form


def _change_password(form: Locator, current: str, new: str, confirmation: str) -> None:
    form.get_by_placeholder("Enter your current password").fill(current)
    form.get_by_placeholder("Enter your new password").fill(new)
    form.get_by_placeholder("Confirm your new password").fill(confirmation)
    form.get_by_role("button", name="Update password").click()


def _sign_in(page: Page, email: str, password: str) -> None:
    page.goto("/auth")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def _signs_in(account: Actor, password: str) -> bool:
    answer = httpx.post(
        f"{account.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": password},
    )
    return answer.status_code == 200


def test_a_changed_password_signs_in_and_the_old_one_is_refused(page_for, make_user, page):
    account = make_user()
    acting = page_for(account)

    _change_password(_password_form(acting), account.password, NEW_PASSWORD, NEW_PASSWORD)

    expect(acting).to_have_url(re.compile(r"/auth"))
    expect(acting.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    _sign_in(page, account.email, account.password)
    expect(page.get_by_text(INVALID_CREDENTIALS)).to_be_visible()
    _sign_in(page, account.email, NEW_PASSWORD)
    expect(chat_input(page)).to_be_visible()


def test_a_wrong_current_password_is_refused_and_changes_nothing(page_for, make_user):
    account = make_user()
    page = page_for(account)

    _change_password(_password_form(page), "not-my-password", NEW_PASSWORD, NEW_PASSWORD)

    expect(page.get_by_text("The password provided is incorrect.")).to_be_visible()
    page.keyboard.press("Escape")
    expect(chat_input(page)).to_be_visible()
    assert _signs_in(account, account.password)
    assert not _signs_in(account, NEW_PASSWORD)


def test_a_confirmation_that_does_not_match_is_caught_and_changes_nothing(page_for, make_user):
    account = make_user()
    page = page_for(account)
    password_changes: list[str] = []
    page.on(
        "request",
        lambda request: "/update/password" in request.url and password_changes.append(request.url),
    )

    _change_password(_password_form(page), account.password, NEW_PASSWORD, "harbour-lights-43")

    expect(page.get_by_text("The passwords you entered don't quite match.")).to_be_visible()
    assert password_changes == []
    assert _signs_in(account, account.password)
    assert not _signs_in(account, NEW_PASSWORD)
