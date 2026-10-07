"""Journey: a visitor signs up on the auth page, and an admin lets a pending account in.

With sign-up on and the default role `user`, the sign-up form lands the visitor in the chat; with
the default role `admin` the newcomer's user menu offers the Admin Panel. An email already
registered (in any letter case) or one the server cannot read as an address is refused on the form
with the server's own message, and no account appears. With the default role `pending` the form
lands on the activation screen, which shows the admin's own title and text when they set one, and
once the admin changes the role in Admin > Users the account gets in with Check Again. A pending
visitor can also sign out from that screen: signing in again before the admin acts shows the
screen again, and after it the same credentials go straight to the chat. With sign-up switched
off the auth page offers no sign-up form and the API refuses one.

Discriminates: passes on dev 176d31d1d; in a frontend copy, showing the auth page's sign-up switch
whatever the sign-up setting turns the sign-up-off test red, the pending overlay ignoring the
configured title turns the custom-text test red, Check Again doing nothing turns the activation
test red, and the sign-up handler sending the name as the email turns all four sign-up tests red.
On dev ebc6add67, a frontend copy whose sign-up handler drops the server's error turns the
duplicate and format tests red, and one whose pending screen's Sign Out does nothing turns the
signed-out pending test red; a backend copy whose admin settings refuse the default role `admin`
turns the admin role test red (the newcomer gets the activation screen).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PAGE_TIMEOUT_MS = 30_000
PASSWORD = "signup-password-1"


@pytest.fixture
def signup_settings(admin, preserve):
    """`signup_settings(**changes)` saves admin settings, restored after the test."""
    preserve("admin_config")

    def save(**changes) -> None:
        with admin.client() as client:
            current = client.get(ADMIN_CONFIG)
            current.raise_for_status()
            saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
        assert saved.status_code == 200, saved.text

    return save


def _sign_up(page: Page, name: str, email: str) -> None:
    page.goto("/auth")
    page.get_by_role("button", name="Sign up").click()
    expect(page.get_by_text("Sign up to")).to_be_visible()
    page.get_by_label("Name").fill(name)
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(PASSWORD)
    page.get_by_role("button", name="Create Account").click()


def _new_visitor() -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return f"Visitor {suffix}", f"visitor-{suffix}@example.com"


def _sign_in(page: Page, email: str) -> None:
    page.goto("/auth")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(PASSWORD)
    page.get_by_role("button", name="Sign in", exact=True).click()


def _accounts_with(admin: Actor, email: str) -> list[str]:
    with admin.client() as client:
        found = client.get("/api/v1/users/", params={"query": email})
    found.raise_for_status()
    return [account["email"] for account in found.json()["users"]]


def test_a_visitor_signs_up_and_lands_in_the_chat_with_the_default_role(page, signup_settings):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="user")
    name, email = _new_visitor()

    _sign_up(page, name, email)

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(page.get_by_text("Account Activation Pending")).to_be_hidden()


def test_a_visitor_signing_up_with_the_pending_default_gets_the_activation_screen(
    page, signup_settings
):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="pending")
    name, email = _new_visitor()

    _sign_up(page, name, email)

    expect(page.get_by_text("Account Activation Pending")).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(page.get_by_role("button", name="Check Again")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()


def test_the_pending_screen_shows_the_title_and_text_the_admin_configured(page, signup_settings):
    title = f"Hold on {uuid.uuid4().hex[:6]}"
    signup_settings(
        ENABLE_SIGNUP=True,
        DEFAULT_USER_ROLE="pending",
        PENDING_USER_OVERLAY_TITLE=title,
        PENDING_USER_OVERLAY_CONTENT="Ask the front desk for a badge.",
    )
    name, email = _new_visitor()

    _sign_up(page, name, email)

    expect(page.get_by_text(title)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(page.get_by_text("Ask the front desk for a badge.")).to_be_visible()
    expect(page.get_by_text("Account Activation Pending")).to_be_hidden()


def test_a_pending_visitor_gets_in_once_an_admin_activates_the_account(
    page, page_for, make_user, signup_settings
):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="pending")
    name, email = _new_visitor()
    _sign_up(page, name, email)
    expect(page.get_by_text("Account Activation Pending")).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/users")
    users = admin_page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(email)
    row = users.get_by_role("row").filter(has_text=email)
    expect(row.get_by_role("button", name="Change User Role")).to_have_text("pending")
    row.get_by_role("button", name="Change User Role").click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User")
    editing.get_by_role("combobox", name="Role").select_option(label="User")
    editing.get_by_role("button", name="Save").click()
    expect(row.get_by_role("button", name="Change User Role")).to_have_text("user")

    page.get_by_role("button", name="Check Again").click()

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(page.get_by_text("Account Activation Pending")).to_be_hidden()


def test_with_sign_up_off_the_auth_page_offers_no_form_and_the_api_refuses_one(
    page, signup_settings
):
    signup_settings(ENABLE_SIGNUP=False)
    name, email = _new_visitor()

    page.goto("/auth")

    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Sign up")).to_have_count(0)
    refused = page.request.post(
        "/api/v1/auths/signup", data={"name": name, "email": email, "password": PASSWORD}
    )
    assert refused.status == 403, refused.text()


def test_an_email_already_registered_is_refused_on_the_form(
    page, admin, make_user, signup_settings
):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="user")
    existing = make_user()

    _sign_up(page, "Second Comer", existing.email.upper())

    expect(page.get_by_text("This email is already registered")).to_be_visible()
    expect(page.get_by_role("button", name="Create Account")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()
    assert _accounts_with(admin, existing.email) == [existing.email]


def test_an_email_the_server_cannot_read_is_refused_with_the_format_message(
    page, admin, signup_settings
):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="user")
    name, email = _new_visitor()
    # the browser accepts an address without a dot in the domain, the server does not
    undotted = email.replace("@example.com", "@harbour")

    _sign_up(page, name, undotted)

    expect(page.get_by_text("The email format you entered is invalid.")).to_be_visible()
    expect(page.get_by_role("button", name="Create Account")).to_be_visible()
    assert _accounts_with(admin, undotted) == []


def test_with_the_admin_default_role_a_newcomer_is_offered_the_admin_panel(page, signup_settings):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="admin")
    name, email = _new_visitor()

    _sign_up(page, name, email)
    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    # the changelog modal covers an admin's first page load
    page.request.post(
        "/api/v1/users/user/settings/update",
        data={"ui": {"showChangelog": False}},
        headers={"Authorization": f"Bearer {page.evaluate('localStorage.token')}"},
    )
    page.reload()

    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("link", name="Admin Panel").click()
    expect(page.get_by_role("main").get_by_role("button", name="Add User")).to_be_visible()
    expect(page.get_by_text("Account Activation Pending")).to_be_hidden()


def test_a_pending_visitor_who_signed_out_gets_in_with_the_same_credentials_once_activated(
    page, admin, signup_settings
):
    signup_settings(ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="pending")
    name, email = _new_visitor()
    _sign_up(page, name, email)
    pending_screen = page.get_by_text("Account Activation Pending")
    expect(pending_screen).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    page.get_by_role("button", name="Sign Out").click()
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    _sign_in(page, email)
    expect(pending_screen).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    page.get_by_role("button", name="Sign Out").click()
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()

    with admin.client() as client:
        account_id = client.get("/api/v1/users/", params={"query": email}).json()["users"][0]["id"]
        activated = client.post(f"/api/v1/users/{account_id}/update", json={"role": "user"})
    assert activated.status_code == 200, activated.text
    _sign_in(page, email)

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(pending_screen).to_be_hidden()
