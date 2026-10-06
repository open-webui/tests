"""Journey: the admin account rules in Admin Settings > Authentication, and the auth page errors.

With API Keys on, API Key Endpoint Restrictions limits what an account's key may call: the
endpoints the admin lists answer and every other one is refused, and with the restriction off
both answer. A Default Group chosen on the page receives each account that then signs up through
the auth page, which shows in the group's member count and its Users list, while with None chosen
the newcomer joins no group. On the auth page a wrong password shows the sign-in error and leaves
the sign-in form where it is. On an instance that asks for a password confirmation, a sign-up
whose two passwords differ says so and creates no account, while matching passwords sign in.
With Login Form switched off, the auth page offers no email and password form and a sign-up over
the API is refused, while signing in with a password over the API still works.

Discriminates: passes on dev 30f3f6a8f; in a frontend copy, the page saving the endpoint list as
an empty string turns the restriction test red (the listed endpoint is refused too), the page
saving no group turns the Default Group test red (the group stays empty), the sign-up handler
skipping its password comparison turns the mismatch test red (the account is created), and the
sign-in handler swallowing the error turns the wrong password test red. In a frontend build of dev
30f3f6a8f whose Authentication form saves the stored admin settings, the Login Form test goes red.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import httpx
import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from harness.actors import Actor
from harness.instance import LaunchedInstance
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PERMISSIONS = "/api/v1/users/default/permissions"
INVALID_CREDENTIALS = "The email or password provided is incorrect."
PAGE_TIMEOUT_MS = 30_000
PASSWORD = "signup-password-1"
REFUSED_PATH = "/api/v1/chats/"


def _change_admin_config(admin: Actor, **changes) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def _allow_api_keys_by_default(admin: Actor) -> None:
    with admin.client() as client:
        current = client.get(PERMISSIONS)
        current.raise_for_status()
        permissions = current.json()
        permissions["features"]["api_keys"] = True
        saved = client.post(PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


def _authentication_settings(page: Page) -> Locator:
    page.goto("/admin/settings/authentication")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("JWT Expiration", exact=True)).to_be_visible()
    return settings


def _save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def _status(path: str, key: str, base_url: str) -> int:
    return httpx.get(f"{base_url}{path}", headers={"Authorization": f"Bearer {key}"}).status_code


def _new_api_key(account: Actor) -> str:
    with account.client() as client:
        created = client.post("/api/v1/auths/api_key")
    assert created.status_code == 200, created.text
    return created.json()["api_key"]


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


@pytest.fixture
def keyed_account(admin, make_user, preserve) -> tuple[Actor, str]:
    """An account with an API key, while keys are on and allowed by default."""
    preserve("admin_config", "permissions")
    _change_admin_config(admin, ENABLE_API_KEYS=True, ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=False)
    _allow_api_keys_by_default(admin)
    account = make_user()
    return account, _new_api_key(account)


@pytest.fixture
def default_group(admin, preserve) -> Iterator[tuple[str, str]]:
    """A group (name, id) for newcomers, with sign-up open and the default role `user`."""
    preserve("admin_config")
    _change_admin_config(admin, ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="user", DEFAULT_GROUP_ID="")
    name = f"Newcomers {uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
    yield name, group_id
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _sign_up(page: Page, name: str, email: str, confirmation: str | None = None) -> None:
    page.goto("/auth")
    page.get_by_role("button", name="Sign up").click()
    expect(page.get_by_text("Sign up to")).to_be_visible()
    page.get_by_label("Name").fill(name)
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(PASSWORD)
    if confirmation is not None:
        page.get_by_label("Confirm Password").fill(confirmation)
    page.get_by_role("button", name="Create Account").click()


def _new_visitor() -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return f"Visitor {suffix}", f"visitor-{suffix}@example.com"


def _group_row(page: Page, name: str, members: int) -> Locator:
    page.goto("/admin/users/groups")
    groups = page.get_by_role("main")
    groups.get_by_role("textbox", name="Search Groups").fill(name)
    return groups.get_by_role("button", name=re.compile(rf"^{name} {members} direct members"))


def test_the_endpoint_restriction_lets_the_listed_endpoint_answer_and_refuses_the_others(
    admin_page, keyed_account
):
    account, key = keyed_account
    base_url = account.base_url
    settings = _authentication_settings(admin_page)
    settings.get_by_role("switch", name="API Key Endpoint Restrictions").click()
    settings.get_by_placeholder("e.g.) /api/v1/messages, /api/v1/channels").fill("/api/models")

    _save(admin_page, settings)

    assert _status("/api/models", key, base_url) == 200
    assert _status(REFUSED_PATH, key, base_url) == 403


def test_the_session_of_the_same_account_still_reaches_the_refused_endpoint(
    admin_page, keyed_account
):
    account, key = keyed_account
    settings = _authentication_settings(admin_page)
    settings.get_by_role("switch", name="API Key Endpoint Restrictions").click()
    settings.get_by_placeholder("e.g.) /api/v1/messages, /api/v1/channels").fill("/api/models")
    _save(admin_page, settings)
    assert _status(REFUSED_PATH, key, account.base_url) == 403

    assert _status(REFUSED_PATH, account.token, account.base_url) == 200


def test_without_the_restriction_the_key_reaches_every_endpoint(keyed_account):
    account, key = keyed_account

    assert _status("/api/models", key, account.base_url) == 200
    assert _status(REFUSED_PATH, key, account.base_url) == 200


def test_an_account_signing_up_lands_in_the_default_group(admin_page, page, default_group):
    group_name, _ = default_group
    settings = _authentication_settings(admin_page)
    settings.get_by_role("combobox", name="Select a group").select_option(label=group_name)
    _save(admin_page, settings)
    name, email = _new_visitor()

    _sign_up(page, name, email)
    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    expect(_group_row(admin_page, group_name, members=1)).to_be_visible()
    _group_row(admin_page, group_name, members=1).click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name="Users", exact=True).click()
    editing.get_by_role("textbox", name="Search").fill(name)
    expect(editing.get_by_role("checkbox", name=name)).to_have_attribute("aria-checked", "true")


def test_with_no_default_group_a_newcomer_joins_no_group(admin_page, page, default_group):
    group_name, _ = default_group
    name, email = _new_visitor()

    _sign_up(page, name, email)
    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    expect(_group_row(admin_page, group_name, members=0)).to_be_visible()


def test_a_wrong_password_shows_the_error_and_stays_on_the_sign_in_form(page, make_user):
    account = make_user()
    page.goto("/auth")
    page.get_by_label("Email").fill(account.email)
    page.get_by_label("Password", exact=True).fill("not-the-password")

    page.get_by_role("button", name="Sign in", exact=True).click()

    expect(page.get_by_text(INVALID_CREDENTIALS)).to_be_visible()
    expect(page.get_by_text("Sign in to")).to_be_visible()
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    expect(chat_input(page)).to_be_hidden()


@pytest.fixture
def confirming(instance_with) -> LaunchedInstance:
    launched = instance_with({"ENABLE_SIGNUP_PASSWORD_CONFIRMATION": "true"})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with launched.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        opened = client.post(
            ADMIN_CONFIG,
            json={**current.json(), "ENABLE_SIGNUP": True, "DEFAULT_USER_ROLE": "user"},
        )
    assert opened.status_code == 200, opened.text
    return launched


@pytest.fixture
def visitor_page(browser: Browser, confirming: LaunchedInstance) -> Iterator[Page]:
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=confirming.base_url
    )
    yield context.new_page()
    context.close()


def _accounts_named(instance: LaunchedInstance, email: str) -> list[dict]:
    with instance.client() as client:
        found = client.get("/api/v1/users/", params={"query": email})
    found.raise_for_status()
    return found.json()["users"]


@pytest.mark.slow
def test_a_sign_up_with_two_different_passwords_says_so_and_creates_nothing(
    visitor_page, confirming
):
    name, email = _new_visitor()

    _sign_up(visitor_page, name, email, confirmation="another-password-2")

    expect(visitor_page.get_by_text("Passwords do not match.")).to_be_visible()
    expect(visitor_page.get_by_role("button", name="Create Account")).to_be_visible()
    expect(chat_input(visitor_page)).to_be_hidden()
    assert _accounts_named(confirming, email) == []


@pytest.mark.slow
def test_a_sign_up_with_matching_passwords_creates_the_account(visitor_page, confirming):
    name, email = _new_visitor()

    _sign_up(visitor_page, name, email, confirmation=PASSWORD)

    expect(chat_input(visitor_page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    assert [found["email"] for found in _accounts_named(confirming, email)] == [email]


def test_with_the_login_form_off_the_auth_page_offers_no_password_form(
    admin_page, page, admin, make_user, preserve
):
    preserve("admin_config")
    _change_admin_config(admin, ENABLE_SIGNUP=True, DEFAULT_USER_ROLE="user")
    account = make_user()
    settings = _authentication_settings(admin_page)
    login_form = settings.get_by_role("switch", name="Login Form")
    expect(login_form).to_be_checked()
    login_form.click()
    _save(admin_page, settings)

    page.goto("/auth")

    expect(page.get_by_label("Email")).to_have_count(0)
    expect(page.get_by_label("Password", exact=True)).to_have_count(0)
    name, email = _new_visitor()
    refused = httpx.post(
        f"{account.base_url}/api/v1/auths/signup",
        json={"name": name, "email": email, "password": PASSWORD},
    )
    assert refused.status_code >= 400, refused.text
    signed_in = httpx.post(
        f"{account.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": account.password},
    )
    assert signed_in.status_code == 200, signed_in.text
