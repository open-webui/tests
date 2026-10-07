"""Journey: the password rules set in the environment show as the hint in the browser.

On an instance that checks new passwords against a pattern (at least 12 characters with a digit)
and gives a hint, the hint appears as an error message and nothing changes in four places. A
visitor signing up on the auth page stays on the form and no account appears. An account changing
its password in Settings > Account stays signed in and its old password still signs in. An admin
adding a user in Admin Panel > Users keeps the dialog open and no account appears. The admin's
Edit User with a new password leaves the account's old password working. A password that meets
the rule on the sign-up page lands the visitor in the chat.

Discriminates: passes on dev ebc6add67; in a backend copy where the password check skips the
pattern (the length limit stays), the four refusal tests go red (the visitor lands in the chat,
the old password stops working or the account exists) and the control stays green.
"""

from __future__ import annotations

import json
import uuid
from typing import Callable, Iterator

import httpx
import pytest
from playwright.sync_api import Browser, BrowserContext, Page, expect

from harness.actors import Actor, create_user
from harness.instance import LaunchedInstance
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
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
PAGE_TIMEOUT_MS = 30_000
BROKEN = "short-1"
FITTING = "fitting-password-42"


@pytest.fixture
def ruled(instance_with) -> LaunchedInstance:
    launched = instance_with(RULES)
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
def open_page(browser: Browser, ruled: LaunchedInstance) -> Iterator[Callable[..., Page]]:
    """`open_page()` opens a signed-out page, `open_page(account)` a signed-in one."""
    contexts: list[BrowserContext] = []

    def opening(account: Actor | None = None) -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=ruled.base_url
        )
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        contexts.append(context)
        page = context.new_page()
        if account is not None:
            with account.client() as client:
                # an admin meets the changelog on the first page load
                client.post(
                    "/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}}
                )
            token = json.dumps(account.token)
            page.add_init_script(
                f"try {{ localStorage.setItem('token', {token}); }} catch (e) {{}}"
            )
            page.goto("/")
        return page

    yield opening
    for context in contexts:
        context.close()


def _new_visitor() -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return f"Visitor {suffix}", f"visitor-{suffix}@example.com"


def _accounts_named(instance: LaunchedInstance, email: str) -> list[dict]:
    with instance.client() as client:
        found = client.get("/api/v1/users/", params={"query": email})
    found.raise_for_status()
    return found.json()["users"]


def _can_sign_in(instance: LaunchedInstance, account: Actor, password: str) -> bool:
    signed_in = httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": account.email, "password": password},
        timeout=60.0,
    )
    return signed_in.status_code == 200


def _sign_up(page: Page, name: str, email: str, password: str) -> None:
    page.goto("/auth")
    page.get_by_role("button", name="Sign up").click()
    expect(page.get_by_text("Sign up to")).to_be_visible()
    page.get_by_label("Name").fill(name)
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Create Account").click()


def _change_password_in_settings(page: Page, current: str, new: str) -> None:
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
    page.get_by_role("tab", name="Account").click()
    form = page.locator("form").filter(has_text="Change Password")
    form.get_by_role("button", name="Show").click()
    form.get_by_placeholder("Enter your current password").fill(current)
    form.get_by_placeholder("Enter your new password").fill(new)
    form.get_by_placeholder("Confirm your new password").fill(new)
    form.get_by_role("button", name="Update password").click()


def _user_list(page: Page):
    page.goto("/admin/users")
    users = page.get_by_role("main")
    expect(users.get_by_role("button", name="Add User")).to_be_visible()
    return users


def test_a_sign_up_with_a_password_that_breaks_the_rule_shows_the_hint_and_creates_no_account(
    open_page, ruled
):
    page = open_page()
    name, email = _new_visitor()

    _sign_up(page, name, email, BROKEN)

    expect(page.get_by_text(HINT)).to_be_visible()
    expect(page.get_by_role("button", name="Create Account")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()
    assert _accounts_named(ruled, email) == []


def test_a_sign_up_with_a_password_that_meets_the_rule_lands_in_the_chat(open_page, ruled):
    page = open_page()
    name, email = _new_visitor()

    _sign_up(page, name, email, FITTING)

    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    assert [found["email"] for found in _accounts_named(ruled, email)] == [email]


def test_changing_your_password_to_one_that_breaks_the_rule_shows_the_hint_and_keeps_the_old_one(
    open_page, ruled
):
    account = create_user(ruled)
    page = open_page(account)
    expect(chat_input(page)).to_be_visible()

    _change_password_in_settings(page, account.password, BROKEN)

    expect(page.get_by_text(HINT)).to_be_visible()
    expect(chat_input(page)).to_be_visible()
    assert _can_sign_in(ruled, account, account.password)
    assert not _can_sign_in(ruled, account, BROKEN)


def test_adding_a_user_with_a_password_that_breaks_the_rule_shows_the_hint_and_keeps_the_dialog(
    open_page, ruled
):
    page = open_page(create_user(ruled, role="admin"))
    users = _user_list(page)
    name, email = _new_visitor()

    users.get_by_role("button", name="Add User").click()
    adding = page.get_by_role("dialog").filter(has_text="Add User")
    adding.get_by_role("textbox", name="Name").fill(name)
    adding.get_by_role("textbox", name="Email").fill(email)
    adding.get_by_placeholder("Enter Your Password").fill(BROKEN)
    adding.get_by_role("button", name="Save").click()

    expect(page.get_by_text(HINT)).to_be_visible()
    expect(adding).to_be_visible()
    assert _accounts_named(ruled, email) == []


def test_editing_a_user_with_a_new_password_that_breaks_the_rule_shows_the_hint_and_keeps_the_old(
    open_page, ruled
):
    account = create_user(ruled)
    page = open_page(create_user(ruled, role="admin"))
    users = _user_list(page)

    users.get_by_role("textbox", name="Search").fill(account.email)
    users.get_by_role("row").filter(has_text=account.email).get_by_role(
        "button", name="Edit User"
    ).click()
    editing = page.get_by_role("dialog").filter(has_text="Edit User")
    editing.get_by_label("New Password").fill(BROKEN)
    editing.get_by_role("button", name="Save").click()

    expect(page.get_by_text(HINT)).to_be_visible()
    expect(editing).to_be_visible()
    assert _can_sign_in(ruled, account, account.password)
    assert not _can_sign_in(ruled, account, BROKEN)
