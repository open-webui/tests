"""Regression: changing a password left the account's other browsers signed in.

open-webui 0.11.1 fix `21e390561` (#28725): a password change revoked nothing, so every browser
signed in before it kept working until its token expired, four weeks by default. The server now
revokes every earlier session (this needs Redis) and the Settings page signs the acting browser
out and sends it to the sign-in page.

Two browsers share one account on an instance of its own backed by `StatefulRedis`; the first
changes the password under Settings > Account.

Twin of unit/security/test_password_change_revokes_sessions.py.

Discriminates: passes on dev bbfa876af, fails with both `revoke_user_tokens` calls removed
from the password routes (the second browser is still signed in after a reload). The acting
browser reaches the sign-in page even with the page's own sign-out removed, because its revoked
session is refused anyway, so that check is nearby only.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Generator

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, expect

from harness.actors import create_user, sign_in
from harness.instance import LaunchedInstance
from integration.stateful_redis import StatefulRedis
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NEW_PASSWORD = "N3w-password-after-the-change"
SIGN_IN_URL = re.compile(r"/auth")
PAGE_TIMEOUT_MS = 30_000


@pytest.fixture(scope="session")
def revocation_store() -> Generator[StatefulRedis, None, None]:
    store = StatefulRedis()
    yield store
    store.close()


@pytest.fixture
def redis_instance(revocation_store, instance_with) -> LaunchedInstance:
    launched = instance_with({"REDIS_URL": revocation_store.url})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


@pytest.fixture
def open_signed_in(
    browser: Browser, redis_instance: LaunchedInstance
) -> Generator[Callable[[str], Page], None, None]:
    """`open_signed_in(token)` opens a page carrying that session in a browser of its own."""
    contexts: list[BrowserContext] = []

    def open_page(token: str) -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=redis_instance.base_url
        )
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        contexts.append(context)
        page = context.new_page()
        page.add_init_script(
            f"try {{ localStorage.setItem('token', {json.dumps(token)}); }} catch (e) {{}}"
        )
        page.goto("/")
        return page

    yield open_page
    for context in contexts:
        context.close()


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


def test_changing_the_password_signs_out_the_other_browser(redis_instance, open_signed_in):
    account = create_user(redis_instance)
    with account.client() as client:
        client.post("/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}})
    acting = open_signed_in(account.token)
    other = open_signed_in(sign_in(redis_instance, account.email, account.password))
    expect(chat_input(other)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    _change_password_in_settings(acting, account.password, NEW_PASSWORD)
    expect(acting).to_have_url(SIGN_IN_URL, timeout=PAGE_TIMEOUT_MS)

    other.reload()
    expect(other).to_have_url(SIGN_IN_URL, timeout=PAGE_TIMEOUT_MS)
    expect(chat_input(other)).to_be_hidden()
