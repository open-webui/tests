"""Journey: signing out in one browser ends that browser's session and leaves the others.

An account signs in through the auth page in two browsers on an instance with Redis, which is
where signed-out sessions are remembered. Sign Out in the user menu of the first lands on the
sign-in page, a reload keeps it there, and the session that browser held is refused from then
on. The second browser is still in the chat after a reload. Signing out everywhere at once is the
admin's "Sign out all devices" (e2e/admin/test_user_list.py) or a password change
(e2e/security/test_password_change_revokes_sessions.py).

Discriminates: on dev ebc6add67, a backend copy whose sign-out route skips revoking the token
turns the test red (the signed-out session still answers), and one whose sign-out revokes every
session of the account turns it red as well (the second browser is sent to sign in).
"""

from __future__ import annotations

import re
from typing import Callable, Generator

import httpx
import pytest
from playwright.sync_api import Browser, BrowserContext, Page, expect

from harness.actors import Actor, create_user
from harness.instance import LaunchedInstance
from integration.stateful_redis import StatefulRedis
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TIMEOUT_MS = 30_000
SIGN_IN_URL = re.compile(r"/auth")


@pytest.fixture(scope="module")
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
def new_browser(
    browser: Browser, redis_instance: LaunchedInstance
) -> Generator[Callable[[], Page], None, None]:
    contexts: list[BrowserContext] = []

    def open_page() -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=redis_instance.base_url
        )
        context.set_default_timeout(PAGE_TIMEOUT_MS)
        contexts.append(context)
        return context.new_page()

    yield open_page
    for context in contexts:
        context.close()


def _sign_in(page: Page, account: Actor) -> str:
    page.goto("/auth")
    page.get_by_label("Email").fill(account.email)
    page.get_by_label("Password", exact=True).fill(account.password)
    page.get_by_role("button", name="Sign in", exact=True).click()
    expect(chat_input(page)).to_be_visible()
    return page.evaluate("localStorage.token")


def _session_status(instance: LaunchedInstance, token: str) -> int:
    return httpx.get(
        f"{instance.base_url}/api/v1/auths/", headers={"Authorization": f"Bearer {token}"}
    ).status_code


def test_signing_out_in_one_browser_ends_its_session_and_leaves_the_other(
    redis_instance, new_browser
):
    account = create_user(redis_instance)
    leaving, staying = new_browser(), new_browser()
    leaving_session = _sign_in(leaving, account)
    staying_session = _sign_in(staying, account)

    leaving.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    leaving.get_by_role("button", name="Sign Out").click()

    expect(leaving).to_have_url(SIGN_IN_URL)
    expect(leaving.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    leaving.goto("/")
    expect(leaving).to_have_url(SIGN_IN_URL)
    assert _session_status(redis_instance, leaving_session) == 401, (
        "the signed-out session still answers"
    )
    staying.reload()
    expect(chat_input(staying)).to_be_visible()
    assert _session_status(redis_instance, staying_session) == 200
