"""Journey: an ended session sends the person to the sign-in page and back to where they were.

A page opened without a session goes to the sign-in page with the page it wanted in `redirect`,
and signing in there lands on that page. The admin's JWT Expiration decides how long a session
lasts: while a page is open the app watches the session's expiry, and once it is close says
"Session expired. Please sign in again." and sends the person to sign in, keeping the page; a page
opened later with a session that has already expired does the same without the notice. In each
case signing in again lands on the chat the person had open. The chats are the account's own, so
the page they land on shows its messages.

Discriminates: on dev ebc6add67, a frontend copy whose app layout sends a signed-out visitor to
the sign-in page without the page they wanted turns all three tests red (they land on a new chat),
and one whose expiry watch never ends the session turns the open-page test red.
"""

from __future__ import annotations

import re
import time
import uuid
from urllib.parse import quote

import httpx
import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import Actor, sign_in
from harness.chat import ask
from harness.instance import LaunchedInstance

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
EXPIRED_NOTICE = "Session expired. Please sign in again."
PAGE_TIMEOUT_MS = 30_000
# the app ends a session a minute before it expires and checks every 15 seconds
EXPIRY_WATCH_MS = 40_000


class SessionLength:
    """The admin's JWT Expiration: `set("1m")` changes it, `restore()` puts the original back."""

    def __init__(self, admin: Actor) -> None:
        self.admin = admin
        self.original = self._config()["JWT_EXPIRES_IN"]

    def _config(self) -> dict:
        with self.admin.client() as client:
            current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        return current.json()

    def set(self, duration: str) -> None:
        with self.admin.client() as client:
            saved = client.post(ADMIN_CONFIG, json={**self._config(), "JWT_EXPIRES_IN": duration})
        assert saved.status_code == 200, saved.text

    def restore(self) -> None:
        self.set(self.original)


@pytest.fixture
def session_length(admin, preserve) -> SessionLength:
    preserve("admin_config")
    return SessionLength(admin)


@pytest.fixture
def chat_owner(make_user, upstream) -> tuple[Actor, str, str]:
    """An account with one chat of its own: (account, chat path, the reply in that chat)."""
    account = make_user()
    answer = f"Tide tables for berth {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer))
    with account.client() as client:
        turn, _ = ask(client, "When is high tide?")
    return account, f"/c/{turn.chat_id}", answer


def _sign_in_page_for(path: str) -> re.Pattern:
    return re.compile(re.escape(f"/auth?redirect={quote(path, safe='')}"))


def _sign_in_on_the_form(page: Page, account: Actor) -> None:
    page.get_by_label("Email").fill(account.email)
    page.get_by_label("Password", exact=True).fill(account.password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def _expect_the_chat(page: Page, path: str, answer: str) -> None:
    expect(page).to_have_url(re.compile(re.escape(path) + "$"), timeout=PAGE_TIMEOUT_MS)
    expect(page.get_by_text(answer)).to_be_visible(timeout=PAGE_TIMEOUT_MS)


def _wait_until_refused(instance: LaunchedInstance, token: str) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        answer = httpx.get(
            f"{instance.base_url}/api/v1/auths/", headers={"Authorization": f"Bearer {token}"}
        )
        if answer.status_code == 401:
            return
        time.sleep(0.5)
    raise AssertionError("the short session was still accepted after its expiry")


def test_a_link_opened_without_a_session_lands_on_its_page_after_signing_in(page, chat_owner):
    account, path, answer = chat_owner

    page.goto(path)
    expect(page).to_have_url(_sign_in_page_for(path), timeout=PAGE_TIMEOUT_MS)
    _sign_in_on_the_form(page, account)

    _expect_the_chat(page, path, answer)


def test_a_session_expiring_while_the_page_is_open_signs_in_again_to_the_same_page(
    page, chat_owner, session_length: SessionLength
):
    account, path, answer = chat_owner
    session_length.set("1m")
    page.goto(path)
    _sign_in_on_the_form(page, account)
    _expect_the_chat(page, path, answer)

    expect(page.get_by_text(EXPIRED_NOTICE)).to_be_visible(timeout=EXPIRY_WATCH_MS)
    expect(page).to_have_url(_sign_in_page_for(path))
    session_length.restore()
    _sign_in_on_the_form(page, account)

    _expect_the_chat(page, path, answer)


def test_a_page_opened_with_an_expired_session_signs_in_again_to_that_page(
    page, instance, chat_owner, session_length: SessionLength
):
    account, path, answer = chat_owner
    session_length.set("2s")
    expired = sign_in(instance, account.email, account.password)
    session_length.restore()
    _wait_until_refused(instance, expired)
    page.goto("/auth")
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    page.evaluate("token => localStorage.setItem('token', token)", expired)

    page.goto(path)
    expect(page).to_have_url(_sign_in_page_for(path), timeout=PAGE_TIMEOUT_MS)
    _sign_in_on_the_form(page, account)

    _expect_the_chat(page, path, answer)
