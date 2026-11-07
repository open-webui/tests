"""Journey: approving a native app's sign-in in the browser.

A native app opens a login request and sends the person to `/auth/sessions/<id>`. Someone already
signed in is handed straight back to the app's loopback callback with a one-time code and the
app's state, and the app trades that code for a session of the same account. A signed-out visitor
signs in first and then continues to the app the same way. A request that was already used or
does not exist, a malformed link and a pending account each end on "Could not complete sign-in"
with the reason, and the app gets no callback. The page has no approve or deny buttons: opening it
while signed in is the approval. Added in 404d27a36 and 6cd0691eb (0.13.1); the server side is
integration/auth/test_native_app_sessions.py.

Discriminates: on dev f659c0d12, a backend copy whose approval proof skips the role check turns
the pending test red (the pending account reaches the app's callback), and one whose approval
skips the already-approved checks turns the used-request test red (a second visit approves again).
A copy of the f659c0d12 build whose sign-in page ignores where to continue turns the signed-out and
pending tests red (both land on the chat), and one without the link check on the approval page
turns the malformed-link test red; the signed-in and used-request tests pass on that build.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import re
import secrets
import uuid
from typing import Iterator

import httpx
import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor, sign_in
from harness.listener import Listener, text_answer
from harness.mfa import new_address

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SESSIONS = "/api/v1/auths/sessions"
APP_PAGE = "You can return to the app now"


class NativeApp:
    """An app waiting on its loopback callback with the PKCE pair of one sign-in."""

    def __init__(self, base_url: str, callback: Listener):
        self.base_url = base_url
        self.address = new_address()
        self.name = f"Desktop {uuid.uuid4().hex[:8]}"
        self.callback = f"{callback.base_url}/callback"
        self.received = callback.received
        self.verifier = secrets.token_urlsafe(48)
        self.state = secrets.token_urlsafe(32)
        challenge = hashlib.sha256(self.verifier.encode()).digest()
        opened = self._post(
            "/requests",
            {
                "name": self.name,
                "redirect_uri": self.callback,
                "state": self.state,
                "code_challenge": base64.urlsafe_b64encode(challenge).rstrip(b"=").decode(),
            },
        )
        assert opened.status_code == 200, f"opening the login request failed: {opened.text}"
        self.session_id = opened.json()["session_id"]

    @property
    def approval_page(self) -> str:
        return f"/auth/sessions/{self.session_id}"

    def _post(self, path: str, body: dict) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}{SESSIONS}{path}",
            json=body,
            headers={"X-Forwarded-For": self.address},
            timeout=60.0,
        )

    def code_from(self, page: Page) -> str:
        expect(page.get_by_text(APP_PAGE)).to_be_visible()
        landed = httpx.URL(page.url)
        assert landed.copy_with(query=None) == httpx.URL(self.callback), landed
        assert landed.params["state"] == self.state, "the callback lost the app's state"
        return landed.params["code"]

    def signed_in_account(self, code: str) -> str:
        exchanged = self._post(
            "/exchange",
            {
                "session_id": self.session_id,
                "code": code,
                "code_verifier": self.verifier,
                "redirect_uri": self.callback,
            },
        )
        assert exchanged.status_code == 200, f"the app could not trade its code: {exchanged.text}"
        me = httpx.get(
            f"{self.base_url}/api/v1/auths/",
            headers={"Authorization": f"Bearer {exchanged.json()['token']}"},
            timeout=60.0,
        )
        assert me.status_code == 200, me.text
        return me.json()["id"]


@pytest.fixture
def callback(listener: Listener) -> Iterator[Listener]:
    listener.route("GET", "/callback", text_answer(f"<p>{APP_PAGE}</p>"))
    yield listener


def signed_in(instance, account: Actor) -> Actor:
    """The account with a session from the sign-in form, which carries how the person signed in."""
    return dataclasses.replace(account, token=sign_in(instance, account.email, account.password))


def sign_in_through_the_form(page: Page, account: Actor) -> None:
    expect(page).to_have_url(re.compile(r"/auth\?redirect="))
    page.get_by_label("Email").fill(account.email)
    page.get_by_label("Password", exact=True).fill(account.password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def expect_refusal(page: Page, reason: str, app: NativeApp | None = None) -> None:
    expect(page.get_by_role("heading", name="Could not complete sign-in")).to_be_visible()
    expect(page.get_by_role("alert")).to_have_text(reason)
    expect(page.get_by_text("Return to the app and sign in again.")).to_be_visible()
    if app is not None:
        assert app.received == [], "the app's callback was called after a refused sign-in"


def test_a_signed_in_person_is_sent_back_to_the_app_signed_in(
    page_for, make_user, e2e_instance, callback
):
    account = make_user()
    app = NativeApp(e2e_instance.base_url, callback)
    page = page_for(signed_in(e2e_instance, account))

    page.goto(app.approval_page)

    assert app.signed_in_account(app.code_from(page)) == account.id
    assert len(app.received) == 1


def test_a_signed_out_visitor_signs_in_first_and_then_continues_to_the_app(
    page, make_user, e2e_instance, callback
):
    account = make_user()
    app = NativeApp(e2e_instance.base_url, callback)

    page.goto(app.approval_page)
    sign_in_through_the_form(page, account)

    assert app.signed_in_account(app.code_from(page)) == account.id


def test_a_used_or_unknown_login_request_is_refused(page_for, make_user, e2e_instance, callback):
    account = make_user()
    app = NativeApp(e2e_instance.base_url, callback)
    page = page_for(signed_in(e2e_instance, account))
    page.goto(app.approval_page)
    code = app.code_from(page)
    callback.received.clear()

    page.goto(app.approval_page)
    expect_refusal(page, "Login request expired or already authorized.", app)
    app.signed_in_account(code)
    page.goto(app.approval_page)
    expect_refusal(page, "Login request expired or already authorized.", app)

    page.goto(f"/auth/sessions/{uuid.uuid4()}")
    expect_refusal(page, "Login request expired or already authorized.", app)


def test_a_malformed_login_link_is_refused(page_for, make_user, e2e_instance):
    page = page_for(signed_in(e2e_instance, make_user()))

    page.goto("/auth/sessions/not-a-login-request")

    expect_refusal(page, "Invalid login request.")


def test_a_pending_account_cannot_sign_an_app_in(page, make_user, e2e_instance, callback):
    app = NativeApp(e2e_instance.base_url, callback)

    page.goto(app.approval_page)
    sign_in_through_the_form(page, make_user(role="pending"))

    expect_refusal(page, "Account approval is required.", app)
