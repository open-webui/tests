"""Regression: the sign-in page offers SSO only while OAuth is switched on, and the button works.

`71f8b6d5b` (#26988) gave the admin panel a switch for OAuth. While it is off `/api/config`
lists no providers, so the sign-in page shows no "Continue with SSO" button (and the routes
behind it answer 404, pinned by the integration twin). With it on, the button walks the whole
provider round trip and lands the person signed in.

Twin of unit/security/test_oauth_session_lifecycle.py (its ENABLE_OAUTH part).

Discriminates: passes on dev bbfa876af; with 71f8b6d5b's provider-list gate reverted in a
backend copy the button stays on the page while OAuth is off. The sign-in journey passes on
both and is the control.

With automatic SSO on, an ended session signs back in without the provider's answer being
refused (2fb2e9ea2, PR open-webui/open-webui#33791, issue #33783). The provider holds its
answer until the page's sign-out is answered, which makes the race certain. Passes on dev
e8c0ad85e and fails on that build with 2fb2e9ea2 reverted (the sign-in ends on an error).
"""

from __future__ import annotations

import re
import threading

import httpx
import pytest
from playwright.sync_api import expect

from harness.oidc_provider import oauth_settings, shared_provider, sso_env
from harness.plain_http import serve_at_lan_host
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    launched = instance_with(sso_env(idp))
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    serve_at_lan_host(launched)
    return launched


@pytest.fixture
def sign_in_page(browser, sso):
    """The sign-in page in a browser that has never signed in."""
    context = browser.new_context(viewport={"width": 1920, "height": 1080}, base_url=sso.base_url)
    page = context.new_page()
    page.goto("/auth")
    yield page
    context.close()


def sso_button(page):
    return page.get_by_role("button", name="Continue with SSO")


def test_the_sign_in_page_offers_sso_only_while_oauth_is_on(sign_in_page, sso):
    """Narrow: switching OAuth off takes the provider button off the sign-in page."""
    expect(sso_button(sign_in_page)).to_be_visible()

    with oauth_settings(sso, ENABLE_OAUTH=False):
        sign_in_page.reload()
        expect(sign_in_page.get_by_role("button", name="Sign in")).to_be_visible()
        expect(sso_button(sign_in_page)).to_have_count(0)


def test_continue_with_sso_signs_the_person_in(sign_in_page, sso, idp):
    """Nearby: the button takes the person through the provider and into the app."""
    idp.sign_in_as(roles=["user"])
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        sso_button(sign_in_page).click()
        expect(chat_input(sign_in_page)).to_be_visible()

    expect(sign_in_page).not_to_have_url(re.compile(r"/auth"))
    assert idp.requests_to("/authorize"), "the button never reached the provider"


def _poll(page, done, limit_ms: int) -> bool:
    """Let the page run until `done()` holds, at most `limit_ms`; whether it held."""
    for _ in range(limit_ms // 100):
        if done():
            return True
        page.wait_for_timeout(100)
    return done()


def test_automatic_sso_signs_an_ended_session_back_in_without_a_state_mismatch(
    sign_in_page, sso, idp, preserve
):
    """Regression for open-webui/open-webui#33783 (fix 2fb2e9ea2, PR #33791).

    The page signed the ended session out in the background while automatic SSO already started
    the new sign-in, so the sign-out's answer wiped the sign-in's state and the provider's answer
    was refused as a state mismatch. The sign-out is held here until the new sign-in has begun,
    the way a slow server answers it, and the provider holds its answer until the sign-out is
    answered.
    """
    idp.sign_in_as(roles=["user"])
    page = sign_in_page
    # automatic SSO only redirects a deployment that signs in by SSO alone
    preserve("admin_config", on=sso)
    with sso.client() as admin:
        current = admin.get("/api/v1/auths/admin/config").json()
        saved = admin.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_LOGIN_FORM": False}
        )
        assert saved.status_code == 200, saved.text
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True, OAUTH_AUTO_REDIRECT=True):
        page.goto("/auth")
        expect(chat_input(page)).to_be_visible()
        token = page.evaluate("() => localStorage.token")
        bearer = {"Authorization": f"Bearer {token}"}
        with httpx.Client(base_url=sso.base_url) as client:
            account_id = client.get("/api/v1/auths/", headers=bearer).json()["id"]
            with sso.client() as admin:
                ended = admin.post(f"/api/v1/users/{account_id}/sessions/revoke")
            assert ended.status_code == 200, ended.text
            assert client.get("/api/v1/auths/", headers=bearer).status_code == 401

        signouts, answered = [], []
        page.route("**/api/v1/auths/signout", lambda route: signouts.append(route))
        page.on("response", lambda response: answered.append(response.url))
        logins = len(idp.requests_to("/authorize"))
        idp.authorize_gate = threading.Event()
        try:
            page.evaluate(
                "() => fetch('/api/v1/auths/', "
                "{headers: {Authorization: `Bearer ${localStorage.token}`}})"
            )
            assert _poll(page, lambda: signouts, 15_000), "the ended session was never signed out"
            # the new sign-in may only start once the old session is signed out
            _poll(page, lambda: len(idp.requests_to("/authorize")) > logins, 3_000)
            for route in signouts:
                route.continue_()
            signed_out = _poll(
                page, lambda: any("/auths/signout" in url for url in answered), 5_000
            )
            assert signed_out, "the sign-out never got its answer"
        finally:
            idp.authorize_gate.set()
            idp.authorize_gate = None
        assert _poll(page, lambda: any("/callback?" in url for url in answered), 15_000), (
            "the provider's answer never came back to Open WebUI"
        )
        page.wait_for_load_state()

        expect(chat_input(page), "the automatic sign-in did not complete").to_be_visible()
    expect(page, "the provider's answer was refused (#33783)").not_to_have_url(re.compile("error="))
    assert len(idp.requests_to("/authorize")) > logins, "the provider was never asked again"
