"""Regression: a failed SSO sign-in tells the person the identity provider failed.

`6a2aad92f` (#31629, issue #31627): when the provider let the person through but the account had
no email, the sign-in page said the email or password was wrong, though the person typed neither.
Walking "Continue with SSO" to such a failure now lands back on the sign-in page with a message
that asks them to contact their administrator and never blames the email or password.

Twin of integration/auth/test_sso_sign_in_failure_message.py.

Discriminates: passes on dev a5bc78300; with the fix reverted in a backend copy (the frontend
build stays clean) the sign-in page shows the email/password message instead. The good sign-in
passes on both and is the control.
"""

from __future__ import annotations

import re

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

PROVIDER_FAILED = (
    "Sign-in with your identity provider failed. Please contact your administrator for assistance."
)
EMAIL_OR_PASSWORD = re.compile("email or password", re.IGNORECASE)


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


def test_a_failed_sso_sign_in_shows_the_identity_provider_message(sign_in_page, idp):
    """Narrow (#31629): an account without an email is told the provider failed."""
    idp.sign_in_as(email=None)

    sso_button(sign_in_page).click()

    expect(sign_in_page.get_by_text(PROVIDER_FAILED)).to_be_visible()
    expect(sign_in_page.get_by_text(EMAIL_OR_PASSWORD)).to_have_count(0)
    expect(chat_input(sign_in_page)).to_have_count(0)


def test_a_good_sso_sign_in_shows_no_failure_message(sign_in_page, sso, idp):
    """Nearby: a working sign-in reaches the app and never shows the failure text."""
    idp.sign_in_as(roles=["user"])

    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        sso_button(sign_in_page).click()
        expect(chat_input(sign_in_page)).to_be_visible()

    expect(sign_in_page.get_by_text(PROVIDER_FAILED)).to_have_count(0)
