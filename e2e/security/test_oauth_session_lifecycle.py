"""Regression: the sign-in page offers SSO only while OAuth is switched on, and the button works.

`71f8b6d5b` (#26988) gave the admin panel a switch for OAuth. While it is off `/api/config`
lists no providers, so the sign-in page shows no "Continue with SSO" button (and the routes
behind it answer 404, pinned by the integration twin). With it on, the button walks the whole
provider round trip and lands the person signed in.

Twin of unit/security/test_oauth_session_lifecycle.py (its ENABLE_OAUTH part).

Discriminates: passes on dev bbfa876af; with 71f8b6d5b's provider-list gate reverted in a
backend copy the button stays on the page while OAuth is off. The sign-in journey passes on
both and is the control.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness.oidc_provider import oauth_settings, shared_provider, sso_env
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
