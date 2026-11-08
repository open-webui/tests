"""Journey: "Continue with SSO" with the sign-in state kept in Redis by starsessions.

With `ENABLE_STAR_SESSIONS_MIDDLEWARE` on, the session that carries the OAuth state from the
sign-in page to the provider and back lives in Redis, and the browser only holds its id. The
button walks the whole round trip and lands the person in the app. Twin, in the browser, of
integration/deps/test_server_sessions.py.

Discriminates: passes on the dev ef67cc3fa build; with starsessions' `RedisStore` reading nothing
back (patched in at import in a backend copy) the callback finds no state and the person is sent
back to the sign-in page.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import backends
from harness.oidc_provider import oauth_settings, shared_provider, sso_env
from harness.plain_http import serve_at_lan_host
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def redis_url():
    with backends.redis_server() as url:
        yield url


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sessions(instance_with, idp, redis_url):
    launched = instance_with(
        {**sso_env(idp), "ENABLE_STAR_SESSIONS_MIDDLEWARE": "true", "REDIS_URL": redis_url}
    )
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    serve_at_lan_host(launched)
    return launched


@pytest.fixture
def sign_in_page(browser, sessions):
    """The sign-in page in a browser that has never signed in."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=sessions.base_url
    )
    page = context.new_page()
    page.goto("/auth")
    yield page
    context.close()


def test_continue_with_sso_signs_the_person_in(sign_in_page, sessions, idp):
    idp.sign_in_as(
        sub="lamp-keeper", email="lamp@harbour.example", name="Lamp Keeper", roles=["user"]
    )
    with oauth_settings(sessions, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        sign_in_page.get_by_role("button", name="Continue with SSO").click()
        expect(chat_input(sign_in_page)).to_be_visible()

    expect(sign_in_page).not_to_have_url(re.compile(r"/auth"))
