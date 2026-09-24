"""Point the browser suite at the shared scratch instance and give every account its own browser.

The instance serves the checkout's built frontend (`npm run build`, or `OPEN_WEBUI_BUILD_DIR`)
and answers every chat from the scripted provider, so a test controls what the model says.
Each signed-in page lives in its own browser context: two accounts in one context would share
one `localStorage` token and the second sign-in would silently replace the first.
"""

from __future__ import annotations

import json
from typing import Callable, Generator

import pytest
from playwright.sync_api import Browser, BrowserContext, Page

from conftest import AppConfig
from harness.actors import Actor
from harness.fixtures import TEST_USER_EMAIL, TEST_USER_PASSWORD
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD, LaunchedInstance


@pytest.fixture(scope="session")
def e2e_instance(instance: LaunchedInstance) -> LaunchedInstance:
    if not instance.serves_frontend:
        pytest.skip(
            "no built frontend (run `npm run build` in the checkout or set OPEN_WEBUI_BUILD_DIR)"
        )
    return instance


@pytest.fixture(scope="session")
def config(e2e_instance: LaunchedInstance) -> AppConfig:
    return AppConfig(
        base_url=e2e_instance.base_url,
        test_user_email=TEST_USER_EMAIL,
        test_user_password=TEST_USER_PASSWORD,
        admin_user_email=ADMIN_EMAIL,
        admin_user_password=ADMIN_PASSWORD,
    )


def _dismiss_first_run_modals(actor: Actor) -> None:
    # The changelog modal covers the page on first load and is gated on this flag.
    with actor.client() as client:
        client.post("/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}})


@pytest.fixture(scope="session")
def user_token(user: Actor) -> str:
    _dismiss_first_run_modals(user)
    return user.token


@pytest.fixture(scope="session")
def admin_token(admin: Actor) -> str:
    _dismiss_first_run_modals(admin)
    return admin.token


def _new_context(browser: Browser, config: AppConfig) -> BrowserContext:
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=config.base_url
    )
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    return context


def _signed_in_page(context: BrowserContext, token: str) -> Page:
    """A page whose localStorage already carries the session token.

    Written through an init script: the sign-in route clears the token as it loads, so a value
    written by visiting /auth first is gone again before the app reads it.
    """
    page = context.new_page()
    page.add_init_script(
        f"try {{ localStorage.setItem('token', {json.dumps(token)}); }} catch (e) {{}}"
    )
    page.goto("/")
    return page


@pytest.fixture
def page_for(browser: Browser, config: AppConfig) -> Generator[Callable[[Actor], Page], None, None]:
    """`page_for(actor)` opens a signed-in page for that account in a browser of its own."""
    contexts: list[BrowserContext] = []

    def open_page(actor: Actor) -> Page:
        _dismiss_first_run_modals(actor)
        context = _new_context(browser, config)
        contexts.append(context)
        return _signed_in_page(context, actor.token)

    yield open_page
    for context in contexts:
        context.close()


@pytest.fixture
def authenticated_page(page_for: Callable[[Actor], Page], user: Actor, user_token: str) -> Page:
    return page_for(user)


@pytest.fixture
def admin_page(page_for: Callable[[Actor], Page], admin: Actor, admin_token: str) -> Page:
    return page_for(admin)
