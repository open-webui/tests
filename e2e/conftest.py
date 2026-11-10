"""Point the browser suite at the shared scratch instance and give every account its own browser.

The instance serves the checkout's built frontend (`npm run build`, or `OPEN_WEBUI_BUILD_DIR`)
and answers every chat from the scripted provider, so a test controls what the model says.
Each signed-in page lives in its own browser context: two accounts in one context would share
one `localStorage` token and the second sign-in would silently replace the first. Every context
is traced; a failing test leaves its traces in `test-results/`, which `playwright show-trace`
replays step by step.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Generator, Iterator

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, Playwright
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from conftest import AppConfig
from harness import plain_http
from harness.actors import Actor
from harness.fixtures import TEST_USER_EMAIL, TEST_USER_PASSWORD
from harness.instance import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    LaunchedInstance,
    resolve_backend,
    resolve_frontend_build,
)
from harness.prepared_data import serving
from harness.reply_gate import hold_new_chat_events
from harness.terminals import wait_until_addresses_settle

ARTIFACTS_DIR = Path("test-results")
LOAD_TRIES = 3
LOAD_CHECK_MS = 1000
MICROPHONE_BROWSERS = {"speaking_browser", "capturing_browser"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"report_{report.when}", report)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # the browser offers the microphone only in a secure context
    for item in items:
        if MICROPHONE_BROWSERS & set(item.fixturenames):
            item.add_marker(pytest.mark.secure_context)


def _test_failed(item: pytest.Item) -> bool:
    reports = (getattr(item, f"report_{when}", None) for when in ("setup", "call"))
    return any(report is not None and report.failed for report in reports)


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
    # the changelog modal covers an admin's first page load
    with actor.client() as client:
        dismissed = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}}
        )
    dismissed.raise_for_status()


@pytest.fixture(scope="session")
def user_token(user: Actor) -> str:
    _dismiss_first_run_modals(user)
    return user.token


@pytest.fixture(scope="session")
def admin_token(admin: Actor) -> str:
    _dismiss_first_run_modals(admin)
    return admin.token


def _new_context(
    browser: Browser, config: AppConfig, base_url: str, **context_options
) -> BrowserContext:
    options = {"viewport": {"width": 1920, "height": 1080}, **context_options}
    context = browser.new_context(base_url=base_url, **options)
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    context.tracing.start(screenshots=True, snapshots=True)
    return context


def _close_context(context: BrowserContext, item: pytest.Item, label: str) -> None:
    try:
        if _test_failed(item):
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", item.nodeid)
            context.tracing.stop(path=str(ARTIFACTS_DIR / f"{name}-{label}.zip"))
        else:
            context.tracing.stop()
    finally:
        context.close()


@pytest.fixture
def context(
    browser: Browser, config: AppConfig, request: pytest.FixtureRequest
) -> Generator[BrowserContext, None, None]:
    """The signed-out browser behind `page`, traced like the signed-in ones."""
    browser_context = _new_context(browser, config, config.base_url)
    yield browser_context
    _close_context(browser_context, request.node, "signed-out")


# The splash stays until the app has its config and account; the signed-in layout then shows a
# spinner beside the sidebar until its data is in and #main-content replaces it.
APP_LOADED = """() => !document.getElementById('splash-screen')
    && (!document.getElementById('sidebar-new-chat-button')
        || !!document.getElementById('main-content'))"""


def _loaded(page: Page, path: str, load_timeout_ms: float) -> bool:
    """Load `path` and wait for the app; False when the browser dropped part of the load because
    this machine's network addresses changed under it (net::ERR_NETWORK_CHANGED)."""
    dropped: list[str] = []

    def note_dropped(request) -> None:
        if "ERR_NETWORK_CHANGED" in (request.failure or ""):
            dropped.append(request.url)

    page.on("requestfailed", note_dropped)
    deadline = time.monotonic() + load_timeout_ms / 1000
    try:
        page.goto(path)
        # a cold load is slow on a busy runner; the test's first default wait should not pay for it
        while not dropped:
            try:
                page.wait_for_function(APP_LOADED, timeout=LOAD_CHECK_MS)
                break
            except PlaywrightTimeoutError:
                if time.monotonic() >= deadline:
                    raise
    except PlaywrightError:
        if not dropped:
            raise
    finally:
        page.remove_listener("requestfailed", note_dropped)
    return not dropped


def _signed_in_page(
    context: BrowserContext,
    token: str,
    load_timeout_ms: float,
    hold_events: bool = True,
    path: str = "/",
) -> Page:
    """A page on `path` whose localStorage carries the session token, handed over once loaded.

    Written through an init script: the sign-in route clears the token as it loads, so a value
    written by visiting /auth first is gone again before the app reads it.
    """
    page = context.new_page()
    if hold_events:
        hold_new_chat_events(page)
    page.add_init_script(
        f"try {{ localStorage.setItem('token', {json.dumps(token)}); }} catch (e) {{}}"
    )
    for _ in range(LOAD_TRIES):
        # a container that comes or goes (a terminal's machine) changes this machine's addresses
        wait_until_addresses_settle()
        if _loaded(page, path, load_timeout_ms):
            break
    else:
        pytest.fail(f"{LOAD_TRIES} loads of {path} in a row were cut short by a network change")
    secure = page.evaluate("isSecureContext")
    assert secure is plain_http.is_secure_context_test(), f"the page opened at {page.url}"
    return page


@pytest.fixture
def page_for(
    browser: Browser, config: AppConfig, request: pytest.FixtureRequest
) -> Generator[Callable[[Actor], Page], None, None]:
    """`page_for(actor)` opens a signed-in page on the actor's instance, in a browser of its own.

    `path` opens that page directly in place of the chat page, such as `path="/admin/users"`.
    Further keywords go to the browser context, such as `timezone_id` for a browser in another zone
    or `utils.phone.PHONE` for a phone's screen.
    """
    opened: list[BrowserContext] = []
    hold_events = request.node.get_closest_marker("real_socket_timing") is None

    def open_page(actor: Actor, path: str = "/", **context_options) -> Page:
        _dismiss_first_run_modals(actor)
        browser_context = _new_context(browser, config, actor.base_url, **context_options)
        opened.append(browser_context)
        return _signed_in_page(
            browser_context, actor.token, config.navigation_timeout, hold_events, path
        )

    yield open_page
    with contextlib.ExitStack() as closing:
        for index, browser_context in enumerate(opened):
            closing.callback(_close_context, browser_context, request.node, f"browser-{index}")


@pytest.fixture
def lan_page_for(playwright_instance: Playwright, config) -> Iterator[Callable[[Actor], Page]]:
    """`lan_page_for(actor)` opens a signed-in page on the instance through `LAN_HOST`, over plain
    HTTP, where the page is not a secure context and the browser offers no `crypto.randomUUID`."""
    browser = playwright_instance.chromium.launch(headless=config.headless)

    def open_page(actor: Actor) -> Page:
        _dismiss_first_run_modals(actor)
        context = browser.new_context(
            base_url=plain_http.lan_url(actor.base_url), viewport={"width": 1920, "height": 1080}
        )
        context.set_default_timeout(config.default_timeout)
        page = context.new_page()
        page.add_init_script(
            f"try {{ localStorage.setItem('token', {json.dumps(actor.token)}); }} catch (e) {{}}"
        )
        page.goto("/")
        page.wait_for_function(APP_LOADED, timeout=config.navigation_timeout)
        assert page.evaluate("isSecureContext") is False
        return page

    yield open_page
    browser.close()


@pytest.fixture
def authenticated_page(page_for: Callable[[Actor], Page], user: Actor, user_token: str) -> Page:
    return page_for(user)


@pytest.fixture
def admin_page(page_for: Callable[[Actor], Page], admin: Actor, admin_token: str) -> Page:
    return page_for(admin)


@pytest.fixture
def fresh_install(tmp_path):
    """The checkout serving its built frontend on an empty data directory, before any account."""
    backend = resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    build = resolve_frontend_build(backend)
    if build is None:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    settings = {"FRONTEND_BUILD_DIR": str(build), "CORS_ALLOW_ORIGIN": "*"}
    with serving(tmp_path, settings=settings) as server:
        yield server
