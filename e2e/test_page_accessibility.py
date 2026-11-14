"""
Page Accessibility Tests for Open WebUI

One smoke test each walks the public, the signed-in user's and the admin's pages and checks that
each:
1. Loads successfully (HTTP 200 or appropriate redirect)
2. Has a visible body
3. Shows no JavaScript console errors (public pages)

An unauthenticated visit to a protected page is sent to /auth, and a regular user is sent away
from /admin. Each route list is walked in one test on one page, so one failure names every page
that broke.

Usage:
    # Run all page tests
    pytest e2e/test_page_accessibility.py -v

    # Run only public page tests
    pytest e2e/test_page_accessibility.py -v -m public

    # Run only authenticated page tests
    pytest e2e/test_page_accessibility.py -v -m auth_required

    # Run only admin page tests
    pytest e2e/test_page_accessibility.py -v -m admin_required
"""

import re

import pytest
from playwright.sync_api import BrowserContext, Page, expect

from conftest import (
    ADMIN_ROUTES,
    PUBLIC_ROUTES,
    USER_ROUTES,
    AppConfig,
    AuthHelper,
)


def wait_until_loaded(page: Page) -> None:
    """Wait until the app has removed its splash screen.

    Not `networkidle`: since 24e30d1cb the app's timezone update gets a `no-store` reply it never
    reads, which Chromium counts as a request in flight for as long as the page lives.
    """
    page.wait_for_load_state("load")
    expect(page.locator("#splash-screen")).to_have_count(0)


class PageAccessibilityResult:
    """Result of a page accessibility check."""

    def __init__(
        self,
        path: str,
        description: str,
        success: bool = False,
        title: str = "",
        console_errors: list[str] | None = None,
        load_time_ms: int = 0,
        error_message: str = "",
    ):
        self.path = path
        self.description = description
        self.success = success
        self.title = title
        self.console_errors = console_errors or []
        self.load_time_ms = load_time_ms
        self.error_message = error_message


def check_page_accessibility(
    page: Page,
    path: str,
    description: str,
    expected_redirect: str | None = None,
) -> PageAccessibilityResult:
    """
    Check if a page is accessible and loads correctly.

    Args:
        page: Playwright Page object
        path: URL path to visit
        description: Human-readable description of the page
        expected_redirect: If set, expect a redirect to this path

    Returns:
        PageAccessibilityResult with details of the check
    """
    console_errors: list[str] = []

    def handle_console(msg):
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", handle_console)

    result = PageAccessibilityResult(path=path, description=description)

    try:
        # Navigate to the page
        import time

        start_time = time.time()
        response = page.goto(path)
        wait_until_loaded(page)
        end_time = time.time()

        result.load_time_ms = int((end_time - start_time) * 1000)

        # Check HTTP response
        if response is None:
            result.error_message = "No response received"
            return result

        # Allow redirects (e.g., auth redirect)
        if expected_redirect:
            current_url = page.url
            if expected_redirect not in current_url:
                result.error_message = (
                    f"Expected redirect to {expected_redirect}, got {current_url}"
                )
                return result
        elif response.status >= 400:
            result.error_message = f"HTTP {response.status}"
            return result

        # Check for page title
        result.title = page.title()

        # Check for basic page structure
        # The page should have visible content (not blank)
        try:
            expect(page.locator("body")).to_be_visible()
        except AssertionError:
            result.error_message = "Page body not visible"
            return result

        # Record console errors
        result.console_errors = console_errors

        result.success = True

    except Exception as e:
        result.error_message = str(e)

    return result


# ============================================================================
# Public Page Tests
# ============================================================================


class TestPublicPages:
    """Tests for pages that don't require authentication."""

    @pytest.mark.public
    def test_public_pages_load(self, page: Page):
        """Test that every public page loads successfully."""
        failures: list[str] = []

        for path, description in PUBLIC_ROUTES:
            result = check_page_accessibility(page, path, description)
            if not result.success:
                failures.append(f"'{description}' ({path}) failed to load: {result.error_message}")
                continue
            # warnings are okay
            critical_errors = [e for e in result.console_errors if "error" in e.lower()]
            if critical_errors:
                failures.append(f"'{description}' ({path}) had console errors: {critical_errors}")

        assert PUBLIC_ROUTES
        assert not failures, "\n".join(failures)

    @pytest.mark.public
    def test_auth_page_has_login_form(self, page: Page):
        """Test that the auth page has a functional login form."""
        page.goto("/auth")
        wait_until_loaded(page)

        # Check for email input (could be email or username for LDAP)
        email_input = page.locator('input[autocomplete="email"], input[autocomplete="username"]')
        expect(email_input.first).to_be_visible()

        # Check for password input (current-password for signin, new-password for signup)
        password_input = page.locator('input[type="password"]')
        expect(password_input.first).to_be_visible()

        # Check for submit button
        submit_button = page.locator('button[type="submit"]')
        expect(submit_button).to_be_visible()


# ============================================================================
# Authenticated User Page Tests
# ============================================================================


class TestUserPages:
    """Tests for pages that require regular user authentication."""

    @pytest.mark.auth_required
    def test_user_pages_load(self, authenticated_page: Page):
        """Test that every user page loads successfully when authenticated."""
        failures: list[str] = []

        for path, description in USER_ROUTES:
            result = check_page_accessibility(authenticated_page, path, description)
            if not result.success:
                failures.append(f"'{description}' ({path}) failed to load: {result.error_message}")

        assert USER_ROUTES
        assert not failures, "\n".join(failures)

    @pytest.mark.auth_required
    def test_home_page_structure(self, authenticated_page: Page):
        """Test that the home page has expected structure."""
        authenticated_page.goto("/")
        wait_until_loaded(authenticated_page)

        # Should have visible content - Open WebUI uses #app as the main container
        main_content = authenticated_page.locator("main, [role='main'], #main, #app, .app")
        expect(main_content.first).to_be_visible(timeout=10000)

        # Page should have a title
        assert authenticated_page.title(), "Page should have a title"

    @pytest.mark.auth_required
    def test_unauthenticated_redirect(self, page: Page):
        """Test that protected pages redirect unauthenticated users."""
        # Try to access a protected page without authentication
        page.goto("/workspace")

        # Should be redirected to auth page
        expect(page).to_have_url(re.compile(r"/auth"))


# ============================================================================
# Admin Page Tests
# ============================================================================


class TestAdminPages:
    """Tests for pages that require admin authentication."""

    @pytest.mark.admin_required
    def test_admin_pages_load(self, admin_page: Page):
        """Test that every admin page loads successfully when authenticated as admin."""
        failures: list[str] = []

        for path, description in ADMIN_ROUTES:
            result = check_page_accessibility(admin_page, path, description)
            if not result.success:
                failures.append(
                    f"Admin page '{description}' ({path}) failed to load: {result.error_message}"
                )

        assert ADMIN_ROUTES
        assert not failures, "\n".join(failures)

    @pytest.mark.admin_required
    def test_regular_user_cannot_access_admin(
        self,
        context: BrowserContext,
        config: AppConfig,
    ):
        """Test that regular (non-admin) users cannot access admin pages.

        This test is skipped if TEST_USER and ADMIN_USER are the same,
        since we need a non-admin user to test access denial.
        """
        # Skip if using the same credentials for both users
        if (
            config.test_user_email == config.admin_user_email
            and config.test_user_password == config.admin_user_password
        ):
            pytest.skip(
                "TEST_USER and ADMIN_USER are the same - cannot test non-admin access denial"
            )

        # Login as regular user (not admin)
        page = context.new_page()
        auth = AuthHelper(page, config)

        if not auth.login(config.test_user_email, config.test_user_password):
            pytest.skip("Could not authenticate as regular test user")

        page.goto("/admin")

        # Regular user is sent away from the admin page
        expect(page).not_to_have_url(re.compile(r"/admin"))

        page.close()
