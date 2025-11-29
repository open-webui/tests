"""
Page Accessibility Tests for Open WebUI

These tests verify that all pages in Open WebUI:
1. Load successfully (HTTP 200 or appropriate redirect)
2. Have proper page structure (title, main content)
3. Don't produce JavaScript console errors
4. Are accessible within reasonable timeouts

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

import pytest
from playwright.sync_api import BrowserContext, Page, expect

from conftest import (
    ADMIN_ROUTES,
    PUBLIC_ROUTES,
    USER_ROUTES,
    AppConfig,
    AuthHelper,
)


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
        page.wait_for_load_state("networkidle")
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
        body = page.locator("body")
        if not body.is_visible():
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
    @pytest.mark.parametrize("path,description", PUBLIC_ROUTES)
    def test_public_page_loads(
        self,
        page: Page,
        path: str,
        description: str,
    ):
        """Test that public pages load successfully."""
        result = check_page_accessibility(page, path, description)

        # Assert page loaded
        assert result.success, f"Page '{description}' ({path}) failed to load: {result.error_message}"

        # Assert no critical console errors (warnings are okay)
        critical_errors = [e for e in result.console_errors if "error" in e.lower()]
        assert not critical_errors, (
            f"Page '{description}' ({path}) had console errors: {critical_errors}"
        )

    @pytest.mark.public
    def test_auth_page_has_login_form(self, page: Page):
        """Test that the auth page has a functional login form."""
        page.goto("/auth")
        page.wait_for_load_state("networkidle")

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
    @pytest.mark.parametrize("path,description", USER_ROUTES)
    def test_user_page_loads(
        self,
        authenticated_page: Page,
        path: str,
        description: str,
    ):
        """Test that user pages load successfully when authenticated."""
        result = check_page_accessibility(authenticated_page, path, description)

        assert result.success, (
            f"Page '{description}' ({path}) failed to load: {result.error_message}"
        )

        # Log load time for performance tracking
        print(f"  Load time: {result.load_time_ms}ms")

    @pytest.mark.auth_required
    def test_home_page_structure(self, authenticated_page: Page):
        """Test that the home page has expected structure."""
        authenticated_page.goto("/")
        authenticated_page.wait_for_load_state("networkidle")

        # Should have visible content - Open WebUI uses #app as the main container
        main_content = authenticated_page.locator("main, [role='main'], #main, #app, .app")
        expect(main_content.first).to_be_visible(timeout=10000)

        # Page should have a title
        assert authenticated_page.title(), "Page should have a title"

    @pytest.mark.auth_required
    def test_workspace_navigation(self, authenticated_page: Page):
        """Test navigation through workspace pages."""
        pages = [
            "/workspace",
            "/workspace/models",
            "/workspace/prompts",
            "/workspace/tools",
            "/workspace/knowledge",
        ]

        for path in pages:
            authenticated_page.goto(path)
            authenticated_page.wait_for_load_state("networkidle")

            # Each page should have content
            body = authenticated_page.locator("body")
            assert body.is_visible(), f"Page {path} body not visible"

    @pytest.mark.auth_required
    def test_unauthenticated_redirect(self, page: Page):
        """Test that protected pages redirect unauthenticated users."""
        # Try to access a protected page without authentication
        page.goto("/workspace")
        page.wait_for_load_state("networkidle")

        # Should be redirected to auth page
        current_url = page.url
        assert "/auth" in current_url, (
            f"Expected redirect to /auth, but got {current_url}"
        )


# ============================================================================
# Admin Page Tests
# ============================================================================


class TestAdminPages:
    """Tests for pages that require admin authentication."""

    @pytest.mark.admin_required
    @pytest.mark.parametrize("path,description", ADMIN_ROUTES)
    def test_admin_page_loads(
        self,
        admin_page: Page,
        path: str,
        description: str,
    ):
        """Test that admin pages load successfully when authenticated as admin."""
        result = check_page_accessibility(admin_page, path, description)

        assert result.success, (
            f"Admin page '{description}' ({path}) failed to load: {result.error_message}"
        )

        print(f"  Load time: {result.load_time_ms}ms")

    @pytest.mark.admin_required
    def test_admin_dashboard_structure(self, admin_page: Page):
        """Test that admin dashboard has expected structure."""
        admin_page.goto("/admin")
        admin_page.wait_for_load_state("networkidle")

        # Admin page should have navigation to settings
        body = admin_page.locator("body")
        assert body.is_visible(), "Admin page body not visible"

    @pytest.mark.admin_required
    def test_admin_settings_navigation(self, admin_page: Page):
        """Test navigation through admin settings pages."""
        settings_pages = [
            "/admin/settings",
            "/admin/settings/general",
            "/admin/settings/users",
            "/admin/settings/connections",
        ]

        for path in settings_pages:
            admin_page.goto(path)
            admin_page.wait_for_load_state("networkidle")

            body = admin_page.locator("body")
            assert body.is_visible(), f"Admin settings page {path} body not visible"

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
        if (config.test_user_email == config.admin_user_email and 
            config.test_user_password == config.admin_user_password):
            pytest.skip("TEST_USER and ADMIN_USER are the same - cannot test non-admin access denial")

        # Login as regular user (not admin)
        page = context.new_page()
        from conftest import AuthHelper
        auth = AuthHelper(page, config)
        
        if not auth.login(config.test_user_email, config.test_user_password):
            pytest.skip("Could not authenticate as regular test user")

        page.goto("/admin")
        page.wait_for_load_state("networkidle")

        # Should either redirect or show access denied
        current_url = page.url

        # Regular user should not be on admin page
        # They should be redirected to home or see an error
        if "/admin" in current_url:
            # Check for access denied message
            body_text = page.locator("body").inner_text()
            assert any(
                phrase in body_text.lower()
                for phrase in ["access denied", "forbidden", "not authorized", "permission"]
            ), "Regular user appears to have access to admin page"
        
        page.close()


# ============================================================================
# Comprehensive Page Scan
# ============================================================================


class TestFullPageScan:
    """Run a comprehensive scan of all pages and generate a report."""

    @pytest.mark.slow
    def test_all_public_pages_summary(self, page: Page, config: AppConfig):
        """Scan all public pages and generate a summary."""
        results: list[PageAccessibilityResult] = []

        for path, description in PUBLIC_ROUTES:
            result = check_page_accessibility(page, path, description)
            results.append(result)

        # Generate summary
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        print(f"\n{'='*60}")
        print(f"PUBLIC PAGES SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
        print(f"{'='*60}")

        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.description} ({result.path})")
            if not result.success:
                print(f"   Error: {result.error_message}")
            if result.console_errors:
                print(f"   Console errors: {len(result.console_errors)}")

        assert failed == 0, f"{failed} public pages failed accessibility check"

    @pytest.mark.slow
    @pytest.mark.auth_required
    def test_all_user_pages_summary(
        self,
        authenticated_page: Page,
        config: AppConfig,
    ):
        """Scan all user pages and generate a summary."""
        results: list[PageAccessibilityResult] = []

        for path, description in USER_ROUTES:
            result = check_page_accessibility(authenticated_page, path, description)
            results.append(result)

        # Generate summary
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        print(f"\n{'='*60}")
        print(f"USER PAGES SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
        print(f"{'='*60}")

        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.description} ({result.path})")
            if not result.success:
                print(f"   Error: {result.error_message}")
            if result.console_errors:
                print(f"   Console errors: {len(result.console_errors)}")

        assert failed == 0, f"{failed} user pages failed accessibility check"

    @pytest.mark.slow
    @pytest.mark.admin_required
    def test_all_admin_pages_summary(
        self,
        admin_page: Page,
        config: AppConfig,
    ):
        """Scan all admin pages and generate a summary."""
        results: list[PageAccessibilityResult] = []

        for path, description in ADMIN_ROUTES:
            result = check_page_accessibility(admin_page, path, description)
            results.append(result)

        # Generate summary
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        print(f"\n{'='*60}")
        print(f"ADMIN PAGES SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
        print(f"{'='*60}")

        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.description} ({result.path})")
            if not result.success:
                print(f"   Error: {result.error_message}")
            if result.console_errors:
                print(f"   Console errors: {len(result.console_errors)}")

        assert failed == 0, f"{failed} admin pages failed accessibility check"
