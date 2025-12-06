"""
Open WebUI Test Suite - Shared Fixtures and Configuration

This module provides pytest fixtures for:
- Playwright browser management
- Authentication (regular user and admin)
- Page navigation utilities
- Console error tracking

Works with both Docker and Python/pip Open WebUI deployments.

Authentication Strategy:
- Login is performed once per session for each user type
- Browser storage state (cookies, localStorage) is saved and reused
- This prevents rate limiting from repeated login attempts
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

# Load environment variables
load_dotenv()


@dataclass
class AppConfig:
    """Test configuration loaded from environment variables."""

    base_url: str = os.getenv("OPEN_WEBUI_URL", "http://localhost:8080")
    
    # Credentials
    test_user_email: str = os.getenv("TEST_USER_EMAIL", "test@example.com")
    test_user_password: str = os.getenv("TEST_USER_PASSWORD", "testpassword123")
    admin_user_email: str = os.getenv("ADMIN_USER_EMAIL", "admin@example.com")
    admin_user_password: str = os.getenv("ADMIN_USER_PASSWORD", "adminpassword123")
    
    # Browser settings
    headless: bool = os.getenv("HEADLESS", "true").lower() == "true"
    slow_mo: int = int(os.getenv("SLOW_MO", "0"))
    
    # Timeouts
    default_timeout: int = int(os.getenv("DEFAULT_TIMEOUT", "30000"))
    navigation_timeout: int = int(os.getenv("NAVIGATION_TIMEOUT", "60000"))


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Provide test configuration."""
    return AppConfig()


@pytest.fixture(scope="session")
def playwright_instance() -> Generator[Playwright, None, None]:
    """Create a Playwright instance for the test session."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright, config: AppConfig) -> Generator[Browser, None, None]:
    """Create a browser instance for the test session."""
    browser = playwright_instance.chromium.launch(
        headless=config.headless,
        slow_mo=config.slow_mo,
    )
    yield browser
    browser.close()


@pytest.fixture(scope="session")
def auth_state_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storing authentication state files."""
    with tempfile.TemporaryDirectory(prefix="openwebui_test_auth_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def context(browser: Browser, config: AppConfig) -> Generator[BrowserContext, None, None]:
    """Create a new browser context for each test."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        base_url=config.base_url,
    )
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create a new page for each test."""
    page = context.new_page()
    yield page
    page.close()


@pytest.fixture(scope="function")
def page_with_console_tracking(context: BrowserContext) -> Generator[tuple[Page, list], None, None]:
    """Create a page that tracks console errors."""
    page = context.new_page()
    console_errors: list[str] = []

    def handle_console(msg):
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", handle_console)
    yield page, console_errors
    page.close()


class AuthHelper:
    """Helper class for authentication operations."""

    def __init__(self, page: Page, config: AppConfig):
        self.page = page
        self.config = config

    def login(self, email: str | None = None, password: str | None = None) -> bool:
        """
        Log in to Open WebUI with the given credentials.
        Returns True if login was successful, False otherwise.
        """
        email = email or self.config.test_user_email
        password = password or self.config.test_user_password

        try:
            # Navigate to auth page
            self.page.goto("/auth")
            self.page.wait_for_load_state("networkidle")

            # Fill in credentials using id selectors (more reliable)
            # For LDAP mode, use username field; for signin mode, use email field
            email_field = self.page.locator('#email, #username').first
            email_field.fill(email)

            password_field = self.page.locator('#password')
            password_field.fill(password)

            # Click sign in button
            self.page.click('button[type="submit"]')

            # Wait for successful login - either redirect to home or URL changes from /auth
            self.page.wait_for_function(
                "() => !window.location.pathname.startsWith('/auth')",
                timeout=15000
            )
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def login_as_admin(self) -> bool:
        """Log in as admin user."""
        return self.login(
            email=self.config.admin_user_email,
            password=self.config.admin_user_password,
        )

    def logout(self) -> None:
        """Log out of Open WebUI."""
        # Navigate to settings and find logout option
        # This may need adjustment based on UI structure
        self.page.goto("/")
        # Click user menu and logout - adjust selectors as needed
        try:
            self.page.click('[data-testid="user-menu"]', timeout=5000)
            self.page.click('[data-testid="logout"]', timeout=5000)
        except Exception:
            # Fallback: clear cookies
            self.page.context.clear_cookies()

    def is_authenticated(self) -> bool:
        """Check if the current session is authenticated."""
        try:
            response = self.page.request.get("/api/v1/auths/")
            return response.ok
        except Exception:
            return False


@pytest.fixture(scope="function")
def auth_helper(page: Page, config: AppConfig) -> AuthHelper:
    """Provide authentication helper for tests."""
    return AuthHelper(page, config)


# ============================================================================
# Session-Scoped Authentication State
# ============================================================================
# These fixtures log in once per test session and save the browser state
# (cookies, localStorage) to avoid repeated login requests that trigger
# rate limiting.


def _perform_login(browser: Browser, config: AppConfig, email: str, password: str) -> str | None:
    """
    Perform login and return the storage state as JSON string.
    Returns None if login fails.
    """
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        base_url=config.base_url,
    )
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    
    page = context.new_page()
    auth = AuthHelper(page, config)
    
    try:
        if auth.login(email, password):
            # Save storage state to return
            state = context.storage_state()
            return state
        return None
    finally:
        page.close()
        context.close()


@pytest.fixture(scope="session")
def user_auth_state(browser: Browser, config: AppConfig, auth_state_dir: Path) -> str | None:
    """
    Log in as regular user once per session and save auth state.
    Returns path to storage state file, or None if login fails.
    """
    state = _perform_login(
        browser, config, 
        config.test_user_email, 
        config.test_user_password
    )
    if state:
        state_file = auth_state_dir / "user_auth.json"
        import json
        state_file.write_text(json.dumps(state))
        return str(state_file)
    return None


@pytest.fixture(scope="session")
def admin_auth_state(browser: Browser, config: AppConfig, auth_state_dir: Path) -> str | None:
    """
    Log in as admin user once per session and save auth state.
    Returns path to storage state file, or None if login fails.
    """
    state = _perform_login(
        browser, config,
        config.admin_user_email,
        config.admin_user_password
    )
    if state:
        state_file = auth_state_dir / "admin_auth.json"
        import json
        state_file.write_text(json.dumps(state))
        return str(state_file)
    return None


@pytest.fixture(scope="function")
def authenticated_page(
    browser: Browser, config: AppConfig, user_auth_state: str | None
) -> Generator[Page, None, None]:
    """Provide a page with an authenticated regular user session."""
    if user_auth_state is None:
        pytest.skip("Could not authenticate test user - check credentials")
    
    # Create context with saved auth state
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        base_url=config.base_url,
        storage_state=user_auth_state,
    )
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture(scope="function")
def admin_page(
    browser: Browser, config: AppConfig, admin_auth_state: str | None
) -> Generator[Page, None, None]:
    """Provide a page with an authenticated admin session."""
    if admin_auth_state is None:
        pytest.skip("Could not authenticate admin user - check credentials")

    # Create context with saved auth state
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        base_url=config.base_url,
        storage_state=admin_auth_state,
    )
    context.set_default_timeout(config.default_timeout)
    context.set_default_navigation_timeout(config.navigation_timeout)
    
    page = context.new_page()
    yield page
    page.close()
    context.close()


# ============================================================================
# Page Route Definitions
# ============================================================================

# Public pages (no authentication required)
PUBLIC_ROUTES = [
    ("/auth", "Authentication Page"),
    ("/error", "Error Page"),
]

# Pages requiring regular user authentication
USER_ROUTES = [
    ("/", "Home"),
    ("/playground", "Playground"),
    ("/playground/completions", "Completions Playground"),
    ("/workspace", "Workspace Home"),
    ("/workspace/models", "Models List"),
    ("/workspace/models/create", "Create Model"),
    ("/workspace/prompts", "Prompts List"),
    ("/workspace/prompts/create", "Create Prompt"),
    ("/workspace/tools", "Tools List"),
    ("/workspace/tools/create", "Create Tool"),
    ("/workspace/knowledge", "Knowledge Base List"),
    ("/workspace/knowledge/create", "Create Knowledge"),
    ("/workspace/functions/create", "Create Function"),
    ("/notes", "Notes List"),
    ("/notes/new", "New Note"),
]

# Pages requiring admin authentication
ADMIN_ROUTES = [
    ("/admin", "Admin Dashboard"),
    ("/admin/analytics", "Analytics"),
    ("/admin/settings", "Admin Settings"),
    ("/admin/settings/general", "General Settings"),
    ("/admin/settings/users", "User Settings"),
    ("/admin/settings/connections", "Connection Settings"),
    ("/admin/settings/models", "Model Settings"),
    ("/admin/settings/interface", "Interface Settings"),
    ("/admin/settings/audio", "Audio Settings"),
    ("/admin/settings/images", "Image Settings"),
    ("/admin/settings/pipelines", "Pipeline Settings"),
    ("/admin/settings/documents", "Document Settings"),
    ("/admin/settings/web", "Web Settings"),
    ("/admin/settings/database", "Database Settings"),
    ("/admin/users", "User Management"),
    ("/admin/evaluations", "Evaluations"),
    ("/admin/functions", "Admin Functions"),
    ("/admin/functions/create", "Create Admin Function"),
]

# Dynamic routes that require specific IDs (tested separately)
DYNAMIC_ROUTES = [
    ("/c/{id}", "Chat Conversation"),
    ("/channels/{id}", "Channel View"),
    ("/notes/{id}", "Note View"),
    ("/workspace/models/edit", "Edit Model"),  # Requires query param
    ("/workspace/prompts/edit", "Edit Prompt"),  # Requires query param
    ("/workspace/tools/edit", "Edit Tool"),  # Requires query param
    ("/workspace/knowledge/{id}", "Knowledge Detail"),
    ("/s/{id}", "Shared Content"),
    ("/watch", "Watch Page"),  # May require specific setup
]


@pytest.fixture(scope="session")
def public_routes() -> list[tuple[str, str]]:
    """Return list of public routes (path, description)."""
    return PUBLIC_ROUTES


@pytest.fixture(scope="session")
def user_routes() -> list[tuple[str, str]]:
    """Return list of user-authenticated routes (path, description)."""
    return USER_ROUTES


@pytest.fixture(scope="session")
def admin_routes() -> list[tuple[str, str]]:
    """Return list of admin-authenticated routes (path, description)."""
    return ADMIN_ROUTES
