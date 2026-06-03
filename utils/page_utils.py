"""
Utility functions for Open WebUI testing.

This module provides helper functions for:
- Page accessibility validation
- Console error filtering
- Performance measurement
- Report generation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from playwright.sync_api import Page


class PageStatus(Enum):
    """Status of a page accessibility check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ConsoleMessage:
    """Represents a browser console message."""

    type: str  # "error", "warning", "info", "log"
    text: str
    url: str = ""
    line_number: int = 0

    @property
    def is_error(self) -> bool:
        return self.type == "error"

    @property
    def is_warning(self) -> bool:
        return self.type == "warning"


@dataclass
class PageCheckResult:
    """Comprehensive result of a page accessibility check."""

    path: str
    description: str
    status: PageStatus = PageStatus.FAILED
    http_status: int = 0
    title: str = ""
    final_url: str = ""
    load_time_ms: int = 0
    console_messages: list[ConsoleMessage] = field(default_factory=list)
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def console_errors(self) -> list[ConsoleMessage]:
        return [m for m in self.console_messages if m.is_error]

    @property
    def console_warnings(self) -> list[ConsoleMessage]:
        return [m for m in self.console_messages if m.is_warning]

    @property
    def has_errors(self) -> bool:
        return len(self.console_errors) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "description": self.description,
            "status": self.status.value,
            "http_status": self.http_status,
            "title": self.title,
            "final_url": self.final_url,
            "load_time_ms": self.load_time_ms,
            "console_error_count": len(self.console_errors),
            "console_warning_count": len(self.console_warnings),
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class ConsoleTracker:
    """Track console messages from a Playwright page."""

    def __init__(self, page: Page):
        self.page = page
        self.messages: list[ConsoleMessage] = []
        self._setup_listener()

    def _setup_listener(self):
        def handle_console(msg):
            self.messages.append(
                ConsoleMessage(
                    type=msg.type,
                    text=msg.text,
                    url=msg.location.get("url", "") if msg.location else "",
                    line_number=msg.location.get("lineNumber", 0) if msg.location else 0,
                )
            )

        self.page.on("console", handle_console)

    def clear(self):
        """Clear recorded messages."""
        self.messages = []

    def get_errors(self) -> list[ConsoleMessage]:
        """Get only error messages."""
        return [m for m in self.messages if m.is_error]

    def get_warnings(self) -> list[ConsoleMessage]:
        """Get only warning messages."""
        return [m for m in self.messages if m.is_warning]


def filter_known_errors(errors: list[ConsoleMessage]) -> list[ConsoleMessage]:
    """
    Filter out known/expected console errors that don't indicate real problems.

    Some errors are expected in certain situations (e.g., network errors when
    offline, 401 errors when not authenticated, etc.)
    """
    known_patterns = [
        "Failed to load resource: the server responded with a status of 401",
        "net::ERR_",  # Network errors (often expected)
        "favicon.ico",  # Missing favicon is not critical
    ]

    def is_known_error(error: ConsoleMessage) -> bool:
        return any(pattern in error.text for pattern in known_patterns)

    return [e for e in errors if not is_known_error(e)]


def measure_load_time(page: Page, url: str, wait_state: str = "networkidle") -> tuple[int, int]:
    """
    Measure page load time.

    Args:
        page: Playwright Page object
        url: URL to navigate to
        wait_state: Load state to wait for ("load", "domcontentloaded", "networkidle")

    Returns:
        Tuple of (http_status, load_time_ms)
    """
    import time

    start = time.time()
    response = page.goto(url)
    page.wait_for_load_state(wait_state)
    end = time.time()

    http_status = response.status if response else 0
    load_time_ms = int((end - start) * 1000)

    return http_status, load_time_ms


def check_page_structure(page: Page) -> dict[str, bool]:
    """
    Check for expected page structure elements.

    Returns:
        Dictionary with structure check results
    """
    checks = {
        "has_title": bool(page.title()),
        "has_body": page.locator("body").is_visible(),
        "has_main_content": (
            page.locator("main").count() > 0
            or page.locator("[role='main']").count() > 0
            or page.locator("#main").count() > 0
            or page.locator("#app").count() > 0
        ),
        "has_heading": page.locator("h1, h2, h3").count() > 0,
    }
    return checks


def generate_accessibility_report(
    results: list[PageCheckResult],
    report_title: str = "Page Accessibility Report",
) -> str:
    """
    Generate a text report from page check results.

    Args:
        results: List of PageCheckResult objects
        report_title: Title for the report

    Returns:
        Formatted report string
    """
    passed = [r for r in results if r.status == PageStatus.PASSED]
    failed = [r for r in results if r.status == PageStatus.FAILED]
    errors = [r for r in results if r.status == PageStatus.ERROR]
    skipped = [r for r in results if r.status == PageStatus.SKIPPED]

    lines = [
        "=" * 70,
        report_title.center(70),
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total Pages: {len(results)}",
        f"  ✅ Passed:  {len(passed)}",
        f"  ❌ Failed:  {len(failed)}",
        f"  💥 Errors:  {len(errors)}",
        f"  ⏭️  Skipped: {len(skipped)}",
        "",
        "-" * 70,
        "DETAILED RESULTS",
        "-" * 70,
    ]

    for result in results:
        status_icon = {
            PageStatus.PASSED: "✅",
            PageStatus.FAILED: "❌",
            PageStatus.ERROR: "💥",
            PageStatus.SKIPPED: "⏭️",
        }.get(result.status, "?")

        lines.append(f"{status_icon} {result.description}")
        lines.append(f"   Path: {result.path}")
        lines.append(f"   HTTP Status: {result.http_status}")
        lines.append(f"   Load Time: {result.load_time_ms}ms")

        if result.error_message:
            lines.append(f"   Error: {result.error_message}")

        if result.console_errors:
            lines.append(f"   Console Errors: {len(result.console_errors)}")
            for err in result.console_errors[:3]:  # Show first 3
                lines.append(f"      - {err.text[:80]}...")

        lines.append("")

    # Performance summary
    if passed:
        load_times = [r.load_time_ms for r in passed]
        avg_load = sum(load_times) / len(load_times)
        max_load = max(load_times)
        min_load = min(load_times)

        lines.extend(
            [
                "-" * 70,
                "PERFORMANCE SUMMARY (Passed Pages)",
                "-" * 70,
                f"  Average Load Time: {avg_load:.0f}ms",
                f"  Fastest: {min_load}ms",
                f"  Slowest: {max_load}ms",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def create_page_visitor(
    page: Page,
    before_visit: Callable[[str], None] | None = None,
    after_visit: Callable[[PageCheckResult], None] | None = None,
) -> Callable[[str, str], PageCheckResult]:
    """
    Create a page visitor function with optional callbacks.

    Args:
        page: Playwright Page object
        before_visit: Callback called before visiting each page
        after_visit: Callback called after visiting each page

    Returns:
        Function that visits a page and returns PageCheckResult
    """
    tracker = ConsoleTracker(page)

    def visit_page(path: str, description: str) -> PageCheckResult:
        tracker.clear()

        if before_visit:
            before_visit(path)

        result = PageCheckResult(path=path, description=description)

        try:
            http_status, load_time = measure_load_time(page, path)
            result.http_status = http_status
            result.load_time_ms = load_time
            result.final_url = page.url
            result.title = page.title()
            result.console_messages = tracker.messages.copy()

            if http_status >= 400:
                result.status = PageStatus.FAILED
                result.error_message = f"HTTP {http_status}"
            elif filter_known_errors(tracker.get_errors()):
                result.status = PageStatus.FAILED
                result.error_message = "Console errors detected"
            else:
                result.status = PageStatus.PASSED

        except Exception as e:
            result.status = PageStatus.ERROR
            result.error_message = str(e)

        if after_visit:
            after_visit(result)

        return result

    return visit_page
