"""Driving the Add/Edit Terminal Connection dialog.

An admin opens it from Admin Settings > Integrations and a person from their own Settings >
Integrations. Save first checks that the server answers (the instance asks it for an admin
connection, the browser for a personal one) and says so in the dialog when it does not.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from utils.tooltips import tooltip_button

ADMIN_CONNECTIONS = "/api/v1/configs/terminal_servers"
PERSONAL_SETTINGS = "/api/v1/users/user/settings/update"


def admin_terminal_section(page: Page) -> Locator:
    """Admin Settings > Integrations, opened afresh; returns its Terminal section."""
    page.goto("/admin/settings/integrations")
    heading = page.get_by_role("dialog").get_by_text("Open Terminal", exact=True)
    expect(heading).to_be_visible()
    return heading.locator("xpath=../..")


def terminal_dialog(page: Page, title: str = "Add Terminal Connection") -> Locator:
    dialog = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=title))
    expect(dialog).to_be_visible()
    return dialog


def add_admin_terminal(page: Page) -> Locator:
    """Open Add Terminal Connection from the admin's Terminal section; returns the dialog."""
    tooltip_button(admin_terminal_section(page), "Add Connection").click()
    return terminal_dialog(page)


def fill_terminal(dialog: Locator, name: str, url: str, key: str | None = None) -> None:
    dialog.get_by_label("Name", exact=True).fill(name)
    dialog.get_by_label("URL", exact=True).fill(url)
    if key is not None:
        dialog.get_by_placeholder("API Key").fill(key)


def choose_terminal_auth(dialog: Locator, label: str) -> None:
    dialog.get_by_label("Auth", exact=True).select_option(label=label)


def show_advanced(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Advanced", exact=True).click()
    expect(dialog.get_by_label("User Shell Tools")).to_be_visible()


def save_terminal(page: Page, dialog: Locator, saved_to: str = ADMIN_CONNECTIONS) -> None:
    """Press Save and wait until the connection is stored (`saved_to`) and the dialog is gone."""

    def stores_it(response) -> bool:
        return response.request.method == "POST" and response.url.endswith(saved_to)

    save = dialog.get_by_role("button", name="Save", exact=True)
    refused = _field_the_browser_refuses(dialog)
    if refused:
        save.click()
        expect(dialog).to_be_visible()
        raise AssertionError(f"Save did nothing; the browser points at the field {refused}")
    try:
        with page.expect_response(stores_it) as saved:
            save.click()
    except PlaywrightTimeoutError:
        problem = dialog.get_by_role("alert")
        shown = problem.inner_text() if problem.count() else "nothing"
        raise AssertionError(
            f"the terminal connection was not saved; the dialog says: {shown}"
        ) from None
    assert saved.value.ok, saved.value.text()
    expect(dialog).to_be_hidden()


def _field_the_browser_refuses(dialog: Locator) -> str | None:
    """The field the browser will not submit, with the message it shows in a bubble beside it."""
    # the bubble is browser chrome, not in the page; :invalid is the only way to the field
    return dialog.locator("form").evaluate(
        """form => {
            const field = form.querySelector('input:invalid, select:invalid, textarea:invalid');
            return field ? `"${field.placeholder || field.id}": ${field.validationMessage}` : null;
        }"""
    )
