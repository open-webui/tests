"""Driving the Add/Edit Terminal Connection dialog.

An admin opens it from Admin Settings > Integrations and a person from their own Settings >
Integrations. Save first checks that the server answers (the instance asks it for an admin
connection, the browser for a personal one) and says so in the dialog when it does not.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from utils.tool_servers import connection_row
from utils.user_menu import open_settings

ADMIN_INTEGRATIONS = "/admin/settings/integrations"
ADMIN_CONNECTIONS = "/api/v1/configs/terminal_servers"
PERSONAL_SETTINGS = "/api/v1/users/user/settings/update"


def shown_admin_terminal_section(page: Page) -> Locator:
    """The Terminal section of Admin Settings > Integrations on a page opened there with
    `page_for(account, path=ADMIN_INTEGRATIONS)`, as it stands."""
    heading = page.get_by_role("dialog").get_by_text("Open Terminal", exact=True)
    expect(heading).to_be_visible()
    return heading.locator("xpath=../..")


def terminal_dialog(page: Page, title: str = "Add Terminal Connection") -> Locator:
    dialog = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=title))
    expect(dialog).to_be_visible()
    return dialog


def personal_terminal_section(page: Page) -> Locator:
    """A person's own Settings > Integrations, opened from the user menu; returns its Open
    Terminal section."""
    open_settings(page)
    settings = page.get_by_role("dialog")
    # an admin's settings list the personal tab before the admin tab of the same name
    settings.get_by_role("tab", name="Integrations", exact=True).first.click()
    heading = settings.get_by_role("heading", name="Open Terminal", exact=True)
    expect(heading).to_be_visible()
    return heading.locator("xpath=../../..")


def add_terminal(section: Locator) -> Locator:
    """Open Add Terminal Connection from the admin's or a person's own terminal section, as it
    stands; returns the dialog."""
    section.get_by_role("button", name="Add Connection", exact=True).click()
    return terminal_dialog(section.page)


def edit_terminal(section: Locator, name: str) -> Locator:
    """Open Edit Terminal Connection of the connection the section lists as `name`; returns the
    dialog."""
    connection_row(section, name).get_by_role("button", name="Configure", exact=True).click()
    return terminal_dialog(section.page, "Edit Terminal Connection")


def leave_settings(page: Page) -> None:
    """Close the settings with Back; the chat page under them was never reloaded."""
    page.get_by_role("dialog").get_by_role("button", name="Back", exact=True).click()
    expect(page.get_by_role("dialog")).to_have_count(0)


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


def verify_terminal(dialog: Locator) -> None:
    """Press Verify Connection and wait until the check has come back."""
    button = dialog.get_by_role("button", name="Verify Connection", exact=True)
    button.click()
    # the button is off for as long as the check runs
    expect(button).to_be_enabled()


def terminal_problem(dialog: Locator) -> Locator:
    """What the dialog says went wrong with the last check or save."""
    return dialog.get_by_role("alert")


def delete_terminal(page: Page, dialog: Locator, confirm: bool = True) -> None:
    """Press Delete in the Edit dialog and answer its question."""
    dialog.get_by_role("button", name="Delete", exact=True).click()
    asked = page.get_by_role("dialog").filter(has_text="Delete this connection?")
    expect(asked).to_be_visible()
    asked.get_by_role("button", name="Delete" if confirm else "Cancel", exact=True).click()
    expect(asked).to_be_hidden()


def save_terminal(page: Page, dialog: Locator, saved_to: str = ADMIN_CONNECTIONS) -> None:
    """Press Save and wait until the connection is stored (`saved_to`) and the dialog is gone."""

    def stores_it(response) -> bool:
        return response.request.method == "POST" and response.url.endswith(saved_to)

    save = dialog.get_by_role("button", name="Save", exact=True)
    refused = field_the_browser_refuses(dialog)
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


def field_the_browser_refuses(dialog: Locator) -> str | None:
    """The field the browser will not submit, with the message it shows in a bubble beside it."""
    # the bubble is browser chrome, not in the page; :invalid is the only way to the field
    return dialog.locator("form").evaluate(
        """form => {
            const field = form.querySelector('input:invalid, select:invalid, textarea:invalid');
            return field ? `"${field.placeholder || field.id}": ${field.validationMessage}` : null;
        }"""
    )
