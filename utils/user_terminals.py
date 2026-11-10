"""The New terminal and Terminal settings dialog of a terminal on a Terminals orchestrator.

Both open from the chat input's Terminal menu: New terminal from the + beside the Terminals
heading, for a person who may create terminals on a connection, and Terminal settings from the
More options of a terminal the person manages. Terminal settings lists the terminal's machines,
each with a menu of its own, and every action that stops or deletes something asks first.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

TEMPLATES = "/api/v1/terminals/templates"


def open_terminal_menu(page: Page, picked: str | None = None) -> Locator:
    """Open the chat input's Terminal menu; returns it. Its button reads the terminal `picked`
    for the chat, if one is."""
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), picked or "Terminal").click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def listed_terminal(page: Page, name: str) -> Locator:
    """The terminal `name` in the open Terminal menu."""
    return page.get_by_role("menu").get_by_role("menuitemradio", name=name, exact=True)


def new_terminal_dialog(page: Page) -> Locator:
    return page.get_by_role("dialog", name="New terminal", exact=True)


def start_new_terminal(page: Page) -> Locator:
    """Open New terminal from the Terminal menu; returns the dialog."""
    open_terminal_menu(page).get_by_role("menuitem", name="New terminal", exact=True).click()
    dialog = new_terminal_dialog(page)
    expect(dialog.get_by_label("Name", exact=True)).to_be_visible()
    return dialog


def open_terminal_settings(page: Page, name: str) -> Locator:
    """Open the Terminal settings of the terminal `name` from the Terminal menu; returns the
    dialog once it shows the terminal."""
    menu = open_terminal_menu(page)
    row = menu.get_by_role("menuitemradio", name=name, exact=True)
    expect(row).to_be_visible()
    # the row's More button shows on hover
    row.hover()
    menu.get_by_role("menuitem", name=f"More options for {name}").click()
    page.get_by_role("menuitem", name="Terminal settings", exact=True).click()
    dialog = page.get_by_role("dialog", name="Terminal settings", exact=True)
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)
    return dialog


def advanced(dialog: Locator) -> Locator:
    """The Advanced row; it reads what is set there, or Default."""
    return dialog.get_by_role("button", name=re.compile(r"^Advanced\b"))


def show_advanced(dialog: Locator) -> None:
    expect(advanced(dialog)).to_have_attribute("aria-expanded", "false")
    advanced(dialog).click()
    expect(dialog.get_by_label("Start directory", exact=True)).to_be_visible()


def hint_of(dialog: Locator, label: str) -> Locator:
    """The line under the field `label`: its hint, or what is wrong with the value."""
    field = dialog.get_by_label(label, exact=True)
    return field.locator("xpath=ancestor::div[label][1]").locator("p")


def suggestions_of(dialog: Locator, label: str) -> list[str]:
    """The values the field `label` offers to pick from."""
    return dialog.get_by_label(label, exact=True).evaluate(
        "(field) => Array.from(field.list.options, (option) => option.value)"
    )


def create_terminal(page: Page, dialog: Locator) -> str:
    """Press Create and wait until the terminal is created and the dialog is gone; returns the
    new terminal's id."""

    def creates_it(response) -> bool:
        return response.request.method == "POST" and response.url.endswith(TEMPLATES)

    with page.expect_response(creates_it) as created:
        dialog.get_by_role("button", name="Create", exact=True).click()
    assert created.value.status == 201, f"the terminal was not created: {created.value.text()}"
    expect(dialog).to_be_hidden()
    return created.value.json()["terminal_id"]


def save_terminal_settings(page: Page, dialog: Locator, terminal_id: str) -> None:
    def updates_it(response) -> bool:
        return response.request.method == "PATCH" and unquote(response.url).endswith(
            f"{TEMPLATES}/{terminal_id}"
        )

    with page.expect_response(updates_it) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()
    expect(dialog).to_be_hidden()


def machines_heading(dialog: Locator, count: int) -> Locator:
    """The Machines heading reading `count`."""
    return dialog.get_by_text(re.compile(rf"^\s*Machines\s*{count}\s*$"))


def machine_row(dialog: Locator, name: str) -> Locator:
    """The machine `name` in the Machines list, with its owner and Running or Stopped."""
    more = dialog.page.get_by_role("button", name=f"More options for {name}", exact=True)
    # the innermost box holding the machine's menu button and its state is its row
    return (
        dialog.locator("div").filter(has=more).filter(has_text=re.compile(r"Running|Stopped")).last
    )


def open_machine_menu(dialog: Locator, name: str) -> Locator:
    """Open the menu of the machine `name`; returns it."""
    more = dialog.get_by_role("button", name=f"More options for {name}", exact=True)
    more.click()
    menu = dialog.page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Delete machine", exact=True)).to_be_visible()
    return menu


def machine_actions(menu: Locator) -> list[str]:
    return [entry.strip() for entry in menu.get_by_role("menuitem").all_inner_texts()]


def confirmation(page: Page, title: str) -> Locator:
    """The question `title` an action asks before it runs."""
    asked = page.get_by_role("dialog", name=title, exact=True)
    expect(asked).to_be_visible()
    return asked
