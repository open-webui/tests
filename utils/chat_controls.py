"""A chat's Controls (files, valves, system prompt, advanced parameters), opened as a person does.

Since 2d7cf5ec9 they sit under the Controls entry of the chat input's Model controls menu, which
opens with Controls expanded when the chat's models bring no model controls of their own. The
menu marks every button drawn inside it as a menu item, so the sections and setting rows are
found under either role.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input
from utils.menus import late_menu_entry


def open_chat_controls(page: Page) -> Locator:
    """The open Model controls menu with its Controls expanded."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Model controls", exact=True).last.click()
    menu = page.get_by_role("menu")
    toggle = late_menu_entry(menu, "Controls", exact=True)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    return menu


def controls_entry(page: Page) -> Locator:
    """The Controls entry of the Model controls menu, opened when the chat input has the button.

    On a model without model controls of its own the menu holds only the Controls, so someone
    without Chat Controls gets no button there.
    """
    expect(chat_input(page)).to_be_visible()
    button = page.get_by_role("button", name="Model controls", exact=True)
    if button.count():
        button.last.click()
    return late_menu_entry(page.get_by_role("menu"), "Controls", exact=True)


def control_button(panel: Locator, label: str) -> Locator:
    """A section title or a setting row's button inside the Controls, by its label."""
    return late_menu_entry(panel, label, exact=True)


def open_section(panel: Locator, title: str) -> Locator:
    """Expand the Controls section `title` if it reads closed."""
    toggle = control_button(panel, title)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    return panel
