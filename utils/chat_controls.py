"""A chat's Controls (files, valves, system prompt, advanced parameters), opened as a person does.

Since 2d7cf5ec9 they sit under the Controls entry of the chat input's Model controls menu, which
opens with Controls expanded when the chat's models bring no model controls of their own. The
menu marks every button drawn inside it as a menu item, so the sections and setting rows are
found under either role.

The helpers that open the menu take the page, meaning its main chat, or a `ChatSurface` for the
chat whose message box the menu belongs to; the open menu belongs to the page.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input
from utils.menus import late_menu_entry


def _page_and_scope(chat: Page | ChatSurface) -> tuple[Page, Page | Locator]:
    """The page, and where the chat's own buttons are: its tab panel or, for a page, the page."""
    if isinstance(chat, ChatSurface):
        return chat.page, chat.panel
    return chat, chat


def open_chat_controls(chat: Page | ChatSurface) -> Locator:
    """The open Model controls menu with its Controls expanded."""
    page, scope = _page_and_scope(chat)
    expect(chat_input(chat)).to_be_visible()
    scope.get_by_role("button", name="Model controls", exact=True).last.click()
    menu = page.get_by_role("menu")
    toggle = late_menu_entry(menu, "Controls", exact=True)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    return menu


def controls_entry(chat: Page | ChatSurface) -> Locator:
    """The Controls entry of the Model controls menu, opened when the chat input has the button.

    On a model without model controls of its own the menu holds only the Controls, so someone
    without Chat Controls gets no button there.
    """
    page, scope = _page_and_scope(chat)
    expect(chat_input(chat)).to_be_visible()
    button = scope.get_by_role("button", name="Model controls", exact=True)
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


def model_control_row(page: Page, label: str) -> Locator:
    """A model control's row in the open Model controls menu; it reads the option in use."""
    return late_menu_entry(page, label, exact=True)


def choose_model_control(chat: Page | ChatSurface, control_label: str, option_label: str) -> None:
    """Pick `option_label` of the model control `control_label` and wait until it is saved."""
    page, scope = _page_and_scope(chat)
    expect(chat_input(chat)).to_be_visible()
    scope.get_by_label("Model controls", exact=True).click()
    model_control_row(page, control_label).click()
    option = page.get_by_role("menuitemradio", name=option_label, exact=True)
    with page.expect_response(re.compile(r"/api/v1/users/user/settings/update")) as saved:
        option.click()
    assert saved.value.ok, saved.value.text()
    expect(option).to_have_attribute("aria-checked", "true")
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")
