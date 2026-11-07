"""Regression: on a phone, opening or closing a menu brought up the on-screen keyboard.

Fix `e797a4480` (PR #31657, issue #31656). Closing the Integrations menu put the focus back in
the message box, and opening any dropdown moved the focus into its list; in the automation
dialog that list is portaled outside the dialog, so the dialog's focus trap pulled the focus back
into one of its text fields. A focused text field is what raises a phone's keyboard. On a narrow
screen neither happens now, and on a desktop the Integrations menu still hands the focus back to
the message box.

Discriminates: passes on the 015dbc861 build; with `e797a4480` reverted in a frontend copy the
two phone tests go red (the message box, then the automation's text field, holds the focus) and
the desktop test stays green.
Retargeted for dev 834c47ee0, which moved the integrations into the + (More) menu: that menu hands
the focus back to the message box on a phone too, so the phone test is red on dev f659c0d12 until
the + menu skips it on a narrow screen as the Integrations menu did.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from utils.chat_ui import chat_input, open_integrations_menu

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PHONE = {"width": 390, "height": 844}

# the tag and name of the focused element when a keyboard would come up for it, else null
FOCUSED_TEXT_FIELD = """() => {
    const element = document.activeElement;
    if (!element) return null;
    const typesText = element.isContentEditable || ['INPUT', 'TEXTAREA'].includes(element.tagName);
    const name = element.id || element.getAttribute('placeholder') || '';
    return typesText ? `${element.tagName} ${name}` : null;
}"""


# iOS Safari does not focus a button that is tapped, Chromium does unless mousedown is cancelled
TAPS_LEAVE_BUTTONS_UNFOCUSED = """document.addEventListener('mousedown', (event) => {
    if (event.target.closest('button, [role="button"]')) event.preventDefault();
}, true);"""


def on_a_phone(page: Page) -> Page:
    page.set_viewport_size(PHONE)
    page.add_init_script(TAPS_LEAVE_BUTTONS_UNFOCUSED)
    page.reload()
    expect(chat_input(page)).to_be_visible()
    # the page focuses the message box once after load; let that land before a test blurs it
    expect(chat_input(page)).to_be_focused()
    return page


def focused_text_field(page: Page) -> str | None:
    return page.evaluate(FOCUSED_TEXT_FIELD)


def dismiss_the_keyboard(page: Page) -> None:
    page.evaluate("() => document.activeElement?.blur()")
    assert focused_text_field(page) is None


def open_and_close_the_integrations_menu(page: Page) -> None:
    open_integrations_menu(page)
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    # a tap on the empty top of the page, outside the menu
    page.mouse.click(page.viewport_size["width"] // 2, 80)
    expect(menu).to_be_hidden()


def test_closing_the_integrations_menu_on_a_phone_leaves_the_message_box_alone(page_for, make_user):
    page = on_a_phone(page_for(make_user()))
    dismiss_the_keyboard(page)

    open_and_close_the_integrations_menu(page)

    expect(chat_input(page)).not_to_be_focused()
    assert focused_text_field(page) is None, "closing the menu focused a text field (#31657)"


def test_closing_the_integrations_menu_on_a_desktop_focuses_the_message_box(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()

    open_and_close_the_integrations_menu(page)

    expect(chat_input(page)).to_be_focused()


def test_opening_a_dropdown_in_the_automation_dialog_on_a_phone_focuses_no_text_field(
    page_for, make_user
):
    # plain accounts need the automations permission, an admin has it
    page = on_a_phone(page_for(make_user(role="admin")))
    page.goto("/automations")
    page.get_by_role("button", name="Create", exact=True).click()
    creating = page.get_by_role("dialog")
    expect(creating.get_by_role("textbox", name="Automation title")).to_be_visible()
    dismiss_the_keyboard(page)

    creating.get_by_role("button", name="Daily").first.click()

    expect(page.get_by_role("combobox")).to_be_visible()
    assert focused_text_field(page) is None, "opening the dropdown focused a text field (#31657)"
