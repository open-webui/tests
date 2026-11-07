"""Driving the chat page the way a person does: type, press Enter, read what appears."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

REPLY_TIMEOUT_MS = 30_000


def chat_input(page: Page) -> Locator:
    return page.locator("#chat-input")


def send(page: Page, text: str) -> None:
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    chat_input(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def link_dialog(page: Page) -> Locator:
    """The dialog a chat link asks in before it loads its `load-url` or `youtube` page or calls."""
    return page.get_by_role("dialog", name="Open link")


def conversation(page: Page) -> Locator:
    """The open chat's messages, never the preview a hovered sidebar chat shows."""
    return page.get_by_role("main").get_by_label("Chat Conversation")


def replies(page: Page) -> Locator:
    return conversation(page).locator(".chat-assistant")


def last_reply(page: Page) -> Locator:
    return replies(page).last


def listed_source(page: Page, name: str) -> Locator:
    """An entry of an opened source list, which reads out its number before the name."""
    label = re.compile(rf"^View source: \d+ {re.escape(name)}$")
    return conversation(page).get_by_role("button", name=label)


def expect_reply(page: Page, text: str) -> None:
    expect(last_reply(page)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def side_chat_tab(page: Page, question: str) -> Locator:
    """The tab of the side chat opened with `question`, which it takes as its title once saved."""
    return page.get_by_role("tab", name=question, exact=True)


def side_chat_panel(page: Page, question: str) -> Locator:
    return page.get_by_role("tabpanel", name=question, exact=True)


def chat_tab_bar(page: Page) -> Locator:
    """The bar over the open chat: its tabs, the "Chat actions" menu and who else has it open."""
    # the bar has no role or name of its own, only the tab list inside it
    return (
        page.get_by_role("main")
        .locator("[data-panel-tab-bar]")
        .filter(has=page.get_by_role("tablist", name="Tabs"))
    )


def stop_button(page: Page) -> Locator:
    return page.get_by_role("button", name="Stop")


def regenerate_buttons(page: Page) -> Locator:
    """The Regenerate button under each finished reply."""
    return conversation(page).get_by_label("Regenerate", exact=True)


def typing_cursor(reply: Locator) -> Locator:
    """The blinking cursor a reply shows while it is still being written."""
    return reply.locator(".animate-cursor-pulse")


def close_menu(page: Page) -> None:
    """Close the open menu and wait until it is gone, so the next click or key reaches the page."""
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)


def menu_entries(menu: Locator) -> Locator:
    """Every entry of a chat input menu: plain items and the ones that switch on and off."""
    return (
        menu.get_by_role("menuitem")
        .or_(menu.get_by_role("menuitemcheckbox"))
        .or_(menu.get_by_role("menuitemradio"))
    )


def _open_composer_menu(page: Page, button: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name=button, exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu_entries(menu).first).to_be_visible()
    return menu


def open_input_menu(page: Page) -> Locator:
    """The chat input's More menu (uploads, attachments, tool permissions), once it shows."""
    return _open_composer_menu(page, "More")


def open_integrations_menu(page: Page) -> Locator:
    """The chat input's Integrations menu (tools, skills, filters, web search), once it shows."""
    return _open_composer_menu(page, "Integrations")


def integration_row(menu: Locator, name: str, role: str = "menuitemcheckbox") -> Locator:
    """The Integrations menu row of `name`: the innermost block holding its entry and the buttons
    beside it (Valves, Disconnect OAuth)."""
    entry = menu.page.get_by_role(role, name=name)
    return menu.locator("div").filter(has=entry).last


def set_tool_approval_mode(page: Page, mode: str) -> None:
    """Pick how this chat's tool calls are approved, under the More menu's Tool Permissions."""
    menu = open_input_menu(page)
    permissions = menu.get_by_role("menuitem", name="Tool Permissions")
    permissions.click()
    menu.get_by_role("menuitemradio", name=mode).click()
    expect(permissions).to_contain_text(mode)
    close_menu(page)


def turn_on_integration(page: Page, name: str) -> None:
    """Switch `name` on for the next message from the chat's Integrations menu."""
    switch = open_integrations_menu(page).get_by_role("menuitemcheckbox", name=name, exact=True)
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    close_menu(page)


def open_add_context(page: Page) -> Locator:
    """The More menu's Add Context list (webpages, files, knowledge, notes, chats), once it shows:
    since dev 834c47ee0 they sit there, a level down from the menu's first page."""
    menu = open_input_menu(page)
    menu.get_by_role("menuitem", name="Add Context").click()
    expect(menu.get_by_role("menuitem", name="Upload Files")).to_have_count(0)
    return menu
