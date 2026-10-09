"""Driving tool server connections: Admin Settings > Integrations and the chat's Tools menu."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from utils.access_control import choose_visibility
from utils.chat_ui import chat_input, conversation
from utils.tooltips import tooltip_button


def open_admin_integrations(page: Page) -> Locator:
    page.goto("/admin/settings/integrations")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("External Tool Servers", exact=True)).to_be_visible()
    return settings


def open_add_connection(page: Page) -> Locator:
    """The Add Connection dialog of External Tool Servers, opened from the admin's settings."""
    tooltip_button(open_admin_integrations(page), "Add Connection").click()
    form = connection_form(page, "Add Connection")
    expect(form).to_be_visible()
    return form


def connection_form(page: Page, heading: str) -> Locator:
    return page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=heading))


def switch_to_mcp(form: Locator) -> None:
    form.get_by_role("button", name="OpenAPI").click()
    expect(form.get_by_role("button", name="MCP")).to_be_visible()


def choose_auth(form: Locator, label: str) -> None:
    session = form.page.get_by_role("option", name="Session")
    auth = form.get_by_role("combobox").filter(has=session)
    auth.select_option(label=label)


def verify(form: Locator, label: str = "Verify Connection") -> None:
    form.get_by_role("button", name=label).click()


def open_access_control(page: Page, form: Locator) -> Locator:
    form.get_by_role("button", name="Access Control").click()
    access = page.get_by_role("dialog").filter(has_text="Access List")
    expect(access).to_be_visible()
    return access


def make_public(page: Page, form: Locator) -> None:
    access = open_access_control(page, form)
    choose_visibility(access, "Public")
    expect(access.get_by_text("Accessible to all users")).to_be_visible()
    page.keyboard.press("Escape")
    expect(access).to_have_count(0)


def share_with_group(page: Page, form: Locator, group_name: str) -> None:
    access = open_access_control(page, form)
    access.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has=page.get_by_role("button", name="Add"))
    picker.get_by_placeholder("Search").fill(group_name)
    picker.get_by_role("button", name=group_name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(access.get_by_text(group_name)).to_be_visible()
    page.keyboard.press("Escape")
    expect(access).to_have_count(0)


def save(page: Page, form: Locator) -> None:
    form.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Connections saved successfully").last).to_be_visible()


def connection_row(settings: Locator, name: str) -> Locator:
    """The listed connection named `name`: the innermost block holding it and its switch."""
    blocks = settings.locator("div").filter(has_text=name)
    return blocks.filter(has=settings.page.get_by_role("switch")).last


def open_tools_menu(page: Page) -> Locator:
    """The chat's Integrations > Tools list, on a freshly loaded chat page."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_placeholder("Search tools")).to_be_visible()
    return menu


def pick_tool(page: Page, name: str) -> None:
    """Turn on the tool or tool server `name` for the next message of a new chat."""
    menu = open_tools_menu(page)
    menu.get_by_role("button", name=name).click()
    expect(menu.get_by_role("button", name=name)).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("Escape")


def tool_output(page: Page, call_name: str) -> Locator:
    """Open the reply's call of `call_name` and return its Output section."""
    conversation(page).get_by_text(f"View Result from {call_name}").click()
    return conversation(page).get_by_text("Output", exact=True).last.locator("..")
