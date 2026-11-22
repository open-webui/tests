"""Driving Admin Settings > Connections: the page, a saved connection's row and its dialogs."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect


def is_openai_save(response) -> bool:
    return "/openai/config/update" in response.url


def is_ollama_save(response) -> bool:
    return "/ollama/config/update" in response.url


def open_admin_connections(page: Page) -> Locator:
    page.goto("/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Connections", exact=True)).to_be_visible()
    expect(settings.get_by_role("switch", name="OpenAI API")).to_be_visible()
    return settings


def connection_row(settings: Locator, url: str) -> Locator:
    """The row of the connection saved with `url`: its address button, buttons and switch."""
    expect(settings.get_by_role("button", name=url, exact=True)).to_be_visible()
    address = settings.page.get_by_role("button", name=url, exact=True)
    configure = settings.page.get_by_role("button", name="Configure", exact=True)
    return settings.locator("div").filter(has=address).filter(has=configure).last


def connection_dialog(page: Page, heading: str) -> Locator:
    dialog = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=heading))
    expect(dialog).to_be_visible()
    return dialog
