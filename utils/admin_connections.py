"""Driving Admin Settings > Connections: the page, a saved connection's row and its dialogs."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

OPENAI_URL_PLACEHOLDER = "API Base URL"
OLLAMA_URL_PLACEHOLDER = "Enter URL (e.g. http://localhost:11434)"


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


def connection_row(settings: Locator, placeholder: str, url: str) -> Locator:
    """The row of the connection saved with `url`: its address, buttons and switch."""
    addresses = settings.get_by_placeholder(placeholder)
    expect(addresses.first).to_be_visible()
    index = addresses.evaluate_all("(inputs, url) => inputs.findIndex((i) => i.value === url)", url)
    assert index >= 0, f"no connection row reads {url}"
    return addresses.nth(index).locator("xpath=ancestor::div[.//button][1]")


def connection_dialog(page: Page, heading: str) -> Locator:
    dialog = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=heading))
    expect(dialog).to_be_visible()
    return dialog


def ollama_section(page: Page, settings: Locator) -> Locator:
    """The Ollama part of the page, switched on."""
    switch = settings.get_by_role("switch", name="Ollama API")
    if not switch.is_checked():
        with page.expect_response(is_ollama_save):
            switch.click()
    return settings.get_by_text("Manage Ollama API Connections").locator("xpath=..")
