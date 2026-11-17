"""Driving Admin Settings > Images: the tab, its image connections and the connection dialog."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

PARAMS_PLACEHOLDER = "Enter additional parameters in JSON format"
SIZE_PLACEHOLDER = "Enter Image Size (e.g. 512x512)"
STEPS_PLACEHOLDER = "Enter Number of Steps (e.g. 50)"


def open_image_settings(page: Page) -> Locator:
    """Admin Settings > Images, freshly loaded from the server."""
    page.goto("/admin/settings/images")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Image Generation")).to_be_visible()
    return settings


def connection_dialog(page: Page) -> Locator:
    heading = page.get_by_text("Add Connection", exact=True).or_(
        page.get_by_text("Edit Connection", exact=True)
    )
    dialog = page.get_by_role("dialog").filter(has=heading).last
    expect(dialog.get_by_label("Connection Type")).to_be_visible()
    return dialog


def add_connection(page: Page, settings: Locator, connection_type: str) -> Locator:
    """Open Add Connection and pick `connection_type` (its label, as "ComfyUI")."""
    settings.get_by_role("button", name="Add Connection").click()
    dialog = connection_dialog(page)
    dialog.get_by_label("Connection Type").select_option(label=connection_type)
    return dialog


def configure_connection(page: Page, settings: Locator, name: str) -> Locator:
    """Open the dialog of the saved connection called `name`."""
    row = settings.get_by_role("button", name="Configure").locator(
        "xpath=ancestor::div[.//label][1]"
    )
    row.filter(has_text=name).get_by_role("button", name="Configure").click()
    dialog = connection_dialog(page)
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)
    return dialog


def verify_connection(page: Page, dialog: Locator) -> None:
    dialog.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()


def add_model_id(dialog: Locator, model_id: str) -> None:
    dialog.get_by_placeholder("Add a model ID").fill(model_id)
    dialog.get_by_role("button", name="Add", exact=True).click()
    expect(dialog.get_by_role("button", name=f"Remove {model_id} from list.")).to_be_visible()


def show_advanced(dialog: Locator) -> None:
    """Open the dialog's Advanced settings: size, steps, API version, parameters."""
    if not dialog.get_by_placeholder(SIZE_PLACEHOLDER).first.is_visible():
        dialog.get_by_role("button", name="Advanced").click()
    expect(dialog.get_by_placeholder(SIZE_PLACEHOLDER).first).to_be_visible()


def save_connection(page: Page, dialog: Locator) -> None:
    """Save the dialog; connections are stored right away."""
    with page.expect_response(lambda response: "/images/config/update" in response.url) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, f"saving the image connection failed: {saved.value.text()}"
    expect(dialog).to_be_hidden()
