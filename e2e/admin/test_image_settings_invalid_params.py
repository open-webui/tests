"""Regression: invalid JSON in the image Additional Parameters left Save spinning with no message.

open-webui issue #31381, fixed by PR #31382: in Admin Settings > Images, saving with text that is
no JSON object in the Additional Parameters of the OpenAI engine or of AUTOMATIC1111 kept the
Save button spinning and showed nothing. Save now reports "Invalid JSON format for Parameters"
and stays usable, so the parameters can be corrected and saved.

open-webui issue #32835: on a fresh install the engine is OpenAI with no key, and Save refused
every change with "OpenAI API Key is required." even with image generation off. A required key is
now asked for only while its feature is on: the OpenAI generation key only while Image Generation
is on, the Gemini edit key only while Image Edit is on.

Since e1d272252 the engine settings are image connections, each set up in a dialog of its own that
saves it right away, with the parameters under Advanced; a connection takes no key at all when its
auth is None, so the tab no longer asks for one. The #32835 tests keep the part that still applies:
with the feature off, a keyless connection does not stop the tab or the connection from saving.

Discriminates: passes on the dev 0d651a816 build; with the check for the two parameter fields
missing from the save handler the message never shows and Save stays disabled. The #32835 checks
pass with the fix and fail on dev 63eed95c8 (the generation key error shows while generation is off;
the empty Gemini edit key blocks the save and the new base URL is not stored). On the dev
00bd84944 build they pass against the connection dialog.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.image_engines import (
    IMAGES_CONFIG,
    image_connection,
    image_settings,
    save_image_settings,
    serve_automatic1111,
)
from utils.admin_settings import save, submit
from utils.image_settings import (
    PARAMS_PLACEHOLDER,
    configure_connection,
    open_image_settings,
    save_connection,
    show_advanced,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

OPENAI_PARAMS = {"quality": "hd"}
AUTOMATIC1111_PARAMS = {"cfg_scale": 7}


def save_refused_then_fixed(page: Page, dialog: Locator, fixed: dict) -> None:
    show_advanced(dialog)
    params = dialog.get_by_placeholder(PARAMS_PLACEHOLDER)
    save = dialog.get_by_role("button", name="Save", exact=True)
    params.fill("{quality: hd")
    save.click()
    expect(
        page.get_by_text("Invalid JSON format for Parameters"),
        "#31381: no message for the invalid parameters",
    ).to_be_visible()
    expect(save, "#31381: Save stayed disabled after the error").to_be_enabled()
    params.fill(json.dumps(fixed))
    save_connection(page, dialog)


def saved_connection(admin, connection_id: str) -> dict:
    with admin.client() as client:
        connections = client.get(IMAGES_CONFIG[0]).json()["IMAGE_CONNECTIONS"]
    [connection] = [entry for entry in connections if entry["id"] == connection_id]
    return connection


def test_invalid_openai_parameters_are_reported_and_save_works_once_fixed(
    page_for, make_user, admin, preserve
):
    preserve(IMAGES_CONFIG)
    openai = image_connection(
        "openai", "openai", "http://127.0.0.1:9/v1", "sk-image-test", models=("gpt-image-1",)
    )
    with admin.client() as client:
        save_image_settings(client, **image_settings(openai))
    page = page_for(make_user(role="admin"))
    settings = open_image_settings(page)
    dialog = configure_connection(page, settings, "openai")

    save_refused_then_fixed(page, dialog, OPENAI_PARAMS)

    assert saved_connection(admin, "openai")["config"]["params"] == OPENAI_PARAMS


def test_invalid_automatic1111_parameters_are_reported_and_save_works_once_fixed(
    page_for, make_user, admin, preserve, listener
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_automatic1111(listener))
    page = page_for(make_user(role="admin"))
    settings = open_image_settings(page)
    dialog = configure_connection(page, settings, "automatic1111")

    save_refused_then_fixed(page, dialog, AUTOMATIC1111_PARAMS)

    assert saved_connection(admin, "automatic1111")["config"]["params"] == AUTOMATIC1111_PARAMS


def test_saving_with_image_generation_off_asks_for_no_key(page_for, make_user, admin, preserve):
    preserve(IMAGES_CONFIG)
    keyless = image_connection("openai", "openai", "http://127.0.0.1:9/v1", models=("gpt-image-1",))
    with admin.client() as client:
        save_image_settings(client, **image_settings(keyless, generation=False, edit=False))
    page = page_for(make_user(role="admin"))
    settings = open_image_settings(page)

    # the connections only show while image generation or editing is on
    add = settings.get_by_role("button", name="Add Connection")
    expect(add).to_have_count(0)
    with page.expect_response(lambda response: "/images/config/update" in response.url) as saved:
        submit(settings)
    assert saved.value.ok, (
        f"#32835: saving with image generation off was refused: {saved.value.text()}"
    )
    expect(
        page.get_by_text("Settings saved successfully!").first,
        "#32835: saving with image generation off still asked for its API key",
    ).to_be_visible()

    image_generation = settings.get_by_role("switch", name="Image Generation")
    image_generation.click()
    expect(image_generation).to_be_checked()
    # turning it on reveals the connections
    expect(add).to_be_visible()
    expect(settings.get_by_role("button", name="Configure")).to_have_count(1)


def test_saving_with_image_edit_off_asks_for_no_edit_key(page_for, make_user, admin, preserve):
    preserve(IMAGES_CONFIG)
    generator = image_connection(
        "openai", "openai", "http://127.0.0.1:9/v1", "sk-image-test", models=("gpt-image-1",)
    )
    editor = image_connection(
        "gemini", "gemini", "http://127.0.0.1:9/v1beta", models=("gemini-2.5-flash-image",)
    )
    with admin.client() as client:
        save_image_settings(client, **image_settings(generator, editor, edit=False))
    page = page_for(make_user(role="admin"))
    settings = open_image_settings(page)
    dialog = configure_connection(page, settings, "gemini")

    dialog.get_by_label("URL", exact=True).fill("http://127.0.0.1:9/v1beta/edited")
    save_connection(page, dialog)
    stored = saved_connection(admin, "gemini")
    assert stored["url"] == "http://127.0.0.1:9/v1beta/edited", (
        f"#32835: the keyless Gemini edit connection was not saved: {stored}"
    )
    save(settings)
