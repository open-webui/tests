"""Nearby for #27244: a regular user's image in chat shows in the reply and leaves the checkpoint.

open-webui 0.11.0 fix `8becf9443` (#27244) made switching the shared Automatic1111 checkpoint
admin-only, because `image_generations` saved and applied any `model` a request carried. Image
generation in the chat runs through the same function, via the `generate_image` tool once the
user turns Image on under Integrations. This walks that path in the browser against a local
Automatic1111 stand-in: the image must appear in the reply and the checkpoint must not move.

Twin of unit/security/test_image_checkpoint_admin_only.py; the regular user's `model` override
itself is only reachable over the API, see integration/security/test_image_checkpoint_admin_only.py.

Discriminates: nearby layer, passes on dev bbfa876af and with the admin gate removed (the tool
sends no `model`); fails when `generate_image` stops sending the image files to the chat.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.image_engines import (
    CHECKPOINT,
    IMAGES_CONFIG,
    checkpoint_switches,
    save_image_settings,
    serve_automatic1111,
)
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def test_a_generated_image_shows_in_the_reply(
    page_for, make_user, admin, preserve, listener, upstream
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_automatic1111(listener))
    upstream.queue(
        reply.tool_call("generate_image", {"prompt": "a red square"}),
        reply.text("Here is your red square."),
    )
    page = page_for(make_user())

    page.get_by_label("Integrations").click()
    page.get_by_role("button", name="Image").click()
    page.keyboard.press("Escape")
    send(page, "draw a red square")

    expect_reply(page, "Here is your red square.")
    image = last_reply(page).get_by_role("img")
    expect(image).to_be_visible()
    expect(image).to_have_attribute("src", re.compile(r"/api/v1/files/[^/]+/content"))
    assert listener.requests_to("/sdapi/v1/txt2img"), "the image never reached Automatic1111"
    assert checkpoint_switches(listener) == []
    with admin.client() as client:
        assert client.get(IMAGES_CONFIG[0]).json()["IMAGE_GENERATION_MODEL"] == CHECKPOINT
